import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy.linalg import svd
from tqdm import tqdm
###########################################################
# Potts operators
###########################################################

def potts_ops():
    w = np.exp(2j * np.pi / 3)

    Z = np.array([[1, 0, 0],
                  [0, w, 0],
                  [0, 0, w**2]], dtype=complex)

    X = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]], dtype=complex)

    I = np.eye(3, dtype=complex)
    return Z, X, I


###########################################################
# Bond Hamiltonians
###########################################################

def potts_bond_hamiltonian(J, f, L):
    Z, X, I = potts_ops()
    Xh = X.conj().T

    H_bond = []

    for i in range(L - 1):
        H = np.zeros((9, 9), dtype=complex)

        H += -J * (np.kron(Z.conj().T, Z) + np.kron(Z, Z.conj().T))

        wl = 1.0 if i == 0 else 0.5
        wr = 1.0 if i == L - 2 else 0.5

        H += -wl * f * (np.kron(X, I) + np.kron(Xh, I))
        H += -wr * f * (np.kron(I, X) + np.kron(I, Xh))

        H_bond.append(H.reshape(3, 3, 3, 3))

    return H_bond


def potts_expH(J, f, L, dt, imag=False):
    H = potts_bond_hamiltonian(J, f, L)
    gates = []

    for h in H:
        h = h.reshape(9, 9)
        if imag:
            U = expm(-dt * h)
        else:
            U = expm(-1j * dt * h)

        # indices: (s_out, t_out, s_in, t_in)
        gates.append(U.reshape(3, 3, 3, 3))

    return gates


###########################################################
# Initial MPS
###########################################################

def initialMPS(L, d):
    G = []
    l = []

    for _ in range(L):
        G.append(np.zeros((d, 1, 1), dtype=complex))
        l.append(np.ones(1, dtype=complex))

    l.append(np.ones(1, dtype=complex))

    for i in range(L):
        G[i][0, 0, 0] = 1.0

    return l, G


def safe_inv(x, eps=1e-14):
    y = np.zeros_like(x, dtype=complex)
    mask = np.abs(x) > eps
    y[mask] = 1.0 / x[mask]
    return y


###########################################################
# Two-site TEBD
###########################################################

def tebd_step(l, G, gates, chi_max=50):
    L = len(G)

    for parity in [0, 1]:
        for i in range(parity, L - 1, 2):
            d = G[i].shape[0]
            chi0 = G[i].shape[1]
            chi1 = G[i].shape[2]
            chi2 = G[i + 1].shape[2]

            l0 = l[i]
            l1 = l[i + 1]
            l2 = l[i + 2]

            l0inv = safe_inv(l0)
            l2inv = safe_inv(l2)

            theta = np.tensordot(np.diag(l0), G[i], axes=(1, 1))
            theta = np.tensordot(theta, np.diag(l1), axes=(2, 0))
            theta = np.tensordot(theta, G[i + 1], axes=(2, 1))
            theta = np.tensordot(theta, np.diag(l2), axes=(3, 0))
            # shape: (chi0, d, d, chi2)

            # correct gate application
            theta = np.tensordot(gates[i], theta, axes=([2, 3], [1, 2]))
            theta = np.transpose(theta, (2, 0, 1, 3))
            # shape: (chi0, d, d, chi2)

            theta = theta.reshape(chi0 * d, d * chi2)

            U, S, Vh = svd(theta, full_matrices=False)

            keep = min(len(S), chi_max)

            U = U[:, :keep]
            S = S[:keep]
            Vh = Vh[:keep, :]

            nrm = np.linalg.norm(S)
            if nrm > 0:
                S = S / nrm

            l[i + 1] = S

            U = U.reshape(chi0, d, keep)
            U = np.tensordot(np.diag(l0inv), U, axes=(1, 0))
            U = np.transpose(U, (1, 0, 2))
            G[i] = U

            V = Vh.reshape(keep, d, chi2)
            V = np.transpose(V, (1, 0, 2))
            V = np.tensordot(V, np.diag(l2inv), axes=(2, 0))
            G[i + 1] = V


###########################################################
# Dense state reconstruction for overlap
# reliable for moderate L, not for huge L
###########################################################

def mps_to_state(l, G):
    psi = np.array([[1.0 + 0.0j]])  # (1, chi_left)

    for i in range(len(G)):
        psi = np.tensordot(psi, np.diag(l[i]), axes=(1, 0))   # (..., chi_i)
        psi = np.tensordot(psi, G[i], axes=(1, 1))            # (..., d, chi_{i+1})

    psi = np.tensordot(psi, np.diag(l[len(G)]), axes=(psi.ndim - 1, 0))
    psi = np.squeeze(psi, axis=(-1, 0))
    return psi.reshape(-1)


# def overlap(l1, G1, l2, G2):
#     psi1 = mps_to_state(l1, G1)
#     psi2 = mps_to_state(l2, G2)
#     return np.vdot(psi1, psi2)

def overlap(l1, G1, l2, G2):
    E = np.array([[1.0 + 0.0j]])

    L = len(G1)

    for i in range(L):
        A = np.tensordot(np.diag(l1[i]), G1[i], axes=(1, 1))          # (chiL1,d,chiR1)
        B = np.tensordot(np.diag(l2[i]), G2[i], axes=(1, 1))          # (chiL2,d,chiR2)

        E = np.tensordot(E, A.conj(), axes=(0, 0))                    # (chiL2,d,chiR1)
        E = np.tensordot(E, B, axes=([0, 1], [0, 1]))                 # (chiR1,chiR2)

    E = np.tensordot(E, np.diag(l1[L]).conj(), axes=(0, 0))
    E = np.tensordot(E, np.diag(l2[L]), axes=([0, 1], [0, 1]))

    return E.reshape(())
###########################################################
# Main DQPT simulation
###########################################################

L = 50
J = 1.0

f0 = 10.0
f1 = 0.0

dt = 0.02
steps = 200

d = 3
chi_max = 50

###########################################################
# prepare ground state (imaginary time)
###########################################################

l, G = initialMPS(L, d)

gates0 = potts_expH(J, f0, L, 0.01, imag=True)

print("Preparing ground state with imaginary time evolution...")
# use tqdm for progress bar
for _ in tqdm(range(1000), desc="Imaginary time evolution"):
    tebd_step(l, G, gates0, chi_max=chi_max)

psi0_l = [x.copy() for x in l]
psi0_G = [x.copy() for x in G]

###########################################################
# quench evolution
###########################################################

gates1 = potts_expH(J, f1, L, dt, imag=False)

times = []
rate = []

# use tqdm for progress bar
print("Starting quench evolution...")
for step in tqdm(range(steps), desc="Quench evolution"):
    tebd_step(l, G, gates1, chi_max=chi_max)

    Gt = overlap(psi0_l, psi0_G, l, G)
    I = -(1 / L) * np.log(np.abs(Gt) ** 2)

    times.append((step + 1) * dt * J)
    rate.append(I.real)

###########################################################
# Save data
###########################################################

data = np.column_stack((times, rate))
np.savetxt(
    "dqpt_potts_L%d_f0%.2f_f1%.2f.dat" % (L, f0, f1),
    data,
    header="Jt   I(t)"
)

###########################################################
# Plot rate function
###########################################################

plt.figure(figsize=(6, 4))
plt.plot(times, rate, 'o-')
plt.xlabel("J t")
plt.ylabel("I(t)")
plt.title("DQPT rate function (3-state Potts chain)")
plt.tight_layout()
plt.show()