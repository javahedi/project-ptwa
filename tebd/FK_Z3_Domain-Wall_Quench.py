import numpy as np
from scipy.linalg import expm
from numpy.linalg import svd
import matplotlib.pyplot as plt
from tqdm import tqdm

############################################################
# FK local operators
############################################################

def fk_ops():
    w = np.exp(2j * np.pi / 3)
    U = np.diag([1, w, w**2]).astype(complex)
    # Nilpotent B (lowering operator)
    B = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ], dtype=complex)
    Bd = B.conj().T
    I = np.eye(3, dtype=complex)
    return B, Bd, U, I


def number_op():
    B, Bd, U, I = fk_ops()
    # n = B^† B + (B^†)^2 B^2
    return Bd @ B + (Bd @ Bd) @ (B @ B)



############################################################
# build bond Hamiltonian
############################################################

def bond_hamiltonians(J, g, mu, L):
    B, Bd, U, I = fk_ops()
    n = number_op()
    U2 = U @ U                     # U^2

    Hbond = []

    for j in range(L - 1):
        H = np.zeros((9, 9), dtype=complex)

        # Single-particle hopping: B_j^† U_j B_{j+1} + h.c.
        hop1 = np.kron(Bd @ U, B)
        H += -J * (1 - g) * (hop1 + hop1.conj().T)

        # Pair hopping: (B_j^†)^2 U_j^2 B_{j+1}^2 + h.c.
        hop2 = np.kron(Bd @ Bd @ U2, B @ B)
        H += -J * g * (hop2 + hop2.conj().T)

        # On-site disorder (split between bonds)
        wl = 1.0 if j == 0 else 0.5
        wr = 1.0 if j == L - 2 else 0.5
        H += wl * mu[j]     * np.kron(n, I)
        H += wr * mu[j + 1] * np.kron(I, n)

        Hbond.append(H.reshape(3, 3, 3, 3))

    return Hbond
############################################################
# exponentiated gates
############################################################

def exp_gates(J, g, mu, L, dt):
    Hbond = bond_hamiltonians(J, g, mu, L)
    gates = []

    for H in Hbond:
        Hmat = H.reshape(9, 9)

        # real-time evolution
        G = expm(-1j * dt * Hmat)

        # IMPORTANT:
        # reshape gives indices (sL_out, sR_out, sL_in, sR_in)
        gates.append(G.reshape(3, 3, 3, 3))

    return gates


############################################################
# utilities
############################################################

def safe_inv(x, eps=1e-12):
    out = np.zeros_like(x, dtype=np.complex128)
    mask = np.abs(x) > eps
    out[mask] = 1.0 / x[mask]
    return out


############################################################
# domain wall initial MPS (Vidal form)
############################################################

def domain_wall_mps(L):
    d = 3

    G = []
    l = []

    for _ in range(L):
        G.append(np.zeros((d, 1, 1), dtype=np.complex128))
        l.append(np.ones(1, dtype=np.complex128))

    l.append(np.ones(1, dtype=np.complex128))

    for j in range(L):
        if j < L // 2:
            G[j][1, 0, 0] = 1.0
        else:
            G[j][0, 0, 0] = 1.0

    return l, G


############################################################
# TEBD update
############################################################

def tebd_step(l, G, gates, chi_max=60):
    L = len(G)

    for parity in [0, 1]:
        for j in range(parity, L - 1, 2):
            d = G[j].shape[0]

            chi0 = G[j].shape[1]
            chi1 = G[j].shape[2]
            chi2 = G[j + 1].shape[2]

            lamL = l[j]
            lamM = l[j + 1]
            lamR = l[j + 2]

            inv_lamL = safe_inv(lamL)
            inv_lamR = safe_inv(lamR)

            # Build Theta_{a,s,t,b}
            theta = np.tensordot(np.diag(lamL), G[j], axes=(1, 1))      # (chi0,d,chi1)
            theta = np.tensordot(theta, np.diag(lamM), axes=(2, 0))     # (chi0,d,chi1)
            theta = np.tensordot(theta, G[j + 1], axes=(2, 1))          # (chi0,d,d,chi2)
            theta = np.tensordot(theta, np.diag(lamR), axes=(3, 0))     # (chi0,d,d,chi2)

            # Correct gate application:
            # gates[j] has indices (s_out, t_out, s_in, t_in)
            theta = np.tensordot(gates[j], theta, axes=([2, 3], [1, 2]))
            theta = np.transpose(theta, (2, 0, 1, 3))                   # (chi0,d,d,chi2)

            # SVD on (a,s) | (t,b)
            theta = theta.reshape(chi0 * d, d * chi2)

            X, S, Yh = svd(theta, full_matrices=False)

            keep = min(len(S), chi_max)

            X = X[:, :keep]
            S = S[:keep]
            Yh = Yh[:keep, :]

            # keep Schmidt values normalized
            nrm = np.linalg.norm(S)
            if nrm > 0:
                S = S / nrm

            l[j + 1] = S

            # left Gamma
            X = X.reshape(chi0, d, keep)
            X = np.tensordot(np.diag(inv_lamL), X, axes=(1, 0))         # (chi0,d,keep)
            G[j] = np.transpose(X, (1, 0, 2))                           # (d,chi0,keep)

            # right Gamma
            Y = Yh.reshape(keep, d, chi2)
            Y = np.transpose(Y, (1, 0, 2))                              # (d,keep,chi2)
            Y = np.tensordot(Y, np.diag(inv_lamR), axes=(2, 0))         # (d,keep,chi2)
            G[j + 1] = Y



############################################################
# expectation value of local operator
############################################################
def local_exp(l, G, op, pos):
    psi = np.tensordot(np.diag(l[pos]), G[pos], axes=(1, 1))
    psi = np.tensordot(psi, np.diag(l[pos + 1]), axes=(2, 0))  # (chiL,d,chiR)

    rho = np.tensordot(np.conj(psi), psi, axes=([0, 2], [0, 2]))  # (d,d)

    num = np.tensordot(rho, op, axes=([0, 1], [0, 1]))
    den = np.trace(rho)

    return (num / den).real



############################################################
# imbalance
############################################################
def imbalance(l, G):
    L = len(G)
    n = number_op()

    NL = 0.0
    NR = 0.0

    for j in range(L // 2):
        NL += local_exp(l, G, n, j)

    for j in range(L // 2, L):
        NR += local_exp(l, G, n, j)

    return 2.0 * (NL - NR) / L


############################################################
# simulation parameters
############################################################

L = 12
J = 1.0
g = 0.3
W = 2.0

dt = 0.1
steps = 500
samples = 50
chi_max = 60

np.random.seed(7)

############################################################
# disorder average
############################################################

imbalance_avg = np.zeros(steps, dtype=float)
times         = np.arange(steps) * dt * J

for r in range(samples):
    mu = np.random.uniform(-W, W, L)

    gates = exp_gates(J, g, mu, L, dt)
    l, G = domain_wall_mps(L)

    for t in tqdm(range(steps), desc=f"Sample {r}"):

        tebd_step(l, G, gates, chi_max=chi_max)
        imbalance_avg[t] += imbalance(l, G)

    print("sample", r, "done")

imbalance_avg /= samples


############################################################
# save data
############################################################

data = np.column_stack((times, imbalance_avg))

np.savetxt(
    f"imbalance_fk_z3_L{L}_g{g}_W{W}_U2.dat",
    data,
    header="Jt  I(t)"
)


############################################################
# plot
############################################################

plt.plot(times, imbalance_avg)
plt.xlabel("J t")
plt.ylabel("Imbalance")
plt.title("Domain-wall melting (Z3 FK chain)")
plt.tight_layout()
plt.show()