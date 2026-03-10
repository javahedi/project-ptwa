import numpy as np
from scipy.linalg import expm, eigh
from numpy.linalg import svd

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



def safe_inv(x, eps=1e-12):
    out = np.zeros_like(x, dtype=np.complex128)
    mask = np.abs(x) > eps
    out[mask] = 1.0 / x[mask]
    return out


############################################################
# Build two-site bond Hamiltonians for TEBD
############################################################

def fk_bond_hamiltonians(J, g, mu, L):
    B, Bd, U, I = fk_ops()
    n = number_op()

    H_bond = []

    for j in range(L - 1):
        H = np.zeros((9, 9), dtype=complex)

        # Single-particle hopping: B_j^† U_j B_{j+1} + h.c.
        t1 = np.kron(Bd @ U, B)
        H += -J * (1.0 - g) * (t1 + t1.conj().T)

        # Pair hopping: (B_j^†)^2 B_{j+1}^2 + h.c.
        t2 = np.kron(Bd @ Bd, B @ B)
        H += -J * g * (t2 + t2.conj().T)

        # On-site disorder: split between bonds
        wl = 1.0 if j == 0 else 0.5
        wr = 1.0 if j == L - 2 else 0.5
        H += wl * mu[j]     * np.kron(n, I)
        H += wr * mu[j + 1] * np.kron(I, n)

        H_bond.append(H.reshape(3, 3, 3, 3))

    return H_bond



def fk_expH_bond(J, g, mu, L, delta):
    H_bond = fk_bond_hamiltonians(J, g, mu, L)
    expH_bond = []

    for H in H_bond:
        Hmat = H.reshape(9, 9)
        expH_bond.append(expm(-delta * Hmat).reshape(3, 3, 3, 3))

    return expH_bond, H_bond


############################################################
# Vidal-form MPS initialization
############################################################
def initialMat(L, d, chiMax, state='uniform_superposition'):
    G = []
    l = []
    chi = [0] * (L + 1)

    for pos in range(0, L // 2 + 1):
        chi[pos] = min(chiMax, d**pos)
    for pos in range(L // 2 + 1, L + 1):
        chi[pos] = min(chiMax, d**(L - pos))

    for pos in range(L):
        chi0 = int(chi[pos])
        chi1 = int(chi[pos + 1])

        G.append(np.zeros((d, chi0, chi1), dtype=np.complex128))

        lam = np.zeros(chi0, dtype=np.complex128)
        lam[0] = 1.0
        l.append(lam)

    lam = np.zeros(int(chi[L]), dtype=np.complex128)
    lam[0] = 1.0
    l.append(lam)

    for pos in range(L):
        if state == 'vacuum':
            G[pos][0, 0, 0] = 1.0
        elif state == 'ones':
            G[pos][1, 0, 0] = 1.0
        elif state == 'twos':
            G[pos][2, 0, 0] = 1.0
        elif state == 'uniform_superposition':
            G[pos][:, 0, 0] = np.array([1,1,1], dtype=np.complex128) / np.sqrt(3.0)
        elif state == 'random_product':
            v = np.random.randn(d) + 1j*np.random.randn(d)
            v = v / np.linalg.norm(v)
            G[pos][:, 0, 0] = v
        else:
            raise ValueError("unknown initial state")

    return l, G


############################################################
# TEBD sweep
############################################################

def tebd_sweep_parity(l, G, expH, parity, chiMax=None):
    L = len(G)

    for i in range(parity, L - 1, 2):
        chiL = G[i].shape[1]
        chiR = G[i+1].shape[2]
        d = G[i].shape[0]

        lamL = l[i]
        lamM = l[i+1]
        lamR = l[i+2]

        inv_lamL = safe_inv(lamL)
        inv_lamR = safe_inv(lamR)

        # Theta_{a,s,t,b}
        Theta = np.tensordot(np.diag(lamL), G[i], axes=(1, 1))   # (chiL,d,chiM)
        Theta = np.tensordot(Theta, np.diag(lamM), axes=(2, 0))  # (chiL,d,chiM)
        Theta = np.tensordot(Theta, G[i+1], axes=(2, 1))         # (chiL,d,d,chiR)
        Theta = np.tensordot(Theta, np.diag(lamR), axes=(3, 0))  # (chiL,d,d,chiR)

        # Correct gate action
        # expH[i] indices: (s_out, t_out, s_in, t_in)
        Theta = np.tensordot(expH[i], Theta, axes=([2, 3], [1, 2]))
        Theta = np.transpose(Theta, (2, 0, 1, 3))                # (chiL,d,d,chiR)

        M = Theta.reshape(chiL * d, d * chiR)
        X, S, Yh = np.linalg.svd(M, full_matrices=False)

        keep = len(S)
        if chiMax is not None:
            keep = min(keep, chiMax)

        X = X[:, :keep]
        S = S[:keep]
        Yh = Yh[:keep, :]

        norm = np.linalg.norm(S)
        if norm > 0:
            S = S / norm

        l[i+1] = S

        X = X.reshape(chiL, d, keep)
        X = np.tensordot(np.diag(inv_lamL), X, axes=(1, 0))
        G[i] = np.transpose(X, (1, 0, 2))

        Y = Yh.reshape(keep, d, chiR)
        Y = np.transpose(Y, (1, 0, 2))
        Y = np.tensordot(Y, np.diag(inv_lamR), axes=(2, 0))
        G[i+1] = Y
        
def tebd_second_order_step(l, G, expH_half, expH_full, chiMax=None):
    tebd_sweep_parity(l, G, expH_half, parity=0, chiMax=chiMax)
    tebd_sweep_parity(l, G, expH_full, parity=1, chiMax=chiMax)
    tebd_sweep_parity(l, G, expH_half, parity=0, chiMax=chiMax)


############################################################
# Energy measurement
############################################################

def bond_energy(l, G, H_bond, pos):
    Psi = np.tensordot(np.diag(l[pos]), G[pos], axes=(1, 1))
    Psi = np.tensordot(Psi, np.diag(l[pos + 1]), axes=(2, 0))
    Psi = np.tensordot(Psi, G[pos + 1], axes=(2, 1))
    Psi = np.tensordot(Psi, np.diag(l[pos + 2]), axes=(3, 0))

    rho = np.tensordot(np.conj(Psi), Psi, axes=([0, 3], [0, 3]))
    return np.tensordot(rho, H_bond[pos], axes=([0, 1, 2, 3], [0, 1, 2, 3]))


def total_energy(l, G, H_bond):
    E = 0.0 + 0.0j
    for i in range(len(G) - 1):
        E += bond_energy(l, G, H_bond, i)
    return E


############################################################
# ED Hamiltonian
############################################################

def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def one_site(op, site, L, I):
    ops = [I for _ in range(L)]
    ops[site] = op
    return kron_all(ops)


def two_site(op1, s1, op2, s2, L, I):
    ops = [I for _ in range(L)]
    ops[s1] = op1
    ops[s2] = op2
    return kron_all(ops)

def fk_hamiltonian_ed(J, g, mu, L):
    B, Bd, U, I = fk_ops()
    n = number_op()

    dim = 3**L
    H = np.zeros((dim, dim), dtype=complex)

    for j in range(L - 1):
        # Single-particle hopping
        H += -J * (1 - g) * two_site(Bd @ U, j, B, j + 1, L, I)
        H += -J * (1 - g) * two_site(U.conj().T @ B, j, Bd, j + 1, L, I)

        # Pair hopping (no U factor)
        H += -J * g * two_site(Bd @ Bd, j, B @ B, j + 1, L, I)
        H += -J * g * two_site(B @ B, j, Bd @ Bd, j + 1, L, I)

    # On-site disorder
    for j in range(L):
        H += mu[j] * one_site(n, j, L, I)

    return H


def exact_ground_energy(J, g, mu, L):
    H = fk_hamiltonian_ed(J, g, mu, L)
    evals, evecs = eigh(H)
    return evals[0].real, evals, evecs


############################################################
# Main test
############################################################

if __name__ == "__main__":
    np.random.seed(7)

    # small system for ED comparison
    L = 6
    d = 3
    chiMax = 100

    J = 1.0
    g = 0.5
    W = 1.0

    # one disorder realization
    mu = np.random.uniform(-W, W, L)

    # exact diagonalization
    E_ed, evals, evecs = exact_ground_energy(J, g, mu, L)
    print(f"L = {L}")
    print("mu =", mu)
    print(f"ED ground-state energy = {E_ed:.12f}")

    # TEBD imaginary-time evolution
    state = 'ones'  # 'vacuum' or 'ones'
    print(f"initial state = {state}")
    l, G = initialMat(L, d, chiMax, state=state)

    # step schedule
    schedule = [
        (0.10, 100),
        (0.05, 200),
        (0.02, 400),
        (0.01, 800),
        (0.005, 800),
        (0.002, 1500),
        (0.001, 2000),
    ]

    E_tebd = None

    for delta, steps in schedule:
        expH_half, H_bond = fk_expH_bond(J, g, mu, L, delta / 2.0)
        expH_full, _ = fk_expH_bond(J, g, mu, L, delta)

        for step in range(steps):
            tebd_second_order_step(l, G, expH_half, expH_full, chiMax=chiMax)

        E_tebd = total_energy(l, G, H_bond).real
        print(f"after delta={delta:7.4f}, steps={steps:4d}, E_TEBD={E_tebd:.12f}, err={E_tebd-E_ed:+.6e}")

    print()
    print(f"Final ED energy   = {E_ed:.12f}")
    print(f"Final TEBD energy = {E_tebd:.12f}")
    print(f"Absolute error    = {abs(E_tebd - E_ed):.6e}")