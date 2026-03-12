import numpy as np
from scipy.linalg import eigh

###############################################
# Local Potts operators
###############################################

def potts_operators():
    w = np.exp(2j * np.pi / 3)

    Z = np.array([
        [1, 0, 0],
        [0, w, 0],
        [0, 0, w**2]
    ], dtype=complex)

    X = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=complex)

    I = np.eye(3, dtype=complex)
    return Z, X, I


###############################################
# Embed one-site / two-site operators
###############################################

def kron_all(op_list):
    out = op_list[0]
    for op in op_list[1:]:
        out = np.kron(out, op)
    return out


def one_site_op(op, site, L, d=3):
    I = np.eye(d, dtype=complex)
    ops = [I] * L
    ops[site] = op
    return kron_all(ops)


def two_site_op(op1, site1, op2, site2, L, d=3):
    I = np.eye(d, dtype=complex)
    ops = [I] * L
    ops[site1] = op1
    ops[site2] = op2
    return kron_all(ops)


###############################################
# Exact Potts Hamiltonian (open boundary)
###############################################

def potts_hamiltonian_ed(J, f, L):
    Z, X, I = potts_operators()
    Zd = Z.conj().T
    Xd = X.conj().T

    dim = 3**L
    H = np.zeros((dim, dim), dtype=complex)

    # nearest-neighbor interaction
    for j in range(L - 1):
        H += -J[j] * two_site_op(Zd, j,   Z,  j + 1, L)
        H += -J[j] * two_site_op(Z,  j,   Zd, j + 1, L)

    # onsite transverse field
    for j in range(L):
        H += -f[j] * one_site_op(X,  j, L)
        H += -f[j] * one_site_op(Xd, j, L)

    return H


###############################################
# Exact ground state energy
###############################################

def exact_ground_energy(J, f, L):
    H = potts_hamiltonian_ed(J, f, L)
    evals, evecs = eigh(H)
    return evals[0].real, evals, evecs


###############################################
# Optional: exact local observables
###############################################

def exact_expectation(psi, op):
    return np.vdot(psi, op @ psi)


###############################################
# Example run
###############################################

if __name__ == "__main__":
    L = 7
    J = np.ones(L - 1, dtype=float)   # bonds: 0..L-2
    f = np.ones(L, dtype=float)       # sites: 0..L-1

    E0, evals, evecs = exact_ground_energy(J, f, L)

    print(f"L = {L}")
    print(f"Hilbert space dimension = {3**L}")
    print(f"Exact ground-state energy = {E0:.12f}/{L:.12f} = {E0/L } per site")

    # optional: first few eigenvalues
    print("\nLowest few eigenvalues:")
    for n in range(min(10, len(evals))):
        print(f"{n:2d}  {evals[n].real:.12f}")