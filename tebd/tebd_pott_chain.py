import numpy as np
from scipy.linalg import expm
from numpy.linalg import svd

###############################################
# Local Potts operators
###############################################

def potts_operators():
    w = np.exp(2j*np.pi/3)

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
# Bond Hamiltonians with correct onsite splitting
###############################################

def potts_H_bond(J, f, L):
    Z, X, I = potts_operators()
    Xh = X.conj().T

    H_bond = []

    for i in range(L - 1):
        H = np.zeros((9, 9), dtype=complex)

        # nearest-neighbor Potts interaction
        H += -J[i] * (np.kron(Z.conj().T, Z) + np.kron(Z, Z.conj().T))

        # split transverse field between neighboring bonds
        left_weight  = 1.0 if i == 0     else 0.5
        right_weight = 1.0 if i == L - 2 else 0.5

        H += -left_weight  * f[i]   * (np.kron(X, I) + np.kron(Xh, I))
        H += -right_weight * f[i+1] * (np.kron(I, X) + np.kron(I, Xh))

        H_bond.append(H.reshape(3, 3, 3, 3))

    return H_bond


def potts_expH_bond(J, f, L, delta):
    H_bond = potts_H_bond(J, f, L)
    expH_bond = []

    for H in H_bond:
        U = expm(-delta * H.reshape(9, 9))
        expH_bond.append(U.reshape(3, 3, 3, 3))

    return expH_bond, H_bond


###############################################
# Initial MPS in Vidal form
###############################################

def initialMat(L, d, chiMax):
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
        l.append(np.ones(chi0, dtype=np.complex128))

    l.append(np.ones(int(chi[L]), dtype=np.complex128))

    # product state |0...0>
    for pos in range(L):
        G[pos][0, 0, 0] = 1.0

    return l, G


###############################################
# One TEBD sweep on chosen bond parity
###############################################

def tebd_sweep_parity(l, G, expH, parity):
    L = len(G)

    for i_bond in range(parity, L - 1, 2):
        i0 = i_bond
        i1 = i_bond + 1
        i2 = i_bond + 2

        d = G[i0].shape[0]
        chi0 = G[i0].shape[1]
        chi1 = G[i0].shape[2]
        chi2 = G[i1].shape[2]

        l0inv = 1.0 / (l[i0] + 1e-20)
        l2inv = 1.0 / (l[i2] + 1e-20)

        # theta tensor
        Psi = np.tensordot(np.diag(l[i0]), G[i0], axes=(1, 1))
        Psi = np.tensordot(Psi, np.diag(l[i1]), axes=(2, 0))
        Psi = np.tensordot(Psi, G[i1], axes=(2, 1))
        Psi = np.tensordot(Psi, np.diag(l[i2]), axes=(3, 0))
        # shape: (chi0, d, d, chi2)

        # apply two-site gate
        Phi = np.tensordot(Psi, expH[i0], axes=([1, 2], [0, 1]))
        Phi = np.transpose(Phi, (0, 2, 3, 1))
        Phi = np.reshape(Phi, (chi0 * d, d * chi2))

        # split with SVD
        U, S, Vh = svd(Phi, full_matrices=False)

        keep = min(chi1, len(S))
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        # normalize Schmidt values
        nrm = np.linalg.norm(S)
        if nrm > 0:
            S = S / nrm

        l[i1] = S

        # update Gamma[i0]
        U = U.reshape(chi0, d, keep)
        U = np.tensordot(np.diag(l0inv), U, axes=(1, 0))
        U = np.transpose(U, (1, 0, 2))
        G[i0] = U

        # update Gamma[i1]
        V = Vh.reshape(keep, d, chi2)
        V = np.transpose(V, (1, 0, 2))
        V = np.tensordot(V, np.diag(l2inv), axes=(2, 0))
        G[i1] = V


###############################################
# Second-order Suzuki-Trotter
###############################################

def tebd_second_order_step(l, G, expH_half, expH_full):
    tebd_sweep_parity(l, G, expH_half, parity=0)
    tebd_sweep_parity(l, G, expH_full, parity=1)
    tebd_sweep_parity(l, G, expH_half, parity=0)


###############################################
# Energy measurement
###############################################

def bond_energy(l, G, H, pos):
    Psi = np.tensordot(np.diag(l[pos]), G[pos], axes=(1, 1))
    Psi = np.tensordot(Psi, np.diag(l[pos + 1]), axes=(2, 0))
    Psi = np.tensordot(Psi, G[pos + 1], axes=(2, 1))
    Psi = np.tensordot(Psi, np.diag(l[pos + 2]), axes=(3, 0))

    rho = np.tensordot(np.conj(Psi), Psi, axes=([0, 3], [0, 3]))
    return np.tensordot(rho, H[pos], axes=([0, 1, 2, 3], [0, 1, 2, 3]))


def total_energy(l, G, H_bond):
    E = 0.0 + 0.0j
    for i in range(len(G) - 1):
        E += bond_energy(l, G, H_bond, i)
    return E


###############################################
# Main
###############################################

if __name__ == "__main__":
    L = 7
    d = 3
    chiMax = 30
    delta = 0.01
    steps = 1000

    J = np.ones(L, dtype=float)
    f = np.ones(L, dtype=float)

    l, G = initialMat(L, d, chiMax)

    expH_half, H_bond = potts_expH_bond(J, f, L, delta / 2.0)
    expH_full, _ = potts_expH_bond(J, f, L, delta)
    
    print(f"L = {L}")
    for step in range(steps):
        tebd_second_order_step(l, G, expH_half, expH_full)


        if step % 100 == 0:
            E = total_energy(l, G, H_bond)
            print(f"step {step:3d} energy {E.real:.12f}")

    # Final energy measurement
    print("\nFinal energy: ", total_energy(l, G, H_bond)/L)
    print("Exact ground-state energy = -2.355433747522252")
    print("Difference from exact: ", total_energy(l, G, H_bond)/L - (-2.355433747522252))