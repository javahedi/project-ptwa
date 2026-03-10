import numpy as np
from scipy.linalg import expm
from numpy.linalg import svd
import matplotlib.pyplot as plt
from tqdm import tqdm

############################################################
# Local Z_n Fock-parafermion operators
############################################################

def fk_ops(n):
    """
    Local operators for Z_n Fock parafermions in the occupation basis
    |0>, |1>, ..., |n-1>.

    Returns
    -------
    B   : nilpotent lowering operator
    Bd  : raising operator
    U   : diagonal clock operator diag(1, w, w^2, ...)
    proj: list of projectors |a><a|
    I   : identity
    """
    w = np.exp(2j * np.pi / n)

    U = np.diag([w**a for a in range(n)]).astype(complex)

    B = np.zeros((n, n), dtype=complex)
    for a in range(1, n):
        B[a - 1, a] = 1.0

    Bd = B.conj().T
    I = np.eye(n, dtype=complex)

    proj = []
    for a in range(n):
        P = np.zeros((n, n), dtype=complex)
        P[a, a] = 1.0
        proj.append(P)

    return B, Bd, U, proj, I


def clock_op(n):
    _, _, U, _, _ = fk_ops(n)
    return U


############################################################
# Pair-hopping-only clean Hamiltonian
############################################################

def bond_hamiltonians_pair_clean(J, gpair, L, n):
    """
    Nearest-neighbour clean Hamiltonian with only pair hopping:
        H = -J gpair sum_j [ (Bd_j)^2 U_j^2 (B_{j+1})^2 + h.c. ]
    """
    B, Bd, U, _, _ = fk_ops(n)

    B2 = B @ B
    Bd2 = Bd @ Bd
    U2 = U @ U

    Hbond = []

    for _ in range(L - 1):
        H = np.zeros((n * n, n * n), dtype=complex)

        hop2 = np.kron(Bd2 @ U2, B2)
        H += -J * gpair * (hop2 + hop2.conj().T)

        Hbond.append(H.reshape(n, n, n, n))

    return Hbond


def exp_gates_pair_clean(J, gpair, L, dt, n):
    Hbond = bond_hamiltonians_pair_clean(J, gpair, L, n)
    gates = []

    for H in Hbond:
        Hmat = H.reshape(n * n, n * n)
        G = expm(-1j * dt * Hmat)
        gates.append(G.reshape(n, n, n, n))

    return gates


############################################################
# Utilities
############################################################

def safe_inv(x, eps=1e-12):
    out = np.zeros_like(x, dtype=np.complex128)
    mask = np.abs(x) > eps
    out[mask] = 1.0 / x[mask]
    return out


############################################################
# Product-state MPS in Vidal form
############################################################

def periodic_mps(L, n):
    """
    Period-n product state:
        |0,1,2,...,n-1,0,1,2,...>
    """
    G = []
    l = []

    for _ in range(L):
        G.append(np.zeros((n, 1, 1), dtype=np.complex128))
        l.append(np.ones(1, dtype=np.complex128))

    l.append(np.ones(1, dtype=np.complex128))

    for j in range(L):
        a = j % n
        G[j][a, 0, 0] = 1.0

    return l, G


############################################################
# TEBD update
############################################################

def tebd_step(l, G, gates, chi_max=100):
    L = len(G)

    for parity in [0, 1]:
        for j in range(parity, L - 1, 2):
            d = G[j].shape[0]

            chi0 = G[j].shape[1]
            chi2 = G[j + 1].shape[2]

            lamL = l[j]
            lamM = l[j + 1]
            lamR = l[j + 2]

            inv_lamL = safe_inv(lamL)
            inv_lamR = safe_inv(lamR)

            # Theta(a,s,t,b)
            theta = np.tensordot(np.diag(lamL), G[j], axes=(1, 1))      # (chi0,d,chi1)
            theta = np.tensordot(theta, np.diag(lamM), axes=(2, 0))     # (chi0,d,chi1)
            theta = np.tensordot(theta, G[j + 1], axes=(2, 1))          # (chi0,d,d,chi2)
            theta = np.tensordot(theta, np.diag(lamR), axes=(3, 0))     # (chi0,d,d,chi2)

            # Apply gate with indices (s_out, t_out, s_in, t_in)
            theta = np.tensordot(gates[j], theta, axes=([2, 3], [1, 2]))
            theta = np.transpose(theta, (2, 0, 1, 3))                   # (chi0,d,d,chi2)

            # SVD on (a,s) | (t,b)
            theta = theta.reshape(chi0 * d, d * chi2)
            X, S, Yh = svd(theta, full_matrices=False)

            keep = min(len(S), chi_max)

            X = X[:, :keep]
            S = S[:keep]
            Yh = Yh[:keep, :]

            nrm = np.linalg.norm(S)
            if nrm > 0:
                S = S / nrm

            l[j + 1] = S

            # Left Gamma
            X = X.reshape(chi0, d, keep)
            X = np.tensordot(np.diag(inv_lamL), X, axes=(1, 0))
            G[j] = np.transpose(X, (1, 0, 2))

            # Right Gamma
            Y = Yh.reshape(keep, d, chi2)
            Y = np.transpose(Y, (1, 0, 2))
            Y = np.tensordot(Y, np.diag(inv_lamR), axes=(2, 0))
            G[j + 1] = Y


############################################################
# Convert Vidal tensors to site tensors A_j(alpha,s,beta)
############################################################

def site_tensor(l, G, j):
    A = np.tensordot(np.diag(l[j]), G[j], axes=(1, 1))      # (chiL,d,chiM)
    A = np.tensordot(A, np.diag(l[j + 1]), axes=(2, 0))     # (chiL,d,chiR)
    return A


############################################################
# MPS expectation values
############################################################

def transfer_noop(E, A):
    # E_ab, conj(A)_{a,s,i}, A_{b,s,j} -> E'_{i,j}
    return np.einsum('ab,asi,bsj->ij', E, np.conj(A), A, optimize=True)


def transfer_op(E, A, op):
    # E_ab, conj(A)_{a,s,i}, A_{b,t,j}, op_{s,t} -> E'_{i,j}
    return np.einsum('ab,asi,btj,st->ij', E, np.conj(A), A, op, optimize=True)


def one_point_exp(l, G, op, pos):
    L = len(G)
    E = np.array([[1.0 + 0.0j]])

    for j in range(L):
        A = site_tensor(l, G, j)
        if j == pos:
            E = transfer_op(E, A, op)
        else:
            E = transfer_noop(E, A)

    return E[0, 0]


def two_point_exp(l, G, op1, i, op2, j):
    """
    Computes < op1_i op2_j > for i <= j.
    """
    if i > j:
        return two_point_exp(l, G, op2, j, op1, i)

    L = len(G)
    E = np.array([[1.0 + 0.0j]])

    for site in range(L):
        A = site_tensor(l, G, site)

        if site == i and site == j:
            E = transfer_op(E, A, op1 @ op2)
        elif site == i:
            E = transfer_op(E, A, op1)
        elif site == j:
            E = transfer_op(E, A, op2)
        else:
            E = transfer_noop(E, A)

    return E[0, 0]


def local_profile(l, G, op):
    L = len(G)
    vals = np.zeros(L, dtype=complex)
    for j in range(L):
        vals[j] = one_point_exp(l, G, op, j)
    return vals


def two_point_matrix(l, G, op1, op2):
    L = len(G)
    C = np.zeros((L, L), dtype=complex)

    for i in range(L):
        for j in range(i, L):
            val = two_point_exp(l, G, op1, i, op2, j)
            C[i, j] = val
            if i != j:
                # For Hermitian-local ops this is okay; for general use, compute directly if needed.
                C[j, i] = np.conj(two_point_exp(l, G, op2, j, op1, i))

    return C


############################################################
# Structure factors
############################################################

def momentum_grid(L):
    return 2.0 * np.pi * np.arange(L) / L


def phase_matrix(L, q):
    idx = np.arange(L)
    return np.exp(1j * q * (idx[:, None] - idx[None, :]))


def S_Z_q(l, G, n):
    """
    Non-connected clock structure factor:
        S_Z(q) = (1/L) sum_{j,l} e^{iq(j-l)} < Z_j^\dagger Z_l >
    """
    Z = clock_op(n)
    Zdag = Z.conj().T
    L = len(G)

    C = two_point_matrix(l, G, Zdag, Z)
    qs = momentum_grid(L)
    Sq = np.zeros(L, dtype=float)

    for iq, q in enumerate(qs):
        phase = phase_matrix(L, q)
        Sq[iq] = np.real(np.sum(phase * C) / L)

    return qs, Sq


def S_a_q(l, G, n, a=0):
    """
    Connected projector structure factor:
        S_a(q) = (1/L) sum_{j,l} e^{iq(j-l)}
                 [ <P_j^a P_l^a> - <P_j^a><P_l^a> ]
    """
    _, _, _, proj, _ = fk_ops(n)
    P = proj[a]
    L = len(G)

    one = local_profile(l, G, P)
    two = two_point_matrix(l, G, P, P)
    conn = two - np.outer(one, one)

    qs = momentum_grid(L)
    Sq = np.zeros(L, dtype=float)

    for iq, q in enumerate(qs):
        phase = phase_matrix(L, q)
        Sq[iq] = np.real(np.sum(phase * conn) / L)

    return qs, Sq


############################################################
# Main simulation
############################################################

def run_simulation(
    n,
    L=12,
    J=1.0,
    gpair=0.3,
    dt=0.1,
    steps=200,
    chi_max=100,
    a_proj=0,
):
    gates = exp_gates_pair_clean(J, gpair, L, dt, n)
    l, G = periodic_mps(L, n)

    qs = momentum_grid(L)
    q_star = 2.0 * np.pi / n
    iq_star = int(np.argmin(np.abs(qs - q_star)))

    times = np.arange(steps + 1) * dt * J
    SZ_tq = np.zeros((steps + 1, L), dtype=float)
    Sa_tq = np.zeros((steps + 1, L), dtype=float)
    SZ_qstar = np.zeros(steps + 1, dtype=float)
    Sa_qstar = np.zeros(steps + 1, dtype=float)

    # t = 0
    _, SZ = S_Z_q(l, G, n)
    _, Sa = S_a_q(l, G, n, a=a_proj)

    SZ_tq[0] = SZ
    Sa_tq[0] = Sa
    SZ_qstar[0] = SZ[iq_star]
    Sa_qstar[0] = Sa[iq_star]

    # time evolution
    for t in tqdm(range(1, steps + 1), desc=f"n={n}"):
        tebd_step(l, G, gates, chi_max=chi_max)

        _, SZ = S_Z_q(l, G, n)
        _, Sa = S_a_q(l, G, n, a=a_proj)

        SZ_tq[t] = SZ
        Sa_tq[t] = Sa
        SZ_qstar[t] = SZ[iq_star]
        Sa_qstar[t] = Sa[iq_star]

    return {
        "n": n,
        "L": L,
        "J": J,
        "gpair": gpair,
        "dt": dt,
        "steps": steps,
        "chi_max": chi_max,
        "a_proj": a_proj,
        "times": times,
        "qs": qs,
        "q_star": q_star,
        "iq_star": iq_star,
        "SZ_tq": SZ_tq,
        "Sa_tq": Sa_tq,
        "SZ_qstar": SZ_qstar,
        "Sa_qstar": Sa_qstar,
    }


############################################################
# Plotting
############################################################

def plot_results(res):
    n = res["n"]
    times = res["times"]
    qs = res["qs"]
    q_star = res["q_star"]
    SZ_tq = res["SZ_tq"]
    Sa_tq = res["Sa_tq"]
    SZ_qstar = res["SZ_qstar"]
    Sa_qstar = res["Sa_qstar"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    im0 = axes[0, 0].imshow(
        SZ_tq,
        aspect="auto",
        origin="lower",
        extent=[qs[0], qs[-1], times[0], times[-1]]
    )
    axes[0, 0].set_title(rf"$S_Z(q,t)$ for $n={n}$")
    axes[0, 0].set_xlabel(r"$q$")
    axes[0, 0].set_ylabel(r"$J t$")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(
        Sa_tq,
        aspect="auto",
        origin="lower",
        extent=[qs[0], qs[-1], times[0], times[-1]]
    )
    axes[0, 1].set_title(rf"$S_a(q,t)$ for $n={n}$")
    axes[0, 1].set_xlabel(r"$q$")
    axes[0, 1].set_ylabel(r"$J t$")
    plt.colorbar(im1, ax=axes[0, 1])

    axes[1, 0].plot(times, SZ_qstar, label=rf"$S_Z(q_*,t)$, $q_*=2\pi/{n}$")
    axes[1, 0].plot(times, Sa_qstar, label=rf"$S_a(q_*,t)$")
    axes[1, 0].set_xlabel(r"$J t$")
    axes[1, 0].set_ylabel("mode amplitude")
    axes[1, 0].legend()
    axes[1, 0].set_title("Ordering-wavevector dynamics")

    axes[1, 1].plot(qs, SZ_tq[0], label=r"$t=0$")
    axes[1, 1].plot(qs, SZ_tq[len(times)//2], label=r"mid")
    axes[1, 1].plot(qs, SZ_tq[-1], label=r"final")
    axes[1, 1].axvline(q_star, ls="--", alpha=0.6)
    axes[1, 1].set_xlabel(r"$q$")
    axes[1, 1].set_ylabel(r"$S_Z(q,t)$")
    axes[1, 1].legend()
    axes[1, 1].set_title("Selected momentum profiles")

    plt.tight_layout()
    plt.show()


############################################################
# Save data
############################################################

def save_results(res):
    n = res["n"]
    L = res["L"]
    gpair = res["gpair"]
    a_proj = res["a_proj"]

    np.savez(
        f"structure_factors_pairhop_clean_n{n}_L{L}_g{gpair:.3f}_a{a_proj}.npz",
        times=res["times"],
        qs=res["qs"],
        q_star=res["q_star"],
        SZ_tq=res["SZ_tq"],
        Sa_tq=res["Sa_tq"],
        SZ_qstar=res["SZ_qstar"],
        Sa_qstar=res["Sa_qstar"],
    )

    np.savetxt(
        f"SZ_qstar_pairhop_clean_n{n}_L{L}_g{gpair:.3f}.dat",
        np.column_stack((res["times"], res["SZ_qstar"])),
        header="Jt  S_Z(q_star,t)"
    )

    np.savetxt(
        f"Sa_qstar_pairhop_clean_n{n}_L{L}_g{gpair:.3f}_a{a_proj}.dat",
        np.column_stack((res["times"], res["Sa_qstar"])),
        header="Jt  S_a(q_star,t)"
    )


############################################################
# Parameters and execution
############################################################

if __name__ == "__main__":

    # Global simulation parameters
    n_values = [3, 4, 5]
    L = 12
    J = 1.0
    gpair = 0.3
    dt = 0.1
    steps = 120
    chi_max = 80

    # Choose which projector flavor enters S_a(q,t)
    a_proj = 0

    for n in n_values:
        print(f"\nRunning clean pair-hopping TEBD for n={n}")

        res = run_simulation(
            n=n,
            L=L,
            J=J,
            gpair=gpair,
            dt=dt,
            steps=steps,
            chi_max=chi_max,
            a_proj=a_proj,
        )

        save_results(res)
        plot_results(res)