using LinearAlgebra
using Random
using Printf

# Optional: avoid BLAS oversubscription if you use Julia threads elsewhere.
BLAS.set_num_threads(1)

############################################################
# Local FK / parafermion operators
############################################################

function fk_ops()
    w = cis(2π / 3)

    U = ComplexF64[
        1  0   0
        0  w   0
        0  0  w^2
    ]

    # Nilpotent lowering operator
    B = ComplexF64[
        0  1  0
        0  0  1
        0  0  0
    ]

    Bd = adjoint(B)
    I3 = Matrix{ComplexF64}(I, 3, 3)
    return B, Bd, U, I3
end

function number_op()
    B, Bd, U, I3 = fk_ops()
    return Bd * B + (Bd * Bd) * (B * B)   # diag(0,1,2)
end

@inline function safe_inv_vec(v::AbstractVector{ComplexF64}; eps=1e-12)
    out = similar(v)
    @inbounds for i in eachindex(v)
        out[i] = abs(v[i]) > eps ? inv(v[i]) : 0.0 + 0.0im
    end
    return out
end

############################################################
# Bond Hamiltonians / gates
# Corrected pair-hopping includes the parafermion string phase U^2
############################################################
function fk_bond_hamiltonians(J, g, mu, L)
    B, Bd, U, I3 = fk_ops()
    n = number_op()

    H_bond = Vector{Matrix{ComplexF64}}(undef, L - 1)

    for j in 1:(L - 1)
        H = zeros(ComplexF64, 9, 9)

        # Single-particle hopping
        t1 = kron(Bd * U, B)
        H .+= -J * (1.0 - g) * (t1 + adjoint(t1))

        # Pair hopping
        t2 = kron(Bd * Bd, B * B)
        H .+= -J * g * (t2 + adjoint(t2))

        # Split onsite disorder across bonds
        wl = (j == 1)     ? 1.0 : 0.5
        wr = (j == L - 1) ? 1.0 : 0.5

        H .+= wl * mu[j]     * kron(n, I3)
        H .+= wr * mu[j + 1] * kron(I3, n)

        H_bond[j] = H
    end

    return H_bond
end

function fk_expH_bond(J, g, mu, L, delta)
    H_bond = fk_bond_hamiltonians(J, g, mu, L)
    expH_bond = Vector{Matrix{ComplexF64}}(undef, L - 1)

    for i in 1:(L - 1)
        expH_bond[i] = exp(-delta * H_bond[i])
    end

    return expH_bond, H_bond
end


############################################################
# Vidal-form MPS initialization
############################################################

function initialMat(L, d, chiMax; state="uniform_superposition")
    G = Vector{Array{ComplexF64,3}}(undef, L)
    l = Vector{Vector{ComplexF64}}(undef, L + 1)
    chi = zeros(Int, L + 1)

    for pos in 0:div(L, 2)
        chi[pos + 1] = min(chiMax, d^pos)
    end
    for pos in (div(L, 2) + 1):L
        chi[pos + 1] = min(chiMax, d^(L - pos))
    end

    for pos in 1:L
        χL = chi[pos]
        χR = chi[pos + 1]
        G[pos] = zeros(ComplexF64, d, χL, χR)

        λ = zeros(ComplexF64, χL)
        λ[1] = 1.0 + 0im
        l[pos] = λ
    end

    λ = zeros(ComplexF64, chi[L + 1])
    λ[1] = 1.0 + 0im
    l[L + 1] = λ

    for pos in 1:L
        if state == "vacuum"
            G[pos][1, 1, 1] = 1.0
        elseif state == "ones"
            G[pos][2, 1, 1] = 1.0
        elseif state == "twos"
            G[pos][3, 1, 1] = 1.0
        elseif state == "uniform_superposition"
            G[pos][:, 1, 1] .= ComplexF64[1, 1, 1] ./ sqrt(3.0)
        elseif state == "random_product"
            v = randn(d) .+ im * randn(d)
            v ./= norm(v)
            G[pos][:, 1, 1] .= ComplexF64.(v)
        else
            error("unknown initial state")
        end
    end

    return l, G
end

############################################################
# Fast, allocation-light two-site TEBD update
# Removes generic tensordot/permutedims allocations.
############################################################

function build_bond_matrix(G1, G2, λL, λM, λR, gate)
    d  = size(G1, 1)   # 3
    χL = size(G1, 2)
    χM = size(G1, 3)
    χR = size(G2, 3)

    M = zeros(ComplexF64, χL * d, d * χR)
    v = zeros(ComplexF64, 9)
    w = zeros(ComplexF64, 9)

    @inbounds for a in 1:χL
        λLa = λL[a]
        for b in 1:χR
            λRb = λR[b]

            fill!(v, 0.0 + 0.0im)

            for m in 1:χM
                fac = λLa * λM[m] * λRb

                for s in 1:3, t in 1:3
                    idx = (s - 1) * 3 + t
                    v[idx] += fac * G1[s, a, m] * G2[t, m, b]
                end
            end

            mul!(w, gate, v)   # w = gate * v

            for s in 1:3, t in 1:3
                row = (a - 1) * d + s
                col = (t - 1) * χR + b
                idx = (s - 1) * 3 + t
                M[row, col] = w[idx]
            end
        end
    end

    return M
end



function tebd_update_bond!(l, G, gate, i; chiMax=nothing)
    G1 = G[i]
    G2 = G[i + 1]

    λL = l[i]
    λM = l[i + 1]
    λR = l[i + 2]

    χL = length(λL)
    χR = length(λR)
    d  = size(G1, 1)

    M = build_bond_matrix(G1, G2, λL, λM, λR, gate)

    F = svd(M)
    Umat = F.U
    S    = F.S
    Vh   = F.Vt

    keep = length(S)
    if chiMax !== nothing
        keep = min(keep, chiMax)
    end

    Umat = Umat[:, 1:keep]
    S    = S[1:keep]
    Vh   = Vh[1:keep, :]

    nrm = norm(S)
    if nrm > 0
        S ./= nrm
    end
    l[i + 1] = ComplexF64.(S)

    invλL = safe_inv_vec(λL)
    invλR = safe_inv_vec(λR)

    # Update G[i]: reshape Umat -> (χL, d, keep), divide by λL, permute to (d, χL, keep)
    Gnew1 = zeros(ComplexF64, d, χL, keep)
    @inbounds for a in 1:χL, s in 1:d, m in 1:keep
        Gnew1[s, a, m] = invλL[a] * Umat[(a - 1) * d + s, m]
    end

    # Update G[i+1]: reshape Vh -> (keep, d, χR), divide by λR
    Gnew2 = zeros(ComplexF64, d, keep, χR)
    @inbounds for m in 1:keep, s in 1:d, b in 1:χR
        Gnew2[s, m, b] = Vh[m, (s - 1) * χR + b] * invλR[b]
    end

    G[i]     = Gnew1
    G[i + 1] = Gnew2
    return nothing
end

function tebd_sweep_parity!(l, G, expH, parity; chiMax=nothing)
    L = length(G)
    start = parity + 1
    for i in start:2:(L - 1)
        tebd_update_bond!(l, G, expH[i], i; chiMax=chiMax)
    end
end

function tebd_second_order_step!(l, G, expH_half, expH_full; chiMax=nothing)
    tebd_sweep_parity!(l, G, expH_half, 0; chiMax=chiMax)
    tebd_sweep_parity!(l, G, expH_full, 1; chiMax=chiMax)
    tebd_sweep_parity!(l, G, expH_half, 0; chiMax=chiMax)
end

############################################################
# Allocation-light bond / total energy
############################################################
function bond_energy(l, G, H_bond, pos)
    G1 = G[pos]
    G2 = G[pos + 1]
    λL = l[pos]
    λM = l[pos + 1]
    λR = l[pos + 2]
    H  = H_bond[pos]

    χL = length(λL)
    χM = length(λM)
    χR = length(λR)

    E = 0.0 + 0.0im
    v = zeros(ComplexF64, 9)

    @inbounds for a in 1:χL, b in 1:χR
        fill!(v, 0.0 + 0.0im)

        for m in 1:χM
            fac = λL[a] * λM[m] * λR[b]
            for s in 1:3, t in 1:3
                idx = (s - 1) * 3 + t
                v[idx] += fac * G1[s, a, m] * G2[t, m, b]
            end
        end

        E += dot(v, H * v)
    end

    return E
end



function total_energy(l, G, H_bond)
    E = 0.0 + 0.0im
    for i in 1:(length(G) - 1)
        E += bond_energy(l, G, H_bond, i)
    end
    return E
end

############################################################
# Exact diagonalization Hamiltonian
# Must match the corrected TEBD bond Hamiltonian.
############################################################

function kron_all(ops)
    out = ops[1]
    for op in ops[2:end]
        out = kron(out, op)
    end
    return out
end

function one_site(op, site, L, I3)
    ops = [I3 for _ in 1:L]
    ops[site] = op
    return kron_all(ops)
end

function two_site(op1, s1, op2, s2, L, I3)
    ops = [I3 for _ in 1:L]
    ops[s1] = op1
    ops[s2] = op2
    return kron_all(ops)
end

function fk_hamiltonian_ed(J, g, mu, L)
    B, Bd, U, I3 = fk_ops()
    n = number_op()
    U2 = U * U

    dim = 3^L
    H = zeros(ComplexF64, dim, dim)

    for j in 1:(L - 1)
        # Single-particle hopping
        H .+= -J * (1 - g) * two_site(Bd * U, j, B, j + 1, L, I3)
        H .+= -J * (1 - g) * two_site(adjoint(U) * B, j, Bd, j + 1, L, I3)

        # Pair hopping with string phase
        #H .+= -J * g * two_site(Bd * Bd * U2, j, B * B, j + 1, L, I3)
        #H .+= -J * g * two_site(adjoint(U2) * B * B, j, Bd * Bd, j + 1, L, I3)
        H .+= -J * g * two_site(Bd * Bd, j, B * B, j + 1, L, I3)
        H .+= -J * g * two_site(B * B, j, Bd * Bd, j + 1, L, I3)
    end

    for j in 1:L
        H .+= mu[j] * one_site(n, j, L, I3)
    end

    return H
end

function exact_ground_energy(J, g, mu, L)
    H = fk_hamiltonian_ed(J, g, mu, L)
    evals, evecs = eigen(Hermitian(H))
    return real(evals[1]), evals, evecs
end

############################################################
# Main test
############################################################

function main()
    Random.seed!(7)

    L = 6
    d = 3
    chiMax = 100

    J = 1.0
    g = 0.5
    W = 1.0

    mu = rand(L) .* (2W) .- W
    mu .= 0.0 .* mu   # set to zero for clean case; nonzero for disorder
    mu = [-0.84738342,  0.55983758, -0.12318154,  0.44693036,  0.95597902,  0.07699174]

    # ED
    E_ed, evals, evecs = exact_ground_energy(J, g,  mu, L)
    @printf("L = %d\n", L)
    println("mu = ", mu)
    @printf("ED ground-state energy = %.12f\n", E_ed)

    # TEBD imaginary-time evolution
    state = "ones"
    println("initial state = $state")
    l, G = initialMat(L, d, chiMax; state=state)

    schedule = [
        (0.10, 100),
        (0.05, 200),
        (0.02, 400),
        (0.01, 800),
        (0.005, 800),
        (0.002, 1500),
        (0.001, 2000),
    ]

    E_tebd = 0.0

    for (delta, steps) in schedule
        expH_half, H_bond = fk_expH_bond(J, g, mu, L, delta / 2.0)
        expH_full, _      = fk_expH_bond(J, g, mu, L, delta)

        for _ in 1:steps
            tebd_second_order_step!(l, G, expH_half, expH_full; chiMax=chiMax)
        end

        E_tebd = real(total_energy(l, G, H_bond))
        @printf("after delta=%7.4f, steps=%4d, E_TEBD=%.12f, err=%+.6e\n",
                delta, steps, E_tebd, E_tebd - E_ed)
    end

    println()
    @printf("Final ED energy   = %.12f\n", E_ed)
    @printf("Final TEBD energy = %.12f\n", E_tebd)
    @printf("Absolute error    = %.6e\n", abs(E_tebd - E_ed))
end

main()


