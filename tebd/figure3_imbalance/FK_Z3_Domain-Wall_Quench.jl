using LinearAlgebra
using Random
using Printf
using DelimitedFiles

BLAS.set_num_threads(1)

############################################################
# FK local operators
############################################################

function fk_ops()
    w = cis(2π / 3)

    U = ComplexF64[
        1  0   0
        0  w   0
        0  0  w^2
    ]

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

@inline function safe_inv_vec(v::AbstractVector{ComplexF64}; eps=1e-10)
    out = similar(v)
    @inbounds for i in eachindex(v)
        if abs(v[i]) > eps
            out[i] = inv(v[i])
        else
            out[i] = 0.0 + 0.0im
        end
    end
    return out
end
############################################################
# Bond Hamiltonians / real-time gates
############################################################

function bond_hamiltonians(J, g, mu, L)
    B, Bd, U, I3 = fk_ops()
    n = number_op()
    U2 = U * U

    Hbond = Vector{Matrix{ComplexF64}}(undef, L - 1)

    for j in 1:(L - 1)
        H = zeros(ComplexF64, 9, 9)

        # Single-particle hopping: B_j† U_j B_{j+1} + h.c.
        hop1 = kron(Bd * U, B)
        H .+= -J * (1.0 - g) * (hop1 + adjoint(hop1))

        # Pair hopping: (B_j†)^2 U_j^2 B_{j+1}^2 + h.c.
        hop2 = kron(Bd * Bd * U2, B * B)
        H .+= -J * g * (hop2 + adjoint(hop2))

        # Onsite disorder split between bonds
        wl = (j == 1)     ? 1.0 : 0.5
        wr = (j == L - 1) ? 1.0 : 0.5
        H .+= wl * mu[j]     * kron(n, I3)
        H .+= wr * mu[j + 1] * kron(I3, n)

        Hbond[j] = H
    end

    return Hbond
end

function exp_gates(J, g, mu, L, dt)
    Hbond = bond_hamiltonians(J, g, mu, L)
    gates = Vector{Matrix{ComplexF64}}(undef, L - 1)

    for j in 1:(L - 1)
        gates[j] = exp(-im * dt * Hbond[j])
    end

    return gates
end

############################################################
# Domain-wall initial MPS (Vidal form)
############################################################

function domain_wall_mps(L)
    d = 3
    G = Vector{Array{ComplexF64,3}}(undef, L)
    l = Vector{Vector{ComplexF64}}(undef, L + 1)

    for j in 1:L
        G[j] = zeros(ComplexF64, d, 1, 1)
        l[j] = ComplexF64[1.0 + 0im]
    end
    l[L + 1] = ComplexF64[1.0 + 0im]

    for j in 1:L
        if j <= div(L, 2)
            G[j][2, 1, 1] = 1.0 + 0im   # "ones" on left half
        else
            G[j][1, 1, 1] = 1.0 + 0im   # "vacuum" on right half
        end
    end

    return l, G
end

############################################################
# TEBD update (9x9 gate version)
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

function tebd_update_bond!(l, G, gate, i; chiMax=60)
    G1 = G[i]
    G2 = G[i + 1]

    λL = l[i]
    λM = l[i + 1]
    λR = l[i + 2]

    χL = length(λL)
    χR = length(λR)
    d  = size(G1, 1)

    M = build_bond_matrix(G1, G2, λL, λM, λR, gate)

    if any(!isfinite, M)
        error("Non-finite values detected in TEBD matrix")
    end

    F = svd(M; alg=LinearAlgebra.QRIteration())

    Umat = F.U
    S    = F.S
    Vh   = F.Vt

    keep = min(length(S), chiMax)

    Umat = Umat[:, 1:keep]
    S    = S[1:keep]
    Vh   = Vh[1:keep, :]

    # normalize Schmidt values
    nrm = norm(S)
    if nrm > 0
        S ./= nrm
    end

    # regularization
    eps = 1e-12
    for k in eachindex(S)
        if S[k] < eps
            S[k] = eps
        end
    end

    l[i + 1] = ComplexF64.(S)

    invλL = safe_inv_vec(λL)
    invλR = safe_inv_vec(λR)

    Gnew1 = zeros(ComplexF64, d, χL, keep)
    @inbounds for a in 1:χL, s in 1:d, m in 1:keep
        Gnew1[s, a, m] = invλL[a] * Umat[(a - 1) * d + s, m]
    end

    Gnew2 = zeros(ComplexF64, d, keep, χR)
    @inbounds for m in 1:keep, s in 1:d, b in 1:χR
        Gnew2[s, m, b] = Vh[m, (s - 1) * χR + b] * invλR[b]
    end

    G[i]     = Gnew1
    G[i + 1] = Gnew2
    return nothing
end

function tebd_step!(l, G, gates; chiMax=60)
    L = length(G)
    for parity in 0:1
        start = parity + 1
        for j in start:2:(L - 1)
            tebd_update_bond!(l, G, gates[j], j; chiMax=chiMax)
        end
    end
end

############################################################
# Local expectation value
############################################################

function local_exp(l, G, op, pos)
    # pos is 1-based
    λL = l[pos]
    λR = l[pos + 1]
    A  = G[pos]

    χL = length(λL)
    χR = length(λR)

    ψ = zeros(ComplexF64, χL, 3, χR)

    @inbounds for a in 1:χL, s in 1:3, b in 1:χR
        ψ[a, s, b] = λL[a] * A[s, a, b] * λR[b]
    end

    ρ = zeros(ComplexF64, 3, 3)
    @inbounds for s in 1:3, t in 1:3
        acc = 0.0 + 0.0im
        for a in 1:χL, b in 1:χR
            acc += conj(ψ[a, s, b]) * ψ[a, t, b]
        end
        ρ[s, t] = acc
    end

    num = sum(ρ .* op)
    den = tr(ρ)

    return real(num / den)
end

############################################################
# Imbalance
############################################################

function imbalance(l, G)
    L = length(G)
    n = number_op()

    NL = 0.0
    NR = 0.0

    for j in 1:div(L, 2)
        NL += local_exp(l, G, n, j)
    end

    for j in (div(L, 2) + 1):L
        NR += local_exp(l, G, n, j)
    end

    return 2.0 * (NL - NR) / L
end

############################################################
# Main simulation
############################################################

function main()
    Random.seed!(7)

    L       = 10
    J       = 1.0
    g       = 0.3
    W       = 8.0
    dt      = 0.1
    steps   = 500
    samples = 100
    chiMax  = 100

    imbalance_avg = zeros(Float64, steps)
    times = collect(0:(steps - 1)) .* (dt * J)

    for r in 1:samples
        mu = rand(L) .* (2W) .- W

        gates = exp_gates(J, g, mu, L, dt)
        l, G = domain_wall_mps(L)

        for t in 1:steps
            tebd_step!(l, G, gates; chiMax=chiMax)
            imbalance_avg[t] += imbalance(l, G)
        end

        @printf("sample %d done\n", r)
    end

    imbalance_avg ./= samples

    data = hcat(times, imbalance_avg)
    outfile = @sprintf("imbalance_fk_z3_L%d_g%.2f_W%.2f_U2_julia.dat", L, g, W)
    writedlm(outfile, data)

    println("\nSaved data to: $outfile")
    println("First few rows:")
    for i in 1:min(5, steps)
        @printf("%10.6f  %14.10f\n", data[i,1], data[i,2])
    end
end

main()