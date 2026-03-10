using LinearAlgebra
using Random
using FFTW
using ProgressMeter
using Printf
using DelimitedFiles

BLAS.set_num_threads(1)

############################################################
# Z_n operators
############################################################

function zn_ops(n)
    ω = cis(2π/n)

    Z = Diagonal([ω^(a-1) for a in 1:n]) |> Matrix{ComplexF64}

    B = zeros(ComplexF64, n, n)
    for a in 2:n
        B[a-1, a] = 1
    end

    Bd = Matrix(adjoint(B))
    I_n = Matrix{ComplexF64}(I, n, n)

    proj = [zeros(ComplexF64, n, n) for _ in 1:n]
    for a in 1:n
        proj[a][a, a] = 1
    end

    return B, Bd, Z, proj, I_n
end

############################################################
# Single-hopping FK Hamiltonian (m = 1 only)
#
# H = -J Σ_j [ B_j† Z_j B_{j+1} + h.c. ]
############################################################

function bond_hamiltonians_single(J, L, n)
    B, Bd, Z, proj, I_n = zn_ops(n)

    Hbond = Vector{Matrix{ComplexF64}}(undef, L - 1)
    for j in 1:(L - 1)
        hop = kron(Bd * Z, B)
        Hbond[j] = -J * (hop + adjoint(hop))
    end

    return Hbond
end

function exp_gates(J, L, dt, n)
    Hbond = bond_hamiltonians_single(J, L, n)

    gates = Vector{Matrix{ComplexF64}}(undef, L - 1)
    for j in 1:(L - 1)
        gates[j] = exp(-im * dt * Hbond[j])
    end

    return gates
end

############################################################
# Utilities
############################################################

@inline function safe_inv_vec(v; eps=1e-12)
    out = similar(v)
    for i in eachindex(v)
        if abs(v[i]) > eps
            out[i] = inv(v[i])
        else
            out[i] = 0 + 0im
        end
    end
    return out
end

############################################################
# Domain-wall product MPS
#
# first half in |1>, second half in |n>
############################################################

function domain_wall_mps(L, n)
    G = Vector{Array{ComplexF64,3}}(undef, L)
    l = Vector{Vector{ComplexF64}}(undef, L + 1)

    for j in 1:L
        G[j] = zeros(ComplexF64, n, 1, 1)
        l[j] = ComplexF64[1]
    end
    l[L + 1] = ComplexF64[1]

    for j in 1:L
        if j <= div(L, 2)
            G[j][1, 1, 1] = 1
        else
            G[j][n, 1, 1] = 1
        end
    end

    return l, G
end

############################################################
# TEBD core
############################################################

function build_bond_matrix(G1, G2, λL, λM, λR, gate)
    d  = size(G1, 1)
    χL = size(G1, 2)
    χM = size(G1, 3)
    χR = size(G2, 3)

    M = zeros(ComplexF64, χL*d, d*χR)

    v = zeros(ComplexF64, d*d)
    w = zeros(ComplexF64, d*d)

    for a in 1:χL
        λLa = λL[a]

        for b in 1:χR
            λRb = λR[b]

            fill!(v, 0)

            for m in 1:χM
                fac = λLa * λM[m] * λRb

                for s in 1:d, t in 1:d
                    idx = (s-1)*d + t
                    v[idx] += fac * G1[s, a, m] * G2[t, m, b]
                end
            end

            mul!(w, gate, v)

            for s in 1:d, t in 1:d
                row = (a-1)*d + s
                col = (t-1)*χR + b
                idx = (s-1)*d + t
                M[row, col] = w[idx]
            end
        end
    end

    return M
end

function tebd_update_bond!(l, G, gate, i; chiMax=80)
    G1 = G[i]
    G2 = G[i+1]

    λL = l[i]
    λM = l[i+1]
    λR = l[i+2]

    χL = length(λL)
    χR = length(λR)
    d  = size(G1, 1)

    M = build_bond_matrix(G1, G2, λL, λM, λR, gate)

    F = svd(M)

    U  = F.U
    S  = F.S
    Vh = F.Vt

    keep = min(length(S), chiMax)

    U  = U[:, 1:keep]
    S  = S[1:keep]
    Vh = Vh[1:keep, :]

    nrm = norm(S)
    if nrm > 0
        S ./= nrm
    end

    l[i+1] = ComplexF64.(S)

    invλL = safe_inv_vec(λL)
    invλR = safe_inv_vec(λR)

    Gnew1 = zeros(ComplexF64, d, χL, keep)
    for a in 1:χL, s in 1:d, m in 1:keep
        Gnew1[s, a, m] = invλL[a] * U[(a-1)*d + s, m]
    end

    Gnew2 = zeros(ComplexF64, d, keep, χR)
    for m in 1:keep, s in 1:d, b in 1:χR
        Gnew2[s, m, b] = Vh[m, (s-1)*χR + b] * invλR[b]
    end

    G[i]   = Gnew1
    G[i+1] = Gnew2
end

function tebd_step!(l, G, gates; chiMax=80)
    L = length(G)

    for parity in 0:1
        for j in (parity+1):2:(L-1)
            tebd_update_bond!(l, G, gates[j], j; chiMax=chiMax)
        end
    end
end

############################################################
# Local expectation
############################################################

function local_exp(l, G, op, pos)
    λL = l[pos]
    λR = l[pos+1]
    A  = G[pos]

    χL = length(λL)
    χR = length(λR)
    d  = size(A, 1)

    ψ = zeros(ComplexF64, χL, d, χR)
    for a in 1:χL, s in 1:d, b in 1:χR
        ψ[a, s, b] = λL[a] * A[s, a, b] * λR[b]
    end

    ρ = zeros(ComplexF64, d, d)
    for s in 1:d, t in 1:d
        acc = 0 + 0im
        for a in 1:χL, b in 1:χR
            acc += conj(ψ[a, s, b]) * ψ[a, t, b]
        end
        ρ[s, t] = acc
    end

    return sum(ρ .* op) / tr(ρ)
end

############################################################
# Exact transfer through one site
#
# X acts on left bond indices of this site
# returns Y acting on right bond indices
############################################################

function apply_transfer(X::AbstractMatrix, l, G, site; op=nothing)
    A  = G[site]
    λL = l[site]
    λR = l[site+1]

    χL = length(λL)
    χR = length(λR)
    d  = size(A, 1)

    @assert size(X,1) == χL && size(X,2) == χL

    Y = zeros(ComplexF64, χR, χR)

    if op === nothing
        for β in 1:χR, βp in 1:χR
            acc = 0.0 + 0.0im
            for α in 1:χL, αp in 1:χL, s in 1:d
                Aleft  = λL[α]  * A[s, α,  β]  * λR[β]
                Aright = λL[αp] * A[s, αp, βp] * λR[βp]
                acc += conj(Aleft) * X[α, αp] * Aright
            end
            Y[β, βp] = acc
        end
    else
        for β in 1:χR, βp in 1:χR
            acc = 0.0 + 0.0im
            for α in 1:χL, αp in 1:χL, s in 1:d, t in 1:d
                Aleft  = λL[α]  * A[s, α,  β]  * λR[β]
                Aright = λL[αp] * A[t, αp, βp] * λR[βp]
                acc += conj(Aleft) * X[α, αp] * op[s, t] * Aright
            end
            Y[β, βp] = acc
        end
    end

    return Y
end

############################################################
# Exact two-point correlator <O_j O_k>
############################################################

function two_point_exp(l, G, op, j, k)
    L = length(G)

    if j > k
        j, k = k, j
    end

    X = reshape(ComplexF64[1.0 + 0.0im], 1, 1)

    if j == k
        op2 = op * op
        for site in 1:L
            if site == j
                X = apply_transfer(X, l, G, site; op=op2)
            else
                X = apply_transfer(X, l, G, site)
            end
        end
        return real(X[1,1])
    end

    for site in 1:L
        if site == j || site == k
            X = apply_transfer(X, l, G, site; op=op)
        else
            X = apply_transfer(X, l, G, site)
        end
    end

    return real(X[1,1])
end

############################################################
# Profile of <P_a(j)>
############################################################

function projector_profile(l, G, n, a)
    _, _, _, proj, _ = zn_ops(n)
    P = proj[a]

    L = length(G)
    vals = zeros(Float64, L)

    for j in 1:L
        vals[j] = real(local_exp(l, G, P, j))
    end

    return vals
end

############################################################
# Exact structure factor
#
# S_a(q,t) = (1/L) Σ_{j,k} exp[i q (j-k)] <P_a(j) P_a(k)>
############################################################

function structure_factor_a(l, G, n, a)
    _, _, _, proj, _ = zn_ops(n)
    P = proj[a]

    L = length(G)
    C = zeros(Float64, L, L)

    for j in 1:L
        for k in j:L
            val = two_point_exp(l, G, P, j, k)
            C[j, k] = val
            C[k, j] = val
        end
    end

    qs = 2π * (0:L-1) / L
    Sq = zeros(Float64, L)

    for iq in 1:L
        q = qs[iq]
        acc = 0.0
        for j in 1:L, k in 1:L
            acc += C[j,k] * cos(q * (j-k))
        end
        Sq[iq] = acc / L
    end

    return qs, Sq, C
end


function structure_factor_fast(l, G, n, a)

    _, _, _, proj, _ = zn_ops(n)
    P = proj[a]

    L = length(G)

    vals = zeros(Float64, L)

    for j in 1:L
        vals[j] = real(local_exp(l, G, P, j))
    end

    # connected correlations approximation
    C = zeros(Float64, L)

    for r in 0:(L-1)
        acc = 0.0
        count = 0

        for j in 1:(L-r)
            acc += vals[j] * vals[j+r]
            count += 1
        end

        C[r+1] = acc / count
    end

    Sq = real.(fft(C))
    qs = 2π*(0:L-1)/L

    return qs, Sq
end
############################################################
# Domain-wall imbalance for projector P_a
#
# I_a(t) = (2/L) [ sum_left P_a(j,t) - sum_right P_a(j,t) ]
############################################################

function imbalance_a_from_profile(vals::AbstractVector)
    L = length(vals)
    half = div(L, 2)

    left_sum  = sum(@view vals[1:half])
    right_sum = sum(@view vals[(half+1):L])

    return (2.0 / L) * (left_sum - right_sum)
end

############################################################
# Main simulation
############################################################

function main()
    Random.seed!(7)

    n_list = [3, 4, 5]

    L      = 24
    J      = 1.0
    dt     = 0.05
    steps  = 200
    chiMax = 80

    # projector flavor:
    # a = 1 means projector onto |1> in 1-based Julia indexing
    a_proj = 1

    for n in n_list
        println("\nRunning n = $n")

        gates = exp_gates(J, L, dt, n)
        l, G = domain_wall_mps(L, n)

        qs = 2π * (0:L-1) / L

        Sa_qt = zeros(Float64, L, steps)
        Ia_t  = zeros(Float64, steps)

        prog = Progress(steps)

        # using ProgressMeter
        for t in 1:steps
            tebd_step!(l, G, gates; chiMax=chiMax)

            qs, Sa = structure_factor_fast(l, G, n, a_proj)
            vals = projector_profile(l, G, n, a_proj)

            Sa_qt[:, t] .= Sa
            Ia_t[t] = imbalance_a_from_profile(vals)

            next!(prog)
        end

        times = (0:steps-1) .* dt

        ####################################################
        # Save imbalance dynamics
        ####################################################

        imbalance_data = hcat(times, Ia_t)
        outfile1 = @sprintf("imbalance_a_singlehop_domainwall_n%d_L%d.dat", n, L)
        writedlm(outfile1, imbalance_data)

        ####################################################
        # Save heatmap S_a(q,t)
        # format: first col = q, remaining cols = times
        ####################################################

        heatmap_a = hcat(qs, Sa_qt)
        outfile2 = @sprintf("structure_a_singlehop_domainwall_heatmap_n%d_L%d.dat", n, L)
        writedlm(outfile2, heatmap_a)

        println("Saved:")
        println(outfile1)
        println(outfile2)
    end
end

main()