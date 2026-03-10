# Matrix-free ED (operator-on-the-fly) + Krylov time evolution
# Model: disordered Z3 Fock parafermion chain (nearest-neighbor)
#
# H = -J Σ_j [ (1-g) f_j† f_{j+1} + g (f_j†)^2 f_{j+1}^2 + h.c. ] + Σ_j μ_j n_j
#
# Observable: domain-wall imbalance I(t) = (2/L) (N_L - N_R)
#
# Requires: KrylovKit, JLD2
###############################################################################

using Random
using LinearAlgebra
using KrylovKit
using JLD2
using Statistics

# ----------------------------- Utilities -------------------------------------

"""
Precompute powers3[k] = 3^(k-1) for k=1..L, and dim = 3^L.
Basis index convention:
  idx in 1:dim corresponds to base-3 digits (n1..nL) where
  n_j = digit at site j (0,1,2).
"""
function precompute_powers3(L::Int)
    powers3 = Vector{Int}(undef, L)
    p = 1
    for j in 1:L
        powers3[j] = p
        p *= 3
    end
    return powers3, p  # p == 3^L
end

@inline function get_digit(idx0::Int, pow::Int)
    # idx0 is 0-based basis integer; pow = 3^(j-1)
    return (idx0 ÷ pow) % 3
end

@inline function add_to_index(idx0::Int, pow::Int, delta::Int)
    # change digit at site with weight pow by delta in {-2,-1,1,2}
    return idx0 + delta * pow
end

# ----------------------------- Initial state ---------------------------------

"""
Domain wall |1...1 0...0>, left half occupied with n=1.
Return psi0 as a ComplexF64 vector of length 3^L.
"""
function psi0_domainwall(L::Int, powers3::Vector{Int})
    @assert iseven(L)
    dim = 3^L
    idx0 = 0
    half = L ÷ 2
    for j in 1:half
        idx0 += 1 * powers3[j]  # set n_j = 1 on left half
    end
    ψ = zeros(ComplexF64, dim)
    ψ[idx0 + 1] = 1.0 + 0im
    return ψ
end

# ----------------------------- Observable ------------------------------------

"""
Compute imbalance I(t) = (2/L)*(N_L - N_R) for state ψ.
Here n_j ∈ {0,1,2} in computational basis.

This is O(dim*L). For L<=15 it's fine.
"""
function imbalance_domainwall(ψ::AbstractVector{ComplexF64}, L::Int, powers3::Vector{Int})
    dim = length(ψ)
    half = L ÷ 2
    NL = 0.0
    NR = 0.0
    @inbounds for idx in 1:dim
        w = abs2(ψ[idx])
        idx0 = idx - 1
        # left half
        for j in 1:half
            NL += w * get_digit(idx0, powers3[j])
        end
        # right half
        for j in (half+1):L
            NR += w * get_digit(idx0, powers3[j])
        end
    end
    return (2.0 / L) * (NL - NR)
end

# ----------------------------- Matrix-free H*v --------------------------------
#
# We define mul!(y, H, x) via explicit basis transitions:
#
# Onsite: μ_j n_j
# Hopping: (1-g) f_j† f_{j+1} + h.c.
# Pair hopping: g (f_j†)^2 f_{j+1}^2 + h.c.
#
# With the simple occupancy-ladder action:
#   f |n> = |n-1> for n>0, else 0
#   f†|n> = |n+1> for n<2, else 0
#   f^2 |n> = |n-2> for n>1, else 0
#   (f†)^2|n> = |n+2> for n<1, else 0
#
# If your ED convention uses different amplitudes (e.g. sqrt factors),
# change the constants amp1/amp2 below.

struct ParaEDOp
    L::Int
    J::Float64
    g::Float64
    μ::Vector{Float64}
    powers3::Vector{Int}
    dim::Int
end

function Base.size(A::ParaEDOp)
    return (A.dim, A.dim)
end

function LinearAlgebra.mul!(y::Vector{ComplexF64}, A::ParaEDOp, x::Vector{ComplexF64})
    L = A.L
    J = A.J
    g = A.g
    μ = A.μ
    p3 = A.powers3
    dim = A.dim

    fill!(y, 0.0 + 0im)

    # --- amplitudes (EDIT HERE if needed) ---
    amp1 = 1.0  # amplitude for single hop
    amp2 = 1.0  # amplitude for pair hop

    @inbounds for idx in 1:dim
        xi = x[idx]
        xi == 0 && continue
        idx0 = idx - 1

        # onsite diagonal: Σ μ_j n_j
        e = 0.0
        for j in 1:L
            nj = get_digit(idx0, p3[j])
            e += μ[j] * nj
        end
        y[idx] += e * xi

        # nearest-neighbor terms
        for j in 1:(L-1)
            powj  = p3[j]
            powjp = p3[j+1]
            nj  = get_digit(idx0, powj)
            njp = get_digit(idx0, powjp)

            # --- single hop: (1-g) f_j† f_{j+1} ---
            # requires nj <=1 and njp >=1
            if nj <= 1 && njp >= 1
                idx0p = add_to_index(idx0, powj, +1)
                idx0p = add_to_index(idx0p, powjp, -1)
                y[idx0p + 1] += (-J) * (1 - g) * amp1 * xi
            end

            # --- single hop h.c.: (1-g) f_{j+1}† f_j ---
            if nj >= 1 && njp <= 1
                idx0p = add_to_index(idx0, powj, -1)
                idx0p = add_to_index(idx0p, powjp, +1)
                y[idx0p + 1] += (-J) * (1 - g) * amp1 * xi
            end

            # --- pair hop: g (f_j†)^2 f_{j+1}^2 ---
            # requires nj == 0 and njp == 2
            if nj == 0 && njp == 2
                idx0p = add_to_index(idx0, powj, +2)
                idx0p = add_to_index(idx0p, powjp, -2)
                y[idx0p + 1] += (-J) * g * amp2 * xi
            end

            # --- pair hop h.c.: g (f_{j+1}†)^2 f_j^2 ---
            if nj == 2 && njp == 0
                idx0p = add_to_index(idx0, powj, -2)
                idx0p = add_to_index(idx0p, powjp, +2)
                y[idx0p + 1] += (-J) * g * amp2 * xi
            end
        end
    end

    return y
end


# ----------------------------- Make operator callable -------------------------

function (A::ParaEDOp)(y::Vector{ComplexF64}, x::Vector{ComplexF64})
    mul!(y, A, x)
    return y
end

function (A::ParaEDOp)(x::Vector{ComplexF64})
    y = similar(x)
    mul!(y, A, x)
    return y
end

# ----------------------------- Krylov time evolution --------------------------

"""
Evolve ψ(t+dt) = exp(-i*dt*H) ψ(t) using KrylovKit.exponentiate, matrix-free.
"""
function step_krylov!(ψ::Vector{ComplexF64}, A::ParaEDOp, dt::Float64;
                      krylovdim::Int=30, tol::Float64=1e-9)
    # exponentiate(A, t, v) computes exp(t*A)*v
    # We need exp(-i dt H)*ψ, so t = -im*dt and operator is H itself.
    ψnew, info = exponentiate(A, (-1im)*dt, ψ; krylovdim=krylovdim, tol=tol)
    ψ .= ψnew
    return ψ
end

# ----------------------------- Single disorder run ----------------------------

function run_one_realization(; L::Int, J::Float64, g::Float64, α::Float64, W::Float64,
                             tmax::Float64, dt::Float64, seed::Int,
                             krylovdim::Int=30, tol::Float64=1e-9)

    powers3, dim = precompute_powers3(L)

    rng = MersenneTwister(seed)
    μ = W .* (2 .* rand(rng, L) .- 1)

    H = ParaEDOp(L, J, g, μ, powers3, dim)
    ψ = psi0_domainwall(L, powers3)

    t = collect(0:dt:tmax)
    I = zeros(Float64, length(t))

    for (ti, tt) in enumerate(t)
        I[ti] = imbalance_domainwall(ψ, L, powers3)
        ti < length(t) && step_krylov!(ψ, H, dt; krylovdim=krylovdim, tol=tol)
    end

    return t, I, μ
end

# ----------------------------- Disorder average --------------------------------

function run_disorder_average(; L::Int, J::Float64=1.0, g::Float64=0.5,
                              W::Float64=4.5, nreal::Int=50,
                              tmax::Float64=50.0, dt::Float64=0.1,
                              seed::Int=1, krylovdim::Int=30, tol::Float64=1e-9)

    t_ref = nothing
    Iacc = nothing

    for r in 1:nreal
        t, I, μ = run_one_realization(L=L, J=J, g=g, α=Inf, W=W,
                                      tmax=tmax, dt=dt,
                                      seed=seed + 10_000*r,
                                      krylovdim=krylovdim, tol=tol)
        if r == 1
            t_ref = t
            Iacc = zeros(Float64, length(t))
        end
        Iacc .+= I
        println("realization $r / $nreal done")
    end

    Iavg = Iacc ./ nreal
    return t_ref, Iavg
end

# ----------------------------- Main (example) ---------------------------------

if abspath(PROGRAM_FILE) == @__FILE__

    # ---- parameters ----
    L_list = [8]     # capability demo: include one big L (e.g. 48)
    J = 1.0
    g = 0.5
    W_list = [2.5, 4.5, 6.0]

    tmax = 200.0
    dt   = 0.1

    nreal = 30
    seed  = 1

    krylovdim = 35
    tol = 1e-9

    for L in L_list, W in W_list
        println("\n=== Running matrix-free Krylov ED: L=$L, W=$W ===")
        t, I = run_disorder_average(L=L, J=J, g=g, W=W,
                                    nreal=nreal, tmax=tmax, dt=dt,
                                    seed=seed, krylovdim=krylovdim, tol=tol)

        outfile = "ED_Krylov_domainwall_Z3_L$(L)_W$(W)_g$(g).jld2"
        @save outfile t I L W g J tmax dt nreal krylovdim tol
        println("Saved → $outfile")
    end
end