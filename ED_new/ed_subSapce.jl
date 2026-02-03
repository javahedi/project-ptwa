###############################################################################
# Matrix-free ED in fixed-N sector + Krylov time evolution
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
    return (idx0 ÷ pow) % 3
end

@inline function add_to_index(idx0::Int, pow::Int, delta::Int)
    return idx0 + delta * pow
end

@inline function sum_digits_base3(idx0::Int, powers3::Vector{Int})
    s = 0
    @inbounds for pow in powers3
        s += (idx0 ÷ pow) % 3
    end
    return s
end

# ------------------------ Fixed-N basis construction --------------------------

"""
Build the reduced basis for fixed total number Ntarget.
Returns:
  full_to_red :: Vector{Int}  (length dim_full, 0 if not in sector)
  red_to_full :: Vector{Int}  (length dim_red, stores idx0 (0-based) of each basis state)
  dim_red
"""
function build_fixedN_basis(L::Int, Ntarget::Int, powers3::Vector{Int})
    dim_full = 3^L
    full_to_red = zeros(Int, dim_full)  # 0 means "not in sector"
    red_to_full = Int[]

    # NOTE: O(3^L * L). Fine for L ~ 12–15. If you go bigger, use a recursive generator.
    for idx0 in 0:(dim_full-1)
        if sum_digits_base3(idx0, powers3) == Ntarget
            push!(red_to_full, idx0)
            full_to_red[idx0 + 1] = length(red_to_full)
        end
    end

    return full_to_red, red_to_full, length(red_to_full)
end

# ----------------------------- Initial state ---------------------------------

"""
Domain wall |1...1 0...0>, left half has n=1. Total N = L/2.
Return reduced-basis ψ0 (length dim_red) with a single 1 entry.
"""
function psi0_domainwall_fixedN(L::Int, powers3::Vector{Int},
                                full_to_red::Vector{Int})
    @assert iseven(L)
    half = L ÷ 2

    idx0 = 0
    for j in 1:half
        idx0 += 1 * powers3[j]  # set n_j = 1 on left half
    end

    ridx = full_to_red[idx0 + 1]
    @assert ridx != 0 "Domain-wall state not found in reduced basis (check Ntarget)."

    ψ = zeros(ComplexF64, maximum(full_to_red))  # dim_red
    ψ[ridx] = 1.0 + 0im
    return ψ
end

# ----------------------------- Observable ------------------------------------

"""
Imbalance in reduced basis:
I(t) = (2/L) (N_L - N_R), with N_L = Σ_{j<=L/2} <n_j>, etc.

Cost: O(dim_red * L)
"""
function imbalance_domainwall_fixedN(ψ::AbstractVector{ComplexF64},
                                     L::Int,
                                     powers3::Vector{Int},
                                     red_to_full::Vector{Int})
    half = L ÷ 2
    NL = 0.0
    NR = 0.0

    @inbounds for ridx in eachindex(ψ)
        w = abs2(ψ[ridx])
        idx0 = red_to_full[ridx]

        for j in 1:half
            NL += w * get_digit(idx0, powers3[j])
        end
        for j in (half+1):L
            NR += w * get_digit(idx0, powers3[j])
        end
    end

    return (2.0 / L) * (NL - NR)
end

# ----------------------------- Matrix-free H*v (fixed-N) ----------------------

struct ParaEDOpFixedN
    L::Int
    J::Float64
    g::Float64
    μ::Vector{Float64}
    powers3::Vector{Int}
    full_to_red::Vector{Int}   # full idx0+1 -> red index or 0
    red_to_full::Vector{Int}   # red index -> full idx0
    dim_red::Int
end

Base.size(A::ParaEDOpFixedN) = (A.dim_red, A.dim_red)

function LinearAlgebra.mul!(y::Vector{ComplexF64}, A::ParaEDOpFixedN, x::Vector{ComplexF64})
    L = A.L
    J = A.J
    g = A.g
    μ = A.μ
    p3 = A.powers3
    f2r = A.full_to_red
    r2f = A.red_to_full
    dim = A.dim_red

    fill!(y, 0.0 + 0im)

    # amplitudes (adjust only if your ED convention differs)
    amp1 = 1.0
    amp2 = 1.0

    @inbounds for ridx in 1:dim
        xi = x[ridx]
        xi == 0 && continue

        idx0 = r2f[ridx]  # full basis idx0 (0-based)

        # onsite diagonal: Σ μ_j n_j
        e = 0.0
        for j in 1:L
            nj = get_digit(idx0, p3[j])
            e += μ[j] * nj
        end
        y[ridx] += e * xi

        # nearest-neighbor hopping/pair hopping
        for j in 1:(L-1)
            powj  = p3[j]
            powjp = p3[j+1]
            nj  = get_digit(idx0, powj)
            njp = get_digit(idx0, powjp)

            # single hop: (1-g) f_j† f_{j+1}
            if nj <= 1 && njp >= 1
                idx0p = add_to_index(idx0, powj, +1)
                idx0p = add_to_index(idx0p, powjp, -1)
                rp = f2r[idx0p + 1]
                @assert rp != 0
                y[rp] += (-J) * (1 - g) * amp1 * xi
            end

            # h.c.: (1-g) f_{j+1}† f_j
            if nj >= 1 && njp <= 1
                idx0p = add_to_index(idx0, powj, -1)
                idx0p = add_to_index(idx0p, powjp, +1)
                rp = f2r[idx0p + 1]
                @assert rp != 0
                y[rp] += (-J) * (1 - g) * amp1 * xi
            end

            # pair hop: g (f_j†)^2 f_{j+1}^2
            if nj == 0 && njp == 2
                idx0p = add_to_index(idx0, powj, +2)
                idx0p = add_to_index(idx0p, powjp, -2)
                rp = f2r[idx0p + 1]
                @assert rp != 0
                y[rp] += (-J) * g * amp2 * xi
            end

            # h.c.: g (f_{j+1}†)^2 f_j^2
            if nj == 2 && njp == 0
                idx0p = add_to_index(idx0, powj, -2)
                idx0p = add_to_index(idx0p, powjp, +2)
                rp = f2r[idx0p + 1]
                @assert rp != 0
                y[rp] += (-J) * g * amp2 * xi
            end
        end
    end

    return y
end

# Make operator callable for KrylovKit
(A::ParaEDOpFixedN)(y::Vector{ComplexF64}, x::Vector{ComplexF64}) = mul!(y, A, x)
(A::ParaEDOpFixedN)(x::Vector{ComplexF64}) = (y = similar(x); mul!(y, A, x); y)

# ----------------------------- Krylov time evolution --------------------------

function step_krylov!(ψ::Vector{ComplexF64}, A::ParaEDOpFixedN, dt::Float64;
                      krylovdim::Int=30, tol::Float64=1e-9)
    ψnew, info = exponentiate(A, (-1im)*dt, ψ; krylovdim=krylovdim, tol=tol)
    ψ .= ψnew
    return ψ
end

# ----------------------------- Single disorder run ----------------------------

function run_one_realization_fixedN(; L::Int, J::Float64, g::Float64, W::Float64,
                                    tmax::Float64, dt::Float64, seed::Int,
                                    krylovdim::Int=30, tol::Float64=1e-9)

    powers3, dim_full = precompute_powers3(L)
    Ntarget = L ÷ 2  # for domain-wall |1...10...0|

    full_to_red, red_to_full, dim_red = build_fixedN_basis(L, Ntarget, powers3)
    println("Fixed-N sector: L=$L, N=$Ntarget => dim_red=$dim_red (full 3^L=$dim_full)")

    rng = MersenneTwister(seed)
    μ = W .* (2 .* rand(rng, L) .- 1)

    H = ParaEDOpFixedN(L, J, g, μ, powers3, full_to_red, red_to_full, dim_red)
    ψ = psi0_domainwall_fixedN(L, powers3, full_to_red)

    t = collect(0:dt:tmax)
    I = zeros(Float64, length(t))

    for (ti, _) in enumerate(t)
        I[ti] = imbalance_domainwall_fixedN(ψ, L, powers3, red_to_full)
        ti < length(t) && step_krylov!(ψ, H, dt; krylovdim=krylovdim, tol=tol)
    end

    return t, I, μ, dim_red
end

# ----------------------------- Disorder average --------------------------------

function run_disorder_average_fixedN(; L::Int, J::Float64=1.0, g::Float64=0.5,
                                     W::Float64=4.5, nreal::Int=30,
                                     tmax::Float64=50.0, dt::Float64=0.1,
                                     seed::Int=1, krylovdim::Int=30, tol::Float64=1e-9)

    t_ref = nothing
    Iacc = nothing
    dim_red_saved = nothing

    for r in 1:nreal
        t, I, μ, dim_red = run_one_realization_fixedN(
            L=L, J=J, g=g, W=W, tmax=tmax, dt=dt,
            seed=seed + 10_000*r, krylovdim=krylovdim, tol=tol
        )
        if r == 1
            t_ref = t
            Iacc = zeros(Float64, length(t))
            dim_red_saved = dim_red
        end
        Iacc .+= I
        println("realization $r / $nreal done")
    end

    return t_ref, Iacc ./ nreal, dim_red_saved
end

# ----------------------------- Main -------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__

    L_list = [12]  # try 12, 14, 15 (fixed-N helps a lot)
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
        println("\n=== Fixed-N Krylov ED: L=$L, W=$W ===")
        t, I, dim_red = run_disorder_average_fixedN(
            L=L, J=J, g=g, W=W, nreal=nreal,
            tmax=tmax, dt=dt, seed=seed,
            krylovdim=krylovdim, tol=tol
        )

        outfile = "ED_Krylov_fixedN_domainwall_Z3_L$(L)_W$(W)_g$(g).jld2"
        @save outfile t I L W g J tmax dt nreal krylovdim tol dim_red
        println("Saved → $outfile (dim_red=$dim_red)")
    end
end
