# ed_parafermion_mbl_Z3.jl
#
# Exact time evolution (matrix-free Krylov) for the DISORDERED Z3 Fock-parafermion chain:
#
#   H = -J ∑_j [ (1-g) f†_j f_{j+1} + g (f†_j)^2 f_{j+1}^2 + h.c. ] + ∑_j μ_j n_j
#
# We work in the Fock parafermion (FPF) occupation basis |a1,a2,...,aL>, with a_j ∈ {0,1,2}.
# The parafermion exchange statistics enter through Jordan–Wigner strings:
#   f_j = (∏_{k<j} ω^{n_k}) a_j
# where a_j is a local "lowering" operator:
#   a|0>=0, a|1>=|0>, a|2>=√2 |1>
# and a† raises with the same √ factors. (This matches your Hubbard mapping
# f = X^{01} + √2 X^{12}, f† = X^{10}+√2 X^{21}.)
#
# Then, for nearest-neighbor hopping:
#   f†_j f_{j+1} = ω^{n_j} a†_j a_{j+1}
#   f†_{j+1} f_j = ω^{-n_j} a†_{j+1} a_j
# and similarly for pair hopping:
#   (f†_j)^2 f_{j+1}^2 = ω^{2 n_j} (a†_j)^2 (a_{j+1})^2
#   (f†_{j+1})^2 f_j^2 = ω^{-2 n_j} (a†_{j+1})^2 (a_j)^2
#
# Observables:
# - density profile ⟨n_j(t)⟩, with n_j = a_j (occupation number) in {0,1,2}
# - imbalance I(t) = (2/L) [ N_L(t) - N_R(t) ]
#
# Disorder average: run Ndis samples μ_j ~ Uniform(-W,W), compute I(t) per sample,
# store mean + standard error over disorder.
#
# Output: JLD2 file with times, I_mean, I_err, and optionally n_mean[j,t].

using LinearAlgebra
using LinearMaps
using KrylovKit
using Random
using JLD2

# ----------------------------
# Parameters container
# ----------------------------
struct PFDisorderParams
    L::Int
    J::Float64
    g::Float64
    W::Float64
    mu::Vector{Float64}         # length L
    powN::Vector{Int}           # powN[s] = n^(s-1)
end

const n = 3
const ω = cis(2π / n)          # exp(2π i / 3)

# ----------------------------
# Basis utilities: a_j ∈ {0,1,2} (Z-basis digit)
# ----------------------------
@inline function digit_of(state::Int, site::Int, powN::Vector{Int})
    return (state ÷ powN[site]) % n
end

@inline function set_digit(state::Int, site::Int, newa::Int, powN::Vector{Int})
    olda = digit_of(state, site, powN)
    return state + (newa - olda) * powN[site]
end

# local ladder prefactors for a and a† (qutrit-like: √m)
@inline function pref_lower(a::Int)::Float64
    # a |a> -> √a |a-1>, with √0=0, √1=1, √2=√2
    a == 0 && return 0.0
    a == 1 && return 1.0
    return sqrt(2.0) # a==2
end

@inline function pref_raise(a::Int)::Float64
    # a† |a> -> √(a+1) |a+1>, but truncated at 2
    a == 2 && return 0.0
    a == 0 && return 1.0
    return sqrt(2.0) # a==1
end

# Apply a_j on site: returns (new_state, prefactor) or (state,0) if not possible
@inline function apply_lower(state::Int, site::Int, powN::Vector{Int})
    a = digit_of(state, site, powN)
    p = pref_lower(a)
    p == 0.0 && return state, 0.0
    return set_digit(state, site, a-1, powN), p
end

# Apply a†_j on site
@inline function apply_raise(state::Int, site::Int, powN::Vector{Int})
    a = digit_of(state, site, powN)
    p = pref_raise(a)
    p == 0.0 && return state, 0.0
    return set_digit(state, site, a+1, powN), p
end

# Apply (a_j)^2 on site
@inline function apply_lower2(state::Int, site::Int, powN::Vector{Int})
    a = digit_of(state, site, powN)
    # a^2 |2> -> √2 |0>, else 0
    a == 2 || return state, 0.0
    return set_digit(state, site, 0, powN), sqrt(2.0)
end

# Apply (a†_j)^2 on site
@inline function apply_raise2(state::Int, site::Int, powN::Vector{Int})
    a = digit_of(state, site, powN)
    # (a†)^2 |0> -> √2 |2>, else 0
    a == 0 || return state, 0.0
    return set_digit(state, site, 2, powN), sqrt(2.0)
end

# ----------------------------
# Diagonal disorder energy: ∑ μ_j n_j, where n_j = a_j
# ----------------------------
@inline function diag_disorder_energy(state::Int, p::PFDisorderParams)
    E = 0.0
    @inbounds for j in 1:p.L
        aj = digit_of(state, j, p.powN)
        E += p.mu[j] * aj
    end
    return E
end

# ----------------------------
# Matrix-free action y = H*v
# ----------------------------
function mul_H!(y::Vector{ComplexF64}, v::Vector{ComplexF64}, p::PFDisorderParams)
    fill!(y, 0.0 + 0.0im)
    dim = length(v)
    L = p.L
    J = p.J
    g = p.g

    # hopping coefficients
    t1 = -J * (1 - g)   # multiplies (f†_j f_{j+1} + h.c.)
    t2 = -J * g         # multiplies ((f†)^2 f^2 + h.c.)

    @inbounds for idx in 1:dim
        s  = idx - 1
        vs = v[idx]
        vs == 0 && continue

        # diagonal disorder term
        y[idx] += diag_disorder_energy(s, p) * vs

        # nearest-neighbor terms
        for j in 1:(L-1)
            nj = digit_of(s, j, p.powN)  # needed for string phase ω^{± n_j}

            # ---------- single hopping: f†_j f_{j+1} = ω^{n_j} a†_j a_{j+1} ----------
            s1, p_low = apply_lower(s, j+1, p.powN)   # a_{j+1}
            if p_low != 0.0
                s2, p_r = apply_raise(s1, j, p.powN)  # a†_j
                if p_r != 0.0
                    phase = ω^nj
                    amp   = t1 * phase * (p_low * p_r)
                    y[s2+1] += amp * vs
                end
            end

            # ---------- hermitian conjugate: f†_{j+1} f_j = ω^{-n_j} a†_{j+1} a_j ----------
            s1, p_low = apply_lower(s, j, p.powN)     # a_j
            if p_low != 0.0
                s2, p_r = apply_raise(s1, j+1, p.powN) # a†_{j+1}
                if p_r != 0.0
                    phase = ω^(-nj)
                    amp   = t1 * phase * (p_low * p_r)
                    y[s2+1] += amp * vs
                end
            end

            # ---------- pair hopping: (f†_j)^2 f_{j+1}^2 = ω^{2 n_j} (a†_j)^2 (a_{j+1})^2 ----------
            s1, p_low2 = apply_lower2(s, j+1, p.powN)   # (a_{j+1})^2
            if p_low2 != 0.0
                s2, p_r2 = apply_raise2(s1, j, p.powN)  # (a†_j)^2
                if p_r2 != 0.0
                    phase = ω^(2*nj)
                    amp   = t2 * phase * (p_low2 * p_r2)
                    y[s2+1] += amp * vs
                end
            end

            # ---------- hermitian conjugate pair: (f†_{j+1})^2 f_j^2 = ω^{-2 n_j} (a†_{j+1})^2 (a_j)^2 ----------
            s1, p_low2 = apply_lower2(s, j, p.powN)     # (a_j)^2
            if p_low2 != 0.0
                s2, p_r2 = apply_raise2(s1, j+1, p.powN) # (a†_{j+1})^2
                if p_r2 != 0.0
                    phase = ω^(-2*nj)
                    amp   = t2 * phase * (p_low2 * p_r2)
                    y[s2+1] += amp * vs
                end
            end
        end
    end

    return y
end

# ----------------------------
# Time evolution: KrylovKit exponentiate
# ----------------------------
function evolve_times(psi0::Vector{ComplexF64}, p::PFDisorderParams, times::Vector{Float64};
                      krylovdim::Int=60, tol::Float64=1e-10, max_dt::Float64=0.1)

    dim = length(psi0)

    H = LinearMap{ComplexF64}(
        (y, v) -> mul_H!(y, v, p),
        dim, dim; ismutating=true
    )

    function exp_step_substepped(ψ, dt)
        nsub = max(1, ceil(Int, abs(dt)/max_dt))
        δ = dt / nsub
        for _ in 1:nsub
            ψ, = exponentiate(H, -1im*δ, ψ; krylovdim=krylovdim, tol=tol)
        end
        return ψ
    end

    psit = copy(psi0)
    out = Vector{Vector{ComplexF64}}(undef, length(times))
    tprev = 0.0

    for (k, t) in enumerate(times)
        dt = t - tprev
        psit = exp_step_substepped(psit, dt)

        # optional: normalize as a safety net
        psit ./= norm(psit)

        out[k] = psit
        tprev = t

        k % 10 == 0 && println("ED step $k / $(length(times))")
    end
    return out
end

# ----------------------------
# Measurements: ⟨n_j⟩ and imbalance I(t)
# ----------------------------
function measure_n_profile(psi::Vector{ComplexF64}, p::PFDisorderParams)
    L = p.L
    nprof = zeros(Float64, L)
    @inbounds for idx in 1:length(psi)
        s = idx - 1
        w = abs2(psi[idx])
        w == 0 && continue
        for j in 1:L
            aj = digit_of(s, j, p.powN)
            nprof[j] += aj * w
        end
    end
    return nprof
end

function imbalance_from_n(nprof::AbstractVector{<:Real})
    L = length(nprof)
    half = L ÷ 2
    NL = sum(@view nprof[1:half])
    NR = sum(@view nprof[half+1:L])
    return (2.0 / L) * (NL - NR)
end

# ----------------------------
# Initial state: domain wall at half filling
# |1,1,...,1, 0,0,...,0>
# ----------------------------
function domainwall_state(L::Int, powN::Vector{Int})
    s = 0
    half = L ÷ 2
    for j in 1:half
        s += 1 * powN[j]  # put a_j = 1 on left half
    end
    return s
end

# ----------------------------
# Run: disorder average
# ----------------------------
# system parameters
L      = 8            # use even L for clean domain wall
J      = 1.0
gpair  = 0.3           # g in your paper (0..1)
W      = 4.0           # disorder strength
Ndis   = 50            # number of disorder realizations (increase for publication)
seed   = 1234

# time grid
tmax   = 200.0
Nt     = 101
times  = collect(range(0.0, tmax, length=Nt))

# Krylov settings
krylovdim = 50
tol       = 1e-11

# precompute powers
powN = [n^(s-1) for s in 1:L]
dim  = n^L

# initial state vector
psi0 = zeros(ComplexF64, dim)
s0   = domainwall_state(L, powN)
psi0[s0 + 1] = 1.0 + 0im

# disorder RNG
rng = MersenneTwister(seed)

# accumulators for disorder mean + SEM
I_sum  = zeros(Float64, Nt)
I_sum2 = zeros(Float64, Nt)

n_sum  = zeros(Float64, L, Nt)     # optional: disorder-avg density profiles

for r in 1:Ndis
    mu = (2W) .* rand(rng, L) .- W

    p = PFDisorderParams(L, J, gpair, W, mu, powN)

    psis = evolve_times(psi0, p, times; krylovdim=krylovdim, tol=tol, max_dt=0.05)

    for (k, ψ) in enumerate(psis)
        nprof = measure_n_profile(ψ, p)
        I = imbalance_from_n(nprof)

        I_sum[k]  += I
        I_sum2[k] += I^2
        n_sum[:,k] .+= nprof
    end

    println("Disorder sample $r / $Ndis done")
end

# means
I_mean = I_sum ./ Ndis
n_mean = n_sum ./ Ndis

# standard error over disorder realizations
I_var = (I_sum2 ./ Ndis) .- I_mean.^2
I_err = sqrt.(max.(I_var, 0.0) ./ Ndis)

println("Imbalance I(t_final) = ", I_mean[end], " ± ", I_err[end])

outfile = "ED_Z3_FPF_MBL_L$(L)_g$(gpair)_W$(W)_Ndis$(Ndis).jld2"
@save outfile n L J gpair W Ndis seed times I_mean I_err n_mean
println("Saved to: ", outfile)