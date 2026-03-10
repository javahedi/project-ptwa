###############################################################################
# Correct pTWA for disordered NN Z3 Fock-parafermion chain (string phases correct)
#
# Usage:
#   julia --threads=N run_pTWA_Z3_parafermion_MBL_correct.jl L W Ndis Nmc tmax seed OUTDIR
###############################################################################

using LinearAlgebra
using Random
using Statistics
using JLD2
using Base.Threads

include("sampling.jl")
using .Sampling

# ----------------------------
# Sampling choice
# ----------------------------
sampling = :discrete   # :discrete or :gaussian
cache = (sampling == :discrete) ? build_discrete_cache() : nothing

# ----------------------------
# Constants / local operators
# ----------------------------
const n = 3
const ω = cis(2π / n)
const SQRT2 = sqrt(2.0)

# n̂ = diag(0,1,2)
const N̂ = Diagonal(ComplexF64[0, 1, 2])

# truncated ladder a: a|1>=|0>, a|2>=√2|1>
const â = ComplexF64[
    0  1     0
    0  0  SQRT2
    0  0     0
]
const âd = adjoint(â)

# a^2: a^2|2>=√2|0>
const â2  = ComplexF64[
    0  0  SQRT2
    0  0     0
    0  0     0
]
const â2d = adjoint(â2)

# ω^{n̂}, ω^{-n̂}, ω^{2 n̂}, ω^{-2 n̂}
const ωN   = Diagonal(ComplexF64[1, ω, ω^2])
const ωNm  = Diagonal(ComplexF64[1, conj(ω), conj(ω^2)])
const ω2N  = Diagonal(ComplexF64[1, ω^2, ω])
const ω2Nm = Diagonal(ComplexF64[1, conj(ω^2), conj(ω)])

@inline function trAB(A::AbstractMatrix{ComplexF64}, B::AbstractMatrix{ComplexF64})
    # trace(A*B) without allocating A*B
    s = 0.0 + 0.0im
    @inbounds for i in 1:3, k in 1:3
        s += A[i,k] * B[k,i]
    end
    return s
end

@inline function normalize_site!(xj::Matrix{ComplexF64})
    # enforce Hermiticity + trace 1 (stabilizes integration)
    xj .= (xj .+ xj') ./ 2
    tr = real(xj[1,1] + xj[2,2] + xj[3,3])
    if !(isfinite(tr)) || abs(tr) < 1e-14
        # If trace is garbage, let it crash clearly later
        return
    end
    xj ./= tr
end

# ----------------------------
# Domain-wall initial sampling using your sampling.jl
# left half |1>, right half |0|
# ----------------------------
function sample_domainwall(method::Symbol, L; cache=nothing, rng=Random.default_rng())
    half = L ÷ 2
    x0 = Vector{Matrix{ComplexF64}}(undef, L)

    if method == :gaussian
        for j in 1:L
            a0 = (j <= half) ? 1 : 0
            x0[j] = Sampling.sample_site_gaussian(a0; rng=rng)
            normalize_site!(x0[j])
        end
        return x0
    elseif method == :discrete
        @assert cache !== nothing
        for j in 1:L
            W = (j <= half) ? cache.W1 : cache.W0
            idx = Sampling.sample_index(W, rng)
            x0[j] = copy(cache.x_lookup[idx])
            normalize_site!(x0[j])
        end
        return x0
    else
        error("Unknown sampling method: $method")
    end
end

# ----------------------------
# Params (one disorder realization)
# ----------------------------
struct PTWAPFParams
    L::Int
    J::Float64
    g::Float64
    mu::Vector{Float64}
end

# ----------------------------
# Build h_j(x) bond-by-bond, Hermitian by construction, with correct string phases.
#
# Bond (j,j+1) single-hop:
#   t1 [ ω^{n_j} a†_j a_{j+1} + (ω^{n_j} a†_j a_{j+1})† ]
# = t1 [ (ωN a†)_j a_{j+1} + a†_{j+1} (a ωNm)_j ]
#
# Mean-field decouple:
#   on site j:     t1[(ωN a†)*<a>_{j+1} + (a ωNm)*<a†>_{j+1}]
#   on site j+1:   t1[ a*<ωN a†>_j + a†*<a ωNm>_j ]
#
# Pair-hop similarly with ω2N and a^2, (a†)^2.
# ----------------------------
function build_h!(h::Vector{Matrix{ComplexF64}}, x::Vector{Matrix{ComplexF64}}, p::PTWAPFParams)
    L = p.L
    J = p.J
    g = p.g

    t1 = -J * (1 - g)
    t2 = -J * g

    # reset + onsite disorder
    @inbounds for j in 1:L
        hj = h[j]
        fill!(hj, 0.0 + 0.0im)
        hj .+= p.mu[j] .* Matrix(N̂)
    end

    # precompute local mean fields needed on each site
    ma   = Vector{ComplexF64}(undef, L)  # <a>
    mad  = Vector{ComplexF64}(undef, L)  # <a†>
    m_wad = Vector{ComplexF64}(undef, L) # <ωN a†>
    m_aωm = Vector{ComplexF64}(undef, L) # <a ωNm>
    ma2   = Vector{ComplexF64}(undef, L) # <a^2>
    ma2d  = Vector{ComplexF64}(undef, L) # <(a†)^2>
    m_w2ad2 = Vector{ComplexF64}(undef, L) # <ω2N (a†)^2>
    m_a2ω2m = Vector{ComplexF64}(undef, L) # <(a^2) ω2Nm>

    # operators with correct ordering
    Op_wad   = Matrix(ωN)  * âd          # ωN a†
    Op_aωm   = â * Matrix(ωNm)           # a ω^{-n}
    Op_w2ad2 = Matrix(ω2N) * â2d         # ω2N (a†)^2
    Op_a2ω2m = â2 * Matrix(ω2Nm)         # (a^2) ω^{-2n}

    @inbounds for j in 1:L
        ma[j]   = trAB(â,  x[j])
        mad[j]  = conj(ma[j])

        m_wad[j] = trAB(Op_wad, x[j])
        m_aωm[j] = trAB(Op_aωm, x[j])

        ma2[j]   = trAB(â2, x[j])
        ma2d[j]  = conj(ma2[j])

        m_w2ad2[j] = trAB(Op_w2ad2, x[j])
        m_a2ω2m[j] = trAB(Op_a2ω2m, x[j])
    end

    # bond contributions
    @inbounds for j in 1:(L-1)
        jp = j + 1

        # --- single hop bond (j,j+1)
        # on j:
        h[j]  .+= (t1 * ma[jp])  .* Op_wad
        h[j]  .+= (t1 * mad[jp]) .* Op_aωm

        # on j+1:
        h[jp] .+= (t1 * m_wad[j]) .* â
        h[jp] .+= (t1 * m_aωm[j]) .* âd

        # --- pair hop bond (j,j+1)
        # on j:
        h[j]  .+= (t2 * ma2[jp])  .* Op_w2ad2
        h[j]  .+= (t2 * ma2d[jp]) .* Op_a2ω2m

        # on j+1:
        h[jp] .+= (t2 * m_w2ad2[j]) .* â2
        h[jp] .+= (t2 * m_a2ω2m[j]) .* â2d
    end

    # final: enforce h Hermitian numerically (small symmetry errors can accumulate)
    @inbounds for j in 1:L
        h[j] .= (h[j] .+ h[j]') ./ 2
    end

    return nothing
end

function rhs!(dx::Vector{Matrix{ComplexF64}}, x::Vector{Matrix{ComplexF64}},
              h::Vector{Matrix{ComplexF64}}, p::PTWAPFParams)
    build_h!(h, x, p)
    @inbounds for j in 1:p.L
        dx[j] .= 1im .* (h[j]*x[j] - x[j]*h[j])
    end
    return nothing
end

# ----------------------------
# One trajectory evolve + measure (RK4), on-the-fly
# ----------------------------
function evolve_traj_and_measure!(I_tr::Vector{Float64}, n_tr::Matrix{Float64},
                                  x0::Vector{Matrix{ComplexF64}},
                                  p::PTWAPFParams, times::Vector{Float64})

    L  = p.L
    Nt = length(times)

    x  = [copy(x0[j]) for j in 1:L]

    # workspaces
    h  = [zeros(ComplexF64, 3,3) for _ in 1:L]
    k1 = [zeros(ComplexF64, 3,3) for _ in 1:L]
    k2 = [zeros(ComplexF64, 3,3) for _ in 1:L]
    k3 = [zeros(ComplexF64, 3,3) for _ in 1:L]
    k4 = [zeros(ComplexF64, 3,3) for _ in 1:L]
    xt = [zeros(ComplexF64, 3,3) for _ in 1:L]

    @inline function measure!(k)
        @inbounds for j in 1:L
            # n_j = Tr(n̂ x_j)
            n_tr[j,k] = real(trAB(Matrix(N̂), x[j]))
        end
        half = L ÷ 2
        NL = sum(@view n_tr[1:half, k])
        NR = sum(@view n_tr[half+1:L, k])
        I_tr[k] = (2.0 / L) * (NL - NR)
    end

    # t0
    @inbounds for j in 1:L
        normalize_site!(x[j])
    end
    measure!(1)

    for k in 2:Nt
        dt = times[k] - times[k-1]

        rhs!(k1, x, h, p)

        @inbounds for j in 1:L
            xt[j] .= x[j] .+ (dt/2) .* k1[j]
            normalize_site!(xt[j])
        end
        rhs!(k2, xt, h, p)

        @inbounds for j in 1:L
            xt[j] .= x[j] .+ (dt/2) .* k2[j]
            normalize_site!(xt[j])
        end
        rhs!(k3, xt, h, p)

        @inbounds for j in 1:L
            xt[j] .= x[j] .+ dt .* k3[j]
            normalize_site!(xt[j])
        end
        rhs!(k4, xt, h, p)

        @inbounds for j in 1:L
            x[j] .+= (dt/6) .* (k1[j] .+ 2k2[j] .+ 2k3[j] .+ k4[j])
            normalize_site!(x[j])
        end

        measure!(k)

        # hard guard: bail early if anything goes NaN
        if !(isfinite(I_tr[k]))
            error("NaN/Inf detected at time index k=$k, t=$(times[k]) — reduce dt or check parameters.")
        end
    end

    return nothing
end

# ----------------------------
# MAIN
# ----------------------------
if length(ARGS) < 7
    error("Usage: julia --threads=N script.jl L W Ndis Nmc tmax seed OUTDIR")
end

L      = parse(Int,     ARGS[1])
W      = parse(Float64, ARGS[2])
Ndis   = parse(Int,     ARGS[3])
Nmc    = parse(Int,     ARGS[4])
tmax   = parse(Float64, ARGS[5])
seed   = parse(Int,     ARGS[6])
OUTDIR = ARGS[7]

J     = 1.0
gpair = 0.3

Nt    = 1001
times = collect(range(0.0, tmax, length=Nt))

println("pTWA PF (fixed phases): L=$L W=$W Ndis=$Ndis Nmc=$Nmc sampling=$sampling Threads=$(nthreads())")

# TLS accumulators over disorder (ED-like SEM over disorder)
T = Threads.maxthreadid()
I_sum_tls  = [zeros(Float64, Nt) for _ in 1:T]
I_sum2_tls = [zeros(Float64, Nt) for _ in 1:T]
n_sum_tls  = [zeros(Float64, L, Nt) for _ in 1:T]

rngs = [MersenneTwister(seed + 10000*tid) for tid in 1:T]
print_lock = ReentrantLock()

@threads for r in 1:Ndis
    tid = threadid()
    rng = rngs[tid]

    # disorder sample
    mu = (2W) .* rand(rng, L) .- W
    p  = PTWAPFParams(L, J, gpair, mu)

    # MC average for fixed disorder
    I_mc_sum  = zeros(Float64, Nt)
    I_mc_sum2 = zeros(Float64, Nt)
    n_mc_sum  = zeros(Float64, L, Nt)

    for tr in 1:Nmc
        x0 = sample_domainwall(sampling, L; cache=cache, rng=rng)

        I_tr = zeros(Float64, Nt)
        n_tr = zeros(Float64, L, Nt)

        evolve_traj_and_measure!(I_tr, n_tr, x0, p, times)

        I_mc_sum  .+= I_tr
        I_mc_sum2 .+= I_tr.^2
        n_mc_sum  .+= n_tr
    end

    I_real = I_mc_sum ./ Nmc
    n_real = n_mc_sum ./ Nmc

    I_sum_tls[tid]  .+= I_real
    I_sum2_tls[tid] .+= I_real.^2
    n_sum_tls[tid]  .+= n_real

    if r % max(1, Ndis ÷ 10) == 0
        lock(print_lock) do
            println("done disorder $r / $Ndis")
        end
    end
end

# reduce TLS
I_sum  = zeros(Float64, Nt)
I_sum2 = zeros(Float64, Nt)
n_sum  = zeros(Float64, L, Nt)
for tid in 1:T
    I_sum  .+= I_sum_tls[tid]
    I_sum2 .+= I_sum2_tls[tid]
    n_sum  .+= n_sum_tls[tid]
end

I_mean = I_sum ./ Ndis
n_mean = n_sum ./ Ndis

I_var = (I_sum2 ./ Ndis) .- I_mean.^2
I_err = sqrt.(max.(I_var, 0.0) ./ Ndis)

println("pTWA PF Imbalance I(t_final) = ", I_mean[end], " ± ", I_err[end])

mkpath(OUTDIR)
outfile = "$OUTDIR/pTWA_Z3_parafermion_L$(L)_g$(gpair)_W$(W)_Ndis$(Ndis)_Nmc$(Nmc)_$(sampling).jld2"
@save outfile L J gpair W Ndis Nmc seed tmax times sampling I_mean I_err n_mean
println("Saved -> ", outfile)