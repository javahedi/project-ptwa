###############################################################################
# pTWA for Z₃ parafermion chain with disorder
# Observable: P1-imbalance I(t) (Néel 0/1 => I(0)=1)
# Parallelization: disorder realizations ONLY (thread-safe)
###############################################################################

using Random
using LinearAlgebra
using Statistics
using JLD2

# =============================== Parameters ==================================

struct PTWAParams
    L::Int
    Jr::Vector{Float64}      # Jr[r] = J / r^α
    μ::Vector{Float64}       # disorder potentials
    θ::Float64               # 2π/3
end

function PTWAParams(L::Int; J=1.0, α=6.0, μ=zeros(L))
    Jr = [Float64(J) / (r^Float64(α)) for r in 1:(L-1)]
    return PTWAParams(L, Jr, Float64.(μ), 2π/3)
end

# ============================ Initial states ==================================

function init_neel01(L::Int)
    s = Vector{Int}(undef, L)
    @inbounds for j in 1:L
        s[j] = isodd(j) ? 0 : 1
    end
    return s
end

function init_domainwall10(L::Int)
    @assert iseven(L)
    s = zeros(Int, L)
    for j in 1:(L÷2)
        s[j] = 1
    end
    return s
end


# ========================== Discrete Wigner sampling ==========================

const ω3 = cis(2π/3)
const inv2_mod3 = 2

const Z3 = Diagonal(ComplexF64[ω3^0, ω3^1, ω3^2])
const X3 = ComplexF64[0 0 1;
                      1 0 0;
                      0 1 0]

function Aqp_WH(q::Int, p::Int)
    A = zeros(ComplexF64, 3, 3)
    for m in 0:2, k in 0:2
        phase = mod(p*k - q*m + inv2_mod3*m*k, 3)
        A .+= ω3^phase * (Z3^m * X3^k)
    end
    A ./= 3
    return (A + A') ./ 2
end

function precompute_A()
    Ac = Array{ComplexF64,4}(undef, 3,3,3,3)
    for q in 0:2, p in 0:2
        Ac[:,:,q+1,p+1] = Aqp_WH(q,p)
    end
    return Ac
end


function local_wigner_probs(ρ::Matrix{ComplexF64}, Ac)
    probs = zeros(Float64, 3, 3)
    for q in 0:2, p in 0:2
        A = Ac[:,:,q+1,p+1]
        probs[q+1,p+1] = real(tr(ρ*A)) / 3
    end
    # Optional numerical safety
    @assert minimum(probs) > -1e-12
    probs ./= sum(probs)
    return probs
end

# function local_wigner_probs(ρ::Matrix{ComplexF64}, Ac)
#     probs = zeros(Float64, 3, 3)
#     for q in 0:2, p in 0:2
#         A = Ac[:,:,q+1,p+1]
#         probs[q+1,p+1] = real(tr(ρ*A)) / 3
#     end
#     probs .= max.(probs, 0.0)
#     probs ./= sum(probs)
#     return probs
# end

function sample_initial_discrete_WH(L::Int, s::Vector{Int}; rng::AbstractRNG, Ac)
    x = [zeros(ComplexF64, 3, 3) for _ in 1:L]
    @inbounds for j in 1:L
        ρ = zeros(ComplexF64, 3, 3)
        ρ[s[j]+1, s[j]+1] = 1.0 + 0im
        probs = local_wigner_probs(ρ, Ac)

        r = rand(rng)
        acc = 0.0
        qsel, psel = 0, 0
        for q in 0:2, p in 0:2
            acc += probs[q+1,p+1]
            if r ≤ acc
                qsel, psel = q, p
                break
            end
        end

        A = Ac[:,:,qsel+1,psel+1]
        for a in 1:3, b in 1:3
            x[j][a,b] = A[b,a]
        end
    end
    return x
end

# ============================ Local symbols ===================================

@inline p1_symbol(xj::Matrix{ComplexF64}) = real(xj[2,2])
@inline n_symbol(xj::Matrix{ComplexF64})  = real(xj[2,2]) + 2.0*real(xj[3,3])

@inline f_symbol(xj::Matrix{ComplexF64})    = xj[1,2] + xj[2,3]
@inline fdag_symbol(xj::Matrix{ComplexF64}) = xj[2,1] + xj[3,2]

# ================================ Strings ====================================

@inline function string_factor(params::PTWAParams, nbar::Vector{Float64}, i::Int, j::Int)
    if i == j || abs(i-j) == 1
        return 1.0 + 0im
    end
    if i < j
        s = 0.0
        @inbounds for k in (i+1):(j-1)
            s += nbar[k]
        end
        return cis(params.θ * s)
    else
        s = 0.0
        @inbounds for k in (j+1):(i-1)
            s += nbar[k]
        end
        return cis(-params.θ * s)
    end
end

# ============================ Gradient (single-particle only) =================

function compute_gradient!(G, x, params::PTWAParams)
    L  = params.L
    Jr = params.Jr
    μ  = params.μ
    θ  = params.θ

    @inbounds for j in 1:L
        fill!(G[j], 0.0 + 0im)
    end

    nbar = Vector{Float64}(undef, L)
    f    = Vector{ComplexF64}(undef, L)
    fdag = Vector{ComplexF64}(undef, L)
    @inbounds for j in 1:L
        nbar[j] = n_symbol(x[j])
        f[j]    = f_symbol(x[j])
        fdag[j] = fdag_symbol(x[j])
    end

    # onsite μ n
    @inbounds for j in 1:L
        G[j][2,2] += μ[j]
        G[j][3,3] += 2.0 * μ[j]
    end

    @inline function add_df!(Gj, coef::ComplexF64)
        Gj[1,2] += coef
        Gj[2,3] += coef
    end
    @inline function add_dfdag!(Gj, coef::ComplexF64)
        Gj[2,1] += coef
        Gj[3,2] += coef
    end
    @inline function add_dn!(Gj, coef::ComplexF64)
        Gj[2,2] += coef
        Gj[3,3] += 2.0 * coef
    end

    for r in 1:(L-1)
        J = Jr[r]
        J == 0.0 && continue
        for i in 1:(L-r)
            j = i + r
            Sij = string_factor(params, nbar, i, j)
            Sji = conj(Sij)

            add_dfdag!(G[i], J * Sij * f[j])
            add_df!(G[i],    J * Sji * fdag[j])
            add_dfdag!(G[j], J * Sji * f[i])
            add_df!(G[j],    J * Sij * fdag[i])

            if j > i+1
                amp_forward  = J * (fdag[i] * f[j])
                amp_backward = J * (fdag[j] * f[i])
                @inbounds for k in (i+1):(j-1)
                    add_dn!(G[k], (1im*θ)*Sij*amp_forward + (-1im*θ)*Sji*amp_backward)
                end
            end
        end
    end
    return nothing
end

# ============================== EOM + RK4 ====================================

function rhs!(dx, x, params, Gwork)
    compute_gradient!(Gwork, x, params)
    @inbounds for j in 1:params.L
        hj = transpose(Gwork[j])
        dx[j] .= 1im .* (x[j]*hj - hj*x[j])
    end
end

function step_rk4!(x, params, dt; k1, k2, k3, k4, tmp, Gwork)
    rhs!(k1, x, params, Gwork)
    @inbounds for j in eachindex(x); tmp[j] .= x[j] .+ (0.5*dt).*k1[j]; end

    rhs!(k2, tmp, params, Gwork)
    @inbounds for j in eachindex(x); tmp[j] .= x[j] .+ (0.5*dt).*k2[j]; end

    rhs!(k3, tmp, params, Gwork)
    @inbounds for j in eachindex(x); tmp[j] .= x[j] .+ dt.*k3[j]; end

    rhs!(k4, tmp, params, Gwork)

    @inbounds for j in eachindex(x)
        x[j] .+= (dt/6) .* (k1[j] .+ 2k2[j] .+ 2k3[j] .+ k4[j])
        x[j] .= (x[j] .+ x[j]') ./ 2
    end
end

# ============================ Imbalance =======================================

@inline function imbalance_P1(x, N1)
    L = length(x)
    s = 0.0
    @inbounds for j in 1:L
        stag = isodd(j) ? +1.0 : -1.0   # Néel 0/1: even sites have P1=1 -> gives I(0)=1 after /N1 with this sign
        s += (-stag) * p1_symbol(x[j])  # flip sign so that occupied-even gives +1
    end
    return s / N1
end



@inline function imbalance_domainwall(x)
    L = length(x)
    half = L ÷ 2
    NL = 0.0
    NR = 0.0
    @inbounds for j in 1:half
        NL += n_symbol(x[j])
    end
    @inbounds for j in (half+1):L
        NR += n_symbol(x[j])
    end
    return 2.0 * (NL - NR) / L
end

# ============================ pTWA runner (SERIAL trajectories) ===============

function run_ptwa_imbalance(params::PTWAParams; s,
                            ntraj=500, tmax=50.0, dt=0.1, seed=1, verbose=true)

    rng = MersenneTwister(seed)
    Ac = precompute_A()

    t  = collect(0:dt:tmax)
    nt = length(t)
    I  = zeros(Float64, nt)

    Gwork = [zeros(ComplexF64,3,3) for _ in 1:params.L]
    k1    = [zeros(ComplexF64,3,3) for _ in 1:params.L]
    k2    = [zeros(ComplexF64,3,3) for _ in 1:params.L]
    k3    = [zeros(ComplexF64,3,3) for _ in 1:params.L]
    k4    = [zeros(ComplexF64,3,3) for _ in 1:params.L]
    tmp   = [zeros(ComplexF64,3,3) for _ in 1:params.L]

    for tr in 1:ntraj
        x = sample_initial_discrete_WH(params.L, s; rng=rng, Ac=Ac)

        N1 = sum(p1_symbol(x[j]) for j in 1:params.L)
        if tr == 1 && verbose
            println("Sanity: N1(0) ≈ $N1  expected ~ L/2 = $(params.L/2)")
            println("Sanity: I(0) single traj ≈ ", imbalance_domainwall(x))
        end

        @inbounds for ti in 1:nt
            #I[ti] += imbalance_P1(x, N1)
            I[ti] += imbalance_domainwall(x)
            ti < nt && step_rk4!(x, params, dt; k1=k1,k2=k2,k3=k3,k4=k4,tmp=tmp,Gwork=Gwork)
        end
    end

    return t, I ./ ntraj
end

# ============================ Disorder average (THREADED) ======================

function run_disorder_imbalance(; L, J, α, W,
                               ntraj=500, nreal=50,
                               tmax=50.0, dt=0.1, seed=1)

    s = init_domainwall10(L)
    results = Vector{Vector{Float64}}(undef, nreal)
    t_ref = nothing

    Threads.@threads for r in 1:nreal
        rng = MersenneTwister(seed + r)
        μ = W .* (2 .* rand(rng, L) .- 1)

        params = PTWAParams(L; J=J, α=α, μ=μ)
        t, I = run_ptwa_imbalance(params; s=s, ntraj=ntraj, tmax=tmax, dt=dt,
                                  seed=seed + 10_000*r, verbose=false)

        results[r] = I
        if r == 1
            t_ref = t
        end
        println("disorder realization $r / $nreal on thread $(Threads.threadid())")
    end

    Iavg = zeros(length(t_ref))
    for r in 1:nreal
        Iavg .+= results[r]
    end
    Iavg ./= nreal

    return t_ref, Iavg
end

# ============================ Main ============================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("Using $(Threads.nthreads()) threads")

    L = 12
    α = 6.0
    J = 1.0
    W_list = [1.5, 2.5, 3.5, 4.5, 5.5, 6.0]#[0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    ntraj = 1000
    nreal = 100
    tmax  = 200.0
    dt    = 0.1    
    seed  = 1

    for W in W_list
        println("\n=== Running: L=$L, α=$α, W=$W ===")
        t, I = run_disorder_imbalance(L=L, J=J, α=α, W=W,
                                      ntraj=ntraj, nreal=nreal,
                                      tmax=tmax, dt=dt, seed=seed)

        filename = "imbalance_pTWA_Z3_L$(L)_alpha$(α)_W$(W)_discrete_safe.jld2"
        @save filename t I L α W J ntraj nreal tmax dt
        println("Saved → $filename")
    end
end
