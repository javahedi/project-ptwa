###############################################################################
# pTWA for disordered NN Z₃ Fock parafermion chain (matches ED: J, g, μ)
# H = -J Σ_j [ (1-g) f†_j f_{j+1} + g (f†_j)^2 f_{j+1}^2 + h.c. ] + Σ μ_j n_j
#
# Observable: domain-wall imbalance I(t) = (2/L)*(N_L - N_R)
###############################################################################

using Random
using LinearAlgebra
using Statistics
using JLD2
using Base.Threads

# =============================== Parameters ==================================

struct PTWAParams
    L::Int
    J::Float64
    g::Float64
    μ::Vector{Float64}
    θ::Float64
end

function PTWAParams(L::Int; J=1.0, g=0.5, μ=zeros(L))
    return PTWAParams(L, Float64(J), Float64(g), Float64.(μ), 2π/3)
end

# ============================ Initial state ==================================

function init_domainwall10(L::Int)
    @assert iseven(L)
    s = zeros(Int, L)
    for j in 1:(L ÷ 2)
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
        probs[q+1,p+1] = real(tr(ρ * Ac[:,:,q+1,p+1])) / 3
    end
    @assert minimum(probs) > -1e-12
    probs ./= sum(probs)
    return probs
end

function sample_initial_discrete_WH(L::Int, s::Vector{Int}; rng, Ac)
    x = [zeros(ComplexF64, 3, 3) for _ in 1:L]
    for j in 1:L
        ρ = zeros(ComplexF64, 3, 3)
        ρ[s[j]+1, s[j]+1] = 1.0
        probs = local_wigner_probs(ρ, Ac)

        r = rand(rng)
        acc = 0.0
        for q in 0:2, p in 0:2
            acc += probs[q+1,p+1]
            if r ≤ acc
                x[j] .= Ac[:,:,q+1,p+1]'   # x^{ab} = A_{ba}
                break
            end
        end
    end
    return x
end

# ============================ Local symbols ===================================

@inline n_symbol(xj) = real(xj[2,2]) + 2.0*real(xj[3,3])

# For your current convention (same as your long-range code):
# f  = X01 + X12  -> symbol uses x[1,2] + x[2,3]
# f† = X10 + X21  -> symbol uses x[2,1] + x[3,2]
@inline f_symbol(xj)    = xj[1,2] + xj[2,3]
@inline fdag_symbol(xj) = xj[2,1] + xj[3,2]

# Pair operators: f^2 = X02, (f†)^2 = X20 in this convention
@inline f2_symbol(xj)    = xj[1,3]
@inline fdag2_symbol(xj) = xj[3,1]

# ============================ Gradient (NN only) ==============================

function compute_gradient!(G, x, params::PTWAParams)
    L, J, g, μ, θ = params.L, params.J, params.g, params.μ, params.θ
    fill!.(G, 0.0 + 0im)

    nbar  = [n_symbol(x[j])   for j in 1:L]
    f     = [f_symbol(x[j])   for j in 1:L]
    fdag  = [fdag_symbol(x[j]) for j in 1:L]
    f2    = [f2_symbol(x[j])  for j in 1:L]
    fdag2 = [fdag2_symbol(x[j]) for j in 1:L]

    # onsite μ n
    @inbounds for j in 1:L
        G[j][2,2] += μ[j]
        G[j][3,3] += 2.0 * μ[j]
    end

    # helpers to add derivatives wrt f, f†, f^2, (f†)^2, and n (if needed)
    @inline function add_df!(Gj, coef::ComplexF64)
        Gj[1,2] += coef
        Gj[2,3] += coef
    end
    @inline function add_dfdag!(Gj, coef::ComplexF64)
        Gj[2,1] += coef
        Gj[3,2] += coef
    end
    @inline function add_df2!(Gj, coef::ComplexF64)
        Gj[1,3] += coef
    end
    @inline function add_dfdag2!(Gj, coef::ComplexF64)
        Gj[3,1] += coef
    end

    # ---- nearest-neighbor only: strings are trivial, but keep structure ----
    # Match ED Hamiltonian sign: H_hop = -J[(1-g) f†f + g (f†)^2 f^2 + h.c.]
    J1 = -J * (1 - g)
    J2 = -J * g

    @inbounds for j in 1:(L-1)
        jp = j + 1

        # single hop: J1 ( f†_j f_{j+1} + f†_{j+1} f_j )
        add_dfdag!(G[j],  J1 * f[jp])
        add_df!(G[j],     J1 * fdag[jp])
        add_dfdag!(G[jp], J1 * f[j])
        add_df!(G[jp],    J1 * fdag[j])

        # pair hop: J2 ( (f†_j)^2 f_{j+1}^2 + (f†_{j+1})^2 f_j^2 )
        add_dfdag2!(G[j],  J2 * f2[jp])
        add_df2!(G[j],     J2 * fdag2[jp])
        add_dfdag2!(G[jp], J2 * f2[j])
        add_df2!(G[jp],    J2 * fdag2[j])
    end

    return nothing
end

# ============================== EOM ===========================================

function rhs!(dx, x, params, G)
    compute_gradient!(G, x, params)
    @inbounds for j in 1:params.L
        h = transpose(G[j])
        dx[j] .= 1im*(x[j]*h - h*x[j])
    end
end

function step_rk4!(x, params, dt; k1,k2,k3,k4,tmp,G)
    rhs!(k1,x,params,G)
    @inbounds for j in eachindex(x); tmp[j] .= x[j] .+ 0.5dt*k1[j]; end
    rhs!(k2,tmp,params,G)
    @inbounds for j in eachindex(x); tmp[j] .= x[j] .+ 0.5dt*k2[j]; end
    rhs!(k3,tmp,params,G)
    @inbounds for j in eachindex(x); tmp[j] .= x[j] .+ dt*k3[j]; end
    rhs!(k4,tmp,params,G)
    @inbounds for j in eachindex(x)
        x[j] .+= dt/6*(k1[j]+2k2[j]+2k3[j]+k4[j])
        x[j] .= (x[j]+x[j]')/2
    end
end

# ============================ Imbalance =======================================

function imbalance_domainwall(x)
    L = length(x)
    NL = sum(n_symbol(x[j]) for j in 1:L÷2)
    NR = sum(n_symbol(x[j]) for j in L÷2+1:L)
    return 2*(NL-NR)/L
end

# ============================ Disorder runner =================================

function run_disorder_imbalance(; L, J, g, W, ntraj, nreal, tmax, dt, seed)
    s = init_domainwall10(L)
    Ac = precompute_A()
    t = collect(0:dt:tmax)
    Iavg = zeros(length(t))

    Threads.@threads for r in 1:nreal
        rng = MersenneTwister(seed+r)
        μ = W*(2rand(rng,L).-1)

        params = PTWAParams(L; J=J, g=g, μ=μ)

        G = [zeros(ComplexF64,3,3) for _ in 1:L]
        k1 = [zeros(ComplexF64,3,3) for _ in 1:L]
        k2 = [zeros(ComplexF64,3,3) for _ in 1:L]
        k3 = [zeros(ComplexF64,3,3) for _ in 1:L]
        k4 = [zeros(ComplexF64,3,3) for _ in 1:L]
        tmp = [zeros(ComplexF64,3,3) for _ in 1:L]

        I = zeros(length(t))
        for tr in 1:ntraj
            x = sample_initial_discrete_WH(L, s; rng=rng, Ac=Ac)
            for ti in eachindex(t)
                I[ti] += imbalance_domainwall(x)
                ti < length(t) && step_rk4!(x, params, dt; k1=k1,k2=k2,k3=k3,k4=k4,tmp=tmp,G=G)
            end
        end
        Iavg .+= I/ntraj
    end

    return t, Iavg/nreal
end

# ============================ Main ============================================

if abspath(PROGRAM_FILE)==@__FILE__
    println("Using $(Threads.nthreads()) threads")

    # ED-comparison sizes: small only!
    L_list = [12, 24, 48]           # keep ED feasible
    W_list = [2.5, 4.5, 6.0]

    J = 1.0
    g = 0.5                     # <-- match ED g here

    ntraj = 1000
    nreal = 100
    tmax  = 200.0
    dt    = 0.1
    seed  = 1

    for L in L_list, W in W_list
        println("Running pTWA NN: L=$L  W=$W  g=$g")
        t, I = run_disorder_imbalance(L=L, J=J, g=g, W=W,
                                      ntraj=ntraj, nreal=nreal,
                                      tmax=tmax, dt=dt, seed=seed)

        @save "pTWA_NN_Z3_domainwall_L$(L)_W$(W)_g$(g).jld2" t I L W g J ntraj nreal tmax dt
    end
end
