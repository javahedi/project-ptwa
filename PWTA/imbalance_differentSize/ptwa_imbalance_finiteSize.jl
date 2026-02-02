###############################################################################
# pTWA for Z₃ parafermion chain with disorder
# Observable: domain-wall imbalance I(t)
# Capability benchmark: multiple system sizes L
###############################################################################

using Random
using LinearAlgebra
using Statistics
using JLD2
using Base.Threads

# =============================== Parameters ==================================

struct PTWAParams
    L::Int
    Jr::Vector{Float64}
    μ::Vector{Float64}
    θ::Float64
end

function PTWAParams(L::Int; J=1.0, α=6.0, μ=zeros(L))
    Jr = [Float64(J) / (r^Float64(α)) for r in 1:(L-1)]
    return PTWAParams(L, Jr, Float64.(μ), 2π/3)
end

# ============================ Initial states ==================================

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
                x[j] .= Ac[:,:,q+1,p+1]'
                break
            end
        end
    end
    return x
end

# ============================ Local symbols ===================================

@inline n_symbol(xj) = real(xj[2,2]) + 2.0*real(xj[3,3])
@inline f_symbol(xj) = xj[1,2] + xj[2,3]
@inline fdag_symbol(xj) = xj[2,1] + xj[3,2]

# ================================ Strings ====================================

@inline function string_factor(params, nbar, i, j)
    if abs(i-j) ≤ 1
        return 1.0 + 0im
    end
    s = sum(nbar[min(i,j)+1:max(i,j)-1])
    return cis(params.θ * (i < j ? s : -s))
end

# ============================ Gradient ========================================

function compute_gradient!(G, x, params)
    L, Jr, μ, θ = params.L, params.Jr, params.μ, params.θ
    fill!.(G, 0.0 + 0im)

    nbar = [n_symbol(x[j]) for j in 1:L]
    f    = [f_symbol(x[j]) for j in 1:L]
    fdag = [fdag_symbol(x[j]) for j in 1:L]

    for j in 1:L
        G[j][2,2] += μ[j]
        G[j][3,3] += 2μ[j]
    end

    for r in 1:(L-1), i in 1:(L-r)
        j = i+r
        J = Jr[r]
        J == 0 && continue

        Sij = string_factor(params, nbar, i, j)
        Sji = conj(Sij)

        G[i][2,1] += J*Sij*f[j]
        G[i][3,2] += J*Sij*f[j]
        G[i][1,2] += J*Sji*fdag[j]
        G[i][2,3] += J*Sji*fdag[j]

        G[j][2,1] += J*Sji*f[i]
        G[j][3,2] += J*Sji*f[i]
        G[j][1,2] += J*Sij*fdag[i]
        G[j][2,3] += J*Sij*fdag[i]

        if j > i+1
            amp = J*(fdag[i]*f[j])
            for k in i+1:j-1
                G[k][2,2] +=  1im*θ*Sij*amp
                G[k][3,3] +=  2im*θ*Sij*amp
                G[k][2,2] += -1im*θ*Sji*conj(amp)
                G[k][3,3] += -2im*θ*Sji*conj(amp)
            end
        end
    end
end

# ============================== EOM ===========================================

function rhs!(dx, x, params, G)
    compute_gradient!(G, x, params)
    for j in 1:params.L
        h = transpose(G[j])
        dx[j] .= 1im*(x[j]*h - h*x[j])
    end
end

function step_rk4!(x, params, dt; k1,k2,k3,k4,tmp,G)
    rhs!(k1,x,params,G)
    for j in eachindex(x); tmp[j] .= x[j] .+ 0.5dt*k1[j]; end
    rhs!(k2,tmp,params,G)
    for j in eachindex(x); tmp[j] .= x[j] .+ 0.5dt*k2[j]; end
    rhs!(k3,tmp,params,G)
    for j in eachindex(x); tmp[j] .= x[j] .+ dt*k3[j]; end
    rhs!(k4,tmp,params,G)
    for j in eachindex(x)
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

# ============================ Runner ==========================================

function run_disorder_imbalance(; L, J, α, W, ntraj, nreal, tmax, dt, seed)
    s = init_domainwall10(L)
    Ac = precompute_A()
    t = collect(0:dt:tmax)
    Iavg = zeros(length(t))

    Threads.@threads for r in 1:nreal
        rng = MersenneTwister(seed+r)
        μ = W*(2rand(rng,L).-1)
        params = PTWAParams(L; J=J, α=α, μ=μ)

        G = [zeros(ComplexF64,3,3) for _ in 1:L]
        k1=k2=k3=k4=tmp=deepcopy(G)

        I = zeros(length(t))
        for tr in 1:ntraj
            x = sample_initial_discrete_WH(L,s; rng=rng,Ac=Ac)
            for ti in eachindex(t)
                I[ti] += imbalance_domainwall(x)
                ti<length(t) && step_rk4!(x,params,dt; k1=k1,k2=k2,k3=k3,k4=k4,tmp=tmp,G=G)
            end
        end
        Iavg .+= I/ntraj
    end
    return t, Iavg/nreal
end

# ============================ Main ============================================

if abspath(PROGRAM_FILE)==@__FILE__
    L_list = [12, 24, 48]
    W_list = [2.5, 4.5, 6.0]
    α=6.0; J=1.0
    ntraj=800; nreal=30
    tmax=200.0; dt=0.1; seed=1

    for L in L_list, W in W_list
        println("Running L=$L  W=$W")
        t,I = run_disorder_imbalance(
            L=L,J=J,α=α,W=W,
            ntraj=ntraj,nreal=nreal,
            tmax=tmax,dt=dt,seed=seed
        )
        @save "imbalance_Z3_L$(L)_W$(W).jld2" t I L W α J
    end
end
