###############################################################################
# pTWA for Z₃ parafermion chain with disorder
# Observable: imbalance I(t) for MBL diagnostics (Néel 0/1 => I(0)=1)
###############################################################################

using Random
using LinearAlgebra
using Statistics
using JLD2
using Plots

# =============================== Parameters ==================================

struct PTWAParams
    L::Int
    Jr::Vector{Float64}
    Gr::Vector{Float64}
    μ::Vector{Float64}
    ω::ComplexF64
    θ::Float64
end

function PTWAParams(L::Int; J=1.0, G=0.0, α=Inf, μ=zeros(L))
    Jr = [Float64(J) / (r^Float64(α)) for r in 1:(L-1)]
    Gr = [Float64(G) / (r^Float64(α)) for r in 1:(L-1)]
    return PTWAParams(L, Jr, Gr, Float64.(μ), cis(2π/3), 2π/3)
end

# ============================ Initial states ==================================

function init_neel01(L::Int)
    s = zeros(Int, L)
    for j in 1:L
        s[j] = isodd(j) ? 0 : 1
    end
    return s
end

# ========================== Discrete Wigner sampling ==========================

const ω3 = cis(2π/3)
const inv2_mod3 = 2

const Z3 = Diagonal(ComplexF64[ω3^0, ω3^1, ω3^2])
const X3 = ComplexF64[
    0 0 1;
    1 0 0;
    0 1 0
]

function Aqp_WH(q::Int, p::Int)
    A = zeros(ComplexF64,3,3)
    for m in 0:2, k in 0:2
        phase = mod(p*k - q*m + inv2_mod3*m*k, 3)
        A .+= ω3^phase * (Z3^m * X3^k)
    end
    A ./= 3
    return (A + A') ./ 2
end

function precompute_A()
    Ac = Array{ComplexF64,4}(undef,3,3,3,3)
    for q in 0:2, p in 0:2
        Ac[:,:,q+1,p+1] = Aqp_WH(q,p)
    end
    return Ac
end

function local_wigner_probs(ρ::Matrix{ComplexF64}, Ac)
    probs = zeros(Float64,3,3)
    for q in 0:2, p in 0:2
        probs[q+1,p+1] = real(tr(ρ * Ac[:,:,q+1,p+1])) / 3
    end
    probs .= max.(probs, 0.0)
    return probs ./ sum(probs)
end

function sample_initial_discrete_WH(L::Int, s::Vector{Int}; rng=Random.default_rng(), Ac)
    x = [zeros(ComplexF64,3,3) for _ in 1:L]
    for j in 1:L
        ρ = zeros(ComplexF64,3,3)
        ρ[s[j]+1,s[j]+1] = 1
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

p1_symbol(x) = real(x[2,2])
n_symbol(x)  = real(x[2,2]) + 2real(x[3,3])

f_symbol(x)    = x[1,2] + x[2,3]
fdag_symbol(x) = x[2,1] + x[3,2]

# ================================ Strings ====================================

ωpow(params,x) = cis(params.θ * x)

function string_factor(params::PTWAParams, nbar::Vector{Float64}, i::Int, j::Int)
    if abs(i-j) ≤ 1
        return 1.0 + 0im
    end

    s = 0.0
    for k in min(i,j)+1 : max(i,j)-1
        s += nbar[k]
    end

    # 🔑 project to Z3 sector to prevent runaway phases
    s_mod = mod(s, 3.0)

    return cis(params.θ * s_mod * sign(j-i))
end

# ============================ Gradient ========================================

function compute_gradient!(G,x,params)
    L=params.L
    nbar=[n_symbol(x[j]) for j in 1:L]
    for j in 1:L; fill!(G[j],0); end

    for j in 1:L
        G[j][2,2] += params.μ[j]
        G[j][3,3] += 2params.μ[j]
    end

    for r in 1:(L-1), i in 1:(L-r)
        j=i+r
        S=string_factor(params,nbar,i,j)
        G[i][2,1]+=params.Jr[r]*S*f_symbol(x[j])
        G[j][2,1]+=params.Jr[r]*conj(S)*f_symbol(x[i])
    end
end

# ============================== EOM ===========================================

function rhs!(dx,x,params,G)
    compute_gradient!(G,x,params)
    for j in 1:params.L
        hj = transpose(G[j])
        dx[j] .= 1im*(x[j]*hj - hj*x[j])
    end
end

# ============================== RK4 ===========================================

function step_rk4!(x,params,dt;k1,k2,k3,k4,tmp,G)
    rhs!(k1,x,params,G)
    for j in eachindex(x); tmp[j]=x[j]+0.5dt*k1[j]; end
    rhs!(k2,tmp,params,G)
    for j in eachindex(x); tmp[j]=x[j]+0.5dt*k2[j]; end
    rhs!(k3,tmp,params,G)
    for j in eachindex(x); tmp[j]=x[j]+dt*k3[j]; end
    rhs!(k4,tmp,params,G)
    for j in eachindex(x)
        x[j]+=dt*(k1[j]+2k2[j]+2k3[j]+k4[j])/6
        x[j]=(x[j]+x[j]')/2
    end
end

# ============================ Imbalance =======================================

function imbalance_P1(x, N1)
    s = 0.0
    for j in eachindex(x)
        stag = isodd(j) ? -1.0 : +1.0   # <-- FLIPPED
        s += stag * p1_symbol(x[j])
    end
    return s / N1
end


# ============================ Main runner =====================================

function run_ptwa_imbalance(params;
                            s,
                            sampler=:discrete,
                            ntraj=2000,
                            tmax=5.0,
                            dt=0.05,
                            seed=1)

    rng = MersenneTwister(seed)
    t = collect(0:dt:tmax)
    nt = length(t)
    I = zeros(nt)

    Ac = precompute_A()

    G=[zeros(ComplexF64,3,3) for _ in 1:params.L]
    k1=deepcopy(G); k2=deepcopy(G); k3=deepcopy(G); k4=deepcopy(G)
    tmp=deepcopy(G)

    for tr in 1:ntraj
        x = sample_initial_discrete_WH(params.L,s; rng=rng, Ac=Ac)
        N1 = sum(p1_symbol(x[j]) for j in 1:params.L)

        tr==1 && println("Sanity I(0) = ", imbalance_P1(x,N1))

        for ti in 1:nt
            I[ti]+=imbalance_P1(x,N1)
            ti<nt && step_rk4!(x,params,dt;k1=k1,k2=k2,k3=k3,k4=k4,tmp=tmp,G=G)
        end
    end

    return t, I ./ ntraj
end

# ============================ Disorder wrapper ================================

function run_disorder_imbalance(;L,J,G,α,W,ntraj=2000,nreal=100,
                               tmax=5.0,dt=0.05,seed=1)

    s=init_neel01(L)
    t_ref=nothing
    Iavg=nothing

    for r in 1:nreal
        μ=W*(2rand(L).-1)
        params=PTWAParams(L;J=J,G=G,α=α,μ=μ)
        t,I=run_ptwa_imbalance(params; s=s, ntraj=ntraj, tmax=tmax, dt=dt, seed=seed+r)
        r==1 && (t_ref=t; Iavg=zeros(length(t)))
        Iavg .+= I
        println("disorder realization $r / $nreal")
    end
    return t_ref, Iavg ./ nreal
end

# ============================ Sweep ===========================================

if abspath(PROGRAM_FILE) == @__FILE__
    L=12; α=6.0; J=1.0; G=0.2
    W_list=[0.5,1,2,3,4,5,6]
    ntraj=2000; nreal=100; tmax=300.0; dt=0.1

    for W in W_list
        t,I = run_disorder_imbalance(L=L,J=J,G=G,α=α,W=W,
                                     ntraj=ntraj,nreal=nreal,
                                     tmax=tmax,dt=dt)
        @save "imbalance_pTWA_Z3_L$(L)_alpha$(α)_W$(W)_discrete.jld2" t I
    end
end
