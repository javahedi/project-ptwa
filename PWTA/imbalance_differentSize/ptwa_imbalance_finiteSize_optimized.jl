###############################################################################
# OPTIMIZED pTWA for disordered NN Z₃ Fock parafermion chain
# H = -J Σ_j [ (1-g) f†_j f_{j+1} + g (f†_j)^2 f_{j+1}^2 + h.c. ] + Σ μ_j n_j
#
# Observable: domain-wall imbalance I(t) = (2/L)*(N_L - N_R)
###############################################################################

using Random
using LinearAlgebra
using Statistics
using JLD2
using Base.Threads
using StaticArrays  # CRITICAL for performance

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
    s = zeros(Int8, L)
    half = L ÷ 2
    @inbounds for j in 1:half
        s[j] = 1
    end
    return s
end

# ========================== Discrete Wigner sampling ==========================

const ω3 = cis(2π/3)
const inv2_mod3 = 2

# Use StaticArrays for 3x3 matrices - MUCH faster
const Z3 = @SMatrix [ω3^0 0.0 0.0; 0.0 ω3^1 0.0; 0.0 0.0 ω3^2]
const X3 = @SMatrix [0.0 0.0 1.0+0im;
                     1.0+0im 0.0 0.0;
                     0.0 1.0+0im 0.0]

# Precompute ALL A matrices once globally
function precompute_A_static()
    A_static = Array{SMatrix{3,3,ComplexF64,9},2}(undef, 3, 3)
    
    # Precompute powers of Z3 and X3
    Z3_pow = [Z3^0, Z3^1, Z3^2]
    X3_pow = [X3^0, X3^1, X3^2]
    
    @inbounds for q in 0:2, p in 0:2
        A = zeros(ComplexF64, 3, 3)
        for m in 0:2, k in 0:2
            phase = mod(p*k - q*m + inv2_mod3*m*k, 3)
            A .+= ω3^phase .* (Z3_pow[m+1] * X3_pow[k+1])
        end
        A ./= 3
        A_sym = (A + A') / 2
        A_static[q+1, p+1] = SMatrix{3,3}(A_sym)
    end
    return A_static
end

# Global cache - compute once
const A_GLOBAL = precompute_A_static()

# Precompute cumulative probabilities for faster sampling
function precompute_cumulative_probs()
    cum_probs = zeros(Float64, 3, 3, 3)  # [state, q, p]
    
    for state in 0:2
        ρ = zeros(3, 3)
        ρ[state+1, state+1] = 1.0
        
        total = 0.0
        for q in 0:2, p in 0:2
            prob = real(tr(ρ * A_GLOBAL[q+1, p+1])) / 3
            cum_probs[state+1, q+1, p+1] = total + prob
            total += prob
        end
        # Normalize
        for q in 0:2, p in 0:2
            cum_probs[state+1, q+1, p+1] /= total
        end
    end
    return cum_probs
end

const CUM_PROBS = precompute_cumulative_probs()

function sample_initial_discrete_WH_fast(L::Int, s::Vector{Int8}; rng)
    x = Vector{SMatrix{3,3,ComplexF64,9}}(undef, L)
    
    @inbounds for j in 1:L
        state = s[j]
        r = rand(rng)
        
        # Fast linear search through 9 possibilities (3x3)
        found = false
        for qp in 1:9
            q = (qp-1) ÷ 3
            p = (qp-1) % 3
            if r ≤ CUM_PROBS[state+1, q+1, p+1]
                x[j] = A_GLOBAL[q+1, p+1]'
                found = true
                break
            end
        end
        
        # Fallback in case of floating point issues
        if !found
            x[j] = A_GLOBAL[1, 1]'
        end
    end
    return x
end

# ============================ Local symbols ===================================

# Use @inline and type assertions for better performance
@inline function n_symbol(xj::SMatrix{3,3,ComplexF64,9})
    return real(xj[2,2]) + 2.0*real(xj[3,3])
end

@inline function f_symbol(xj::SMatrix{3,3,ComplexF64,9})
    return xj[1,2] + xj[2,3]
end

@inline function fdag_symbol(xj::SMatrix{3,3,ComplexF64,9})
    return xj[2,1] + xj[3,2]
end

@inline function f2_symbol(xj::SMatrix{3,3,ComplexF64,9})
    return xj[1,3]
end

@inline function fdag2_symbol(xj::SMatrix{3,3,ComplexF64,9})
    return xj[3,1]
end

# ============================ Gradient (NN only) ==============================

function compute_gradient_fast!(G::Vector{SMatrix{3,3,ComplexF64,9}}, 
                                x::Vector{SMatrix{3,3,ComplexF64,9}}, 
                                params::PTWAParams)
    L, J, g, μ = params.L, params.J, params.g, params.μ
    J1 = -J * (1 - g)
    J2 = -J * g
    
    @inbounds for j in 1:L
        # Reset gradient
        G[j] = @SMatrix zeros(ComplexF64, 3, 3)
        
        # Onsite term
        G[j] += @SMatrix [0.0 0.0 0.0;
                          0.0 μ[j] 0.0;
                          0.0 0.0 2.0*μ[j]]
    end
    
    @inbounds for j in 1:(L-1)
        jp = j + 1
        
        # Get symbols once
        f_j = f_symbol(x[j])
        fdag_j = fdag_symbol(x[j])
        f2_j = f2_symbol(x[j])
        fdag2_j = fdag2_symbol(x[j])
        
        f_jp = f_symbol(x[jp])
        fdag_jp = fdag_symbol(x[jp])
        f2_jp = f2_symbol(x[jp])
        fdag2_jp = fdag2_symbol(x[jp])
        
        # Single hop terms
        # ∂/∂f†_j: J1 * f_{j+1}
        G[j] += @SMatrix [0.0 0.0 0.0;
                          J1*f_jp 0.0 0.0;
                          0.0 J1*f_jp 0.0]
        
        # ∂/∂f_j: J1 * f†_{j+1}
        G[j] += @SMatrix [0.0 J1*fdag_jp 0.0;
                          0.0 0.0 J1*fdag_jp;
                          0.0 0.0 0.0]
        
        # Same for j+1
        G[jp] += @SMatrix [0.0 0.0 0.0;
                           J1*f_j 0.0 0.0;
                           0.0 J1*f_j 0.0]
        
        G[jp] += @SMatrix [0.0 J1*fdag_j 0.0;
                           0.0 0.0 J1*fdag_j;
                           0.0 0.0 0.0]
        
        # Pair hop terms
        # ∂/∂(f†_j)^2: J2 * f_{j+1}^2
        G[j] += @SMatrix [0.0 0.0 0.0;
                          0.0 0.0 0.0;
                          J2*f2_jp 0.0 0.0]
        
        # ∂/∂(f_j)^2: J2 * (f†_{j+1})^2
        G[j] += @SMatrix [0.0 0.0 J2*fdag2_jp;
                          0.0 0.0 0.0;
                          0.0 0.0 0.0]
        
        # Same for j+1
        G[jp] += @SMatrix [0.0 0.0 0.0;
                           0.0 0.0 0.0;
                           J2*f2_j 0.0 0.0]
        
        G[jp] += @SMatrix [0.0 0.0 J2*fdag2_j;
                           0.0 0.0 0.0;
                           0.0 0.0 0.0]
    end
end

# ============================== EOM ===========================================

function rhs_fast!(dx::Vector{SMatrix{3,3,ComplexF64,9}}, 
                   x::Vector{SMatrix{3,3,ComplexF64,9}}, 
                   params::PTWAParams,
                   G::Vector{SMatrix{3,3,ComplexF64,9}})
    compute_gradient_fast!(G, x, params)
    @inbounds for j in 1:params.L
        h = transpose(G[j])
        dx[j] = 1im * (x[j] * h - h * x[j])
    end
end

function step_rk4_fast!(x::Vector{SMatrix{3,3,ComplexF64,9}}, 
                        params::PTWAParams, 
                        dt::Float64;
                        k1::Vector{SMatrix{3,3,ComplexF64,9}},
                        k2::Vector{SMatrix{3,3,ComplexF64,9}},
                        k3::Vector{SMatrix{3,3,ComplexF64,9}},
                        k4::Vector{SMatrix{3,3,ComplexF64,9}},
                        tmp::Vector{SMatrix{3,3,ComplexF64,9}},
                        G::Vector{SMatrix{3,3,ComplexF64,9}})
    L = params.L
    
    # k1
    rhs_fast!(k1, x, params, G)
    
    # k2
    @inbounds for j in 1:L
        tmp[j] = x[j] + 0.5dt * k1[j]
    end
    rhs_fast!(k2, tmp, params, G)
    
    # k3
    @inbounds for j in 1:L
        tmp[j] = x[j] + 0.5dt * k2[j]
    end
    rhs_fast!(k3, tmp, params, G)
    
    # k4
    @inbounds for j in 1:L
        tmp[j] = x[j] + dt * k3[j]
    end
    rhs_fast!(k4, tmp, params, G)
    
    # Combine
    @inbounds for j in 1:L
        x[j] += dt/6 * (k1[j] + 2k2[j] + 2k3[j] + k4[j])
        # Enforce Hermiticity
        x[j] = (x[j] + x[j]') / 2
    end
end

# ============================ Imbalance =======================================

function imbalance_domainwall_fast(x::Vector{SMatrix{3,3,ComplexF64,9}})
    L = length(x)
    half = L ÷ 2
    
    NL = 0.0
    NR = 0.0
    
    @inbounds for j in 1:half
        NL += n_symbol(x[j])
    end
    
    @inbounds for j in half+1:L
        NR += n_symbol(x[j])
    end
    
    return 2.0 * (NL - NR) / L
end

# ============================ Single trajectory ==============================

function run_single_trajectory(L::Int, params::PTWAParams, dt::Float64, tmax::Float64,
                               s::Vector{Int8}, rng::AbstractRNG, nsteps::Int)
    
    # Allocate all arrays once
    x = sample_initial_discrete_WH_fast(L, s; rng=rng)
    
    G = [@SMatrix zeros(ComplexF64, 3, 3) for _ in 1:L]
    k1 = [@SMatrix zeros(ComplexF64, 3, 3) for _ in 1:L]
    k2 = [@SMatrix zeros(ComplexF64, 3, 3) for _ in 1:L]
    k3 = [@SMatrix zeros(ComplexF64, 3, 3) for _ in 1:L]
    k4 = [@SMatrix zeros(ComplexF64, 3, 3) for _ in 1:L]
    tmp = [@SMatrix zeros(ComplexF64, 3, 3) for _ in 1:L]
    
    # Time evolution
    I = Vector{Float64}(undef, nsteps)
    
    @inbounds for step in 1:nsteps
        I[step] = imbalance_domainwall_fast(x)
        if step < nsteps
            step_rk4_fast!(x, params, dt; k1=k1, k2=k2, k3=k3, k4=k4, tmp=tmp, G=G)
        end
    end
    
    return I
end

# ============================ Disorder runner =================================

function run_disorder_imbalance_optimized(; L::Int, J::Float64, g::Float64, W::Float64,
                                          ntraj::Int, nreal::Int, tmax::Float64, 
                                          dt::Float64, seed::Int)
    
    s = init_domainwall10(L)
    nsteps = Int(ceil(tmax/dt)) + 1
    t = collect(range(0.0, tmax, length=nsteps))
    
    # Results storage - using regular array
    I_all = zeros(Float64, nsteps, nreal)
    
    Threads.@threads for r in 1:nreal
        # Each thread has its own RNG
        rng = MersenneTwister(seed + r * 10000 + Threads.threadid() * 1000)
        μ = W * (2rand(rng, L) .- 1)
        params = PTWAParams(L; J=J, g=g, μ=μ)
        
        # Accumulator for this realization
        I_real = zeros(Float64, nsteps)
        
        # Run trajectories sequentially for this realization
        for tr in 1:ntraj
            I_traj = run_single_trajectory(L, params, dt, tmax, s, rng, nsteps)
            I_real .+= I_traj
        end
        
        # Average over trajectories
        I_real ./= ntraj
        
        # Write to the results array (no lock needed because each thread writes to a different r)
        I_all[:, r] .= I_real
    end
    
    # Average over realizations
    I_avg = vec(mean(I_all, dims=2))
    
    return t, I_avg
end


# ============================ Main ============================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("Number of threads: $(Threads.nthreads())")
    
    # Test parameters - reduce for debugging
    L_list = [24, 48]  # Reduced for testing
    W_list = [2.5, 4.5, 6.0]     # Just one for testing
    
    J = 1.0
    g = 0.5
    
    # Reduced statistics for testing
    ntraj = 1000   # Reduced from 1000
    nreal = 100    # Reduced from 100, matches thread count
    tmax = 200.0   # Shorter time
    dt = 0.1
    seed = 1
    
    for L in L_list, W in W_list
        println("\n" * "="^60)
        println("OPTIMIZED pTWA NN: L=$L, W=$W, g=$g")
        println("="^60)
        
        @time t, I = run_disorder_imbalance_optimized(
            L=L, J=J, g=g, W=W,
            ntraj=ntraj, nreal=nreal,
            tmax=tmax, dt=dt, seed=seed
        )
        
        outfile = "pTWA_NN_Z3_domainwall_L$(L)_W$(W)_g$(g)_OPT.jld2"
        @save outfile t I L W g J ntraj nreal tmax dt
        println("Saved → $outfile")
        
        # Quick verification
        println("\nFirst 5 time points:")
        for i in 1:min(5, length(t))
            println("  t=$(round(t[i], digits=2)), I=$(round(I[i], digits=4))")
        end
    end
end