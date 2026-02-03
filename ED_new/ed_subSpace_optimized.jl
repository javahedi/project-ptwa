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
    return powers3
end

# ------------------------ Fast fixed-N basis construction ---------------------

"""
Recursively generate all states with fixed total occupation Ntarget.
Much faster than scanning all 3^L states for L > 12.
"""
function build_fixedN_basis_fast(L::Int, Ntarget::Int)
    powers3 = precompute_powers3(L)
    states = Int[]
    
    function backtrack(pos::Int, sum_n::Int, current_idx::Int)
        if pos > L
            if sum_n == Ntarget
                push!(states, current_idx)
            end
            return
        end
        
        # Early pruning: if we can't reach Ntarget even with max occupancy
        remaining_sites = L - pos + 1
        if sum_n > Ntarget || sum_n + 2 * remaining_sites < Ntarget
            return
        end
        
        for n in 0:2
            if sum_n + n <= Ntarget
                backtrack(pos + 1, sum_n + n, current_idx + n * powers3[pos])
            end
        end
    end
    
    backtrack(1, 0, 0)
    
    dim_red = length(states)
    full_to_red = zeros(Int, 3^L)
    for (ridx, idx0) in enumerate(states)
        full_to_red[idx0 + 1] = ridx
    end
    
    return full_to_red, states, dim_red, powers3
end

# ---------------------- Precomputed data structures ---------------------------

struct PrecomputedData
    occ_matrix::Matrix{Int8}           # occ[ridx, site] = occupation (0,1,2)
    diag_energies::Vector{Float64}     # Σ μ_j n_j for each state
    hopping_map::Vector{Vector{Tuple{Int, Float64}}}  # (neighbor_ridx, amplitude)
    NL_vec::Vector{Float64}            # NL = Σ_{j<=L/2} n_j for each state
    NR_vec::Vector{Float64}            # NR = Σ_{j>L/2} n_j for each state
end

function precompute_all_data(red_to_full::Vector{Int}, powers3::Vector{Int}, 
                             μ::Vector{Float64}, L::Int, J::Float64, g::Float64,
                             full_to_red::Vector{Int})  # <-- ADDED THIS ARGUMENT
    dim_red = length(red_to_full)
    half = L ÷ 2
    
    # 1. Occupation matrix
    occ_matrix = Matrix{Int8}(undef, dim_red, L)
    diag_energies = Vector{Float64}(undef, dim_red)
    NL_vec = Vector{Float64}(undef, dim_red)
    NR_vec = Vector{Float64}(undef, dim_red)
    
    # Use serial loop instead of threaded to avoid scoping issues
    @inbounds for ridx in 1:dim_red
        idx0 = red_to_full[ridx]
        e_diag = 0.0
        NL = 0.0
        NR = 0.0
        
        for j in 1:L
            pow = powers3[j]
            occ = (idx0 ÷ pow) % 3
            occ_matrix[ridx, j] = occ
            e_diag += μ[j] * occ
            
            if j <= half
                NL += occ
            else
                NR += occ
            end
        end
        
        diag_energies[ridx] = e_diag
        NL_vec[ridx] = NL
        NR_vec[ridx] = NR
    end
    
    # 2. Hopping connections (amplitudes are real for this Hamiltonian)
    hopping_map = [Vector{Tuple{Int, Float64}}() for _ in 1:dim_red]
    
    @inbounds for ridx in 1:dim_red
        idx0 = red_to_full[ridx]
        
        # Single hopping terms
        for j in 1:(L-1)
            nj = occ_matrix[ridx, j]
            njp = occ_matrix[ridx, j+1]
            powj = powers3[j]
            powjp = powers3[j+1]
            
            # f_j† f_{j+1}
            if nj <= 1 && njp >= 1
                idx0p = idx0 + powj - powjp
                rp = full_to_red[idx0p + 1]
                @assert rp != 0
                push!(hopping_map[rp], (ridx, -J * (1 - g)))
                push!(hopping_map[ridx], (rp, -J * (1 - g)))
            end
            
            # (f_j†)^2 f_{j+1}^2
            if nj == 0 && njp == 2
                idx0p = idx0 + 2powj - 2powjp
                rp = full_to_red[idx0p + 1]
                @assert rp != 0
                push!(hopping_map[rp], (ridx, -J * g))
                push!(hopping_map[ridx], (rp, -J * g))
            end
        end
    end
    
    return PrecomputedData(occ_matrix, diag_energies, hopping_map, NL_vec, NR_vec)
end

# ----------------------------- Initial state ---------------------------------

function psi0_domainwall_fixedN(L::Int, powers3::Vector{Int},
                                full_to_red::Vector{Int})
    @assert iseven(L)
    half = L ÷ 2
    
    idx0 = 0
    for j in 1:half
        idx0 += 1 * powers3[j]
    end
    
    ridx = full_to_red[idx0 + 1]
    @assert ridx != 0
    
    # Find actual dimension of reduced basis
    max_ridx = 0
    for val in full_to_red
        if val > max_ridx
            max_ridx = val
        end
    end
    
    ψ = zeros(ComplexF64, max_ridx)
    ψ[ridx] = 1.0 + 0im
    return ψ
end

# ----------------------------- Matrix-free H*v -------------------------------

struct ParaEDOpFixedN
    L::Int
    J::Float64
    g::Float64
    μ::Vector{Float64}
    dim_red::Int
    precomp::PrecomputedData
    full_to_red::Vector{Int}
end

Base.size(A::ParaEDOpFixedN) = (A.dim_red, A.dim_red)

function LinearAlgebra.mul!(y::Vector{ComplexF64}, A::ParaEDOpFixedN, x::Vector{ComplexF64})
    fill!(y, 0.0 + 0im)
    
    # Diagonal part (onsite disorder)
    diagE = A.precomp.diag_energies
    @inbounds for ridx in 1:A.dim_red
        xi = x[ridx]
        if xi != 0
            y[ridx] += diagE[ridx] * xi
        end
    end
    
    # Hopping terms from precomputed map
    hopping_map = A.precomp.hopping_map
    @inbounds for ridx in 1:A.dim_red
        xi = x[ridx]
        if xi != 0
            for (neigh_idx, amp) in hopping_map[ridx]
                y[neigh_idx] += amp * xi
            end
        end
    end
    
    return y
end

# Make operator callable for KrylovKit
(A::ParaEDOpFixedN)(y::Vector{ComplexF64}, x::Vector{ComplexF64}) = mul!(y, A, x)
(A::ParaEDOpFixedN)(x::Vector{ComplexF64}) = (y = similar(x); mul!(y, A, x); y)

# ----------------------------- Fast observable -------------------------------

function imbalance_domainwall_fast(ψ::AbstractVector{ComplexF64},
                                   precomp::PrecomputedData,
                                   L::Int)
    NL = 0.0
    NR = 0.0
    
    @inbounds for ridx in eachindex(ψ)
        w = abs2(ψ[ridx])
        NL += w * precomp.NL_vec[ridx]
        NR += w * precomp.NR_vec[ridx]
    end
    
    return (2.0 / L) * (NL - NR)
end

# ----------------------------- Krylov time evolution --------------------------

function step_krylov!(ψ::Vector{ComplexF64}, A::ParaEDOpFixedN, dt::Float64;
                      krylovdim::Int=30, tol::Float64=1e-9)
    ψnew, info = exponentiate(A, (-1im)*dt, ψ; 
                              krylovdim=krylovdim, tol=tol,
                              ishermitian=true, verbosity=0)
    copyto!(ψ, ψnew)
    return ψ
end

# ----------------------------- Single disorder run ----------------------------

function run_one_realization_fixedN_optimized(; L::Int, J::Float64, g::Float64, W::Float64,
                                              tmax::Float64, dt::Float64, seed::Int,
                                              krylovdim::Int=30, tol::Float64=1e-9,
                                              save_every::Int=1)

    rng = MersenneTwister(seed)
    μ = W .* (2 .* rand(rng, L) .- 1)
    
    Ntarget = L ÷ 2
    full_to_red, red_to_full, dim_red, powers3 = build_fixedN_basis_fast(L, Ntarget)
    println("Fixed-N sector: L=$L, N=$Ntarget => dim_red=$dim_red (full 3^L=$(3^L))")
    
    # Precompute all data structures (PASS full_to_red as argument)
    precomp = precompute_all_data(red_to_full, powers3, μ, L, J, g, full_to_red)
    
    H = ParaEDOpFixedN(L, J, g, μ, dim_red, precomp, full_to_red)
    ψ = psi0_domainwall_fixedN(L, powers3, full_to_red)
    
    # Time evolution
    t_points = ceil(Int, tmax/dt) + 1
    I = zeros(Float64, t_points)
    
    step_count = 0
    save_idx = 1
    
    # Initial state
    I[save_idx] = imbalance_domainwall_fast(ψ, precomp, L)
    save_idx += 1
    step_count += 1
    
    t_elapsed = 0.0
    while t_elapsed < tmax - 1e-10
        step_krylov!(ψ, H, dt; krylovdim=krylovdim, tol=tol)
        t_elapsed += dt
        step_count += 1
        
        if step_count % save_every == 0
            I[save_idx] = imbalance_domainwall_fast(ψ, precomp, L)
            save_idx += 1
        end
    end
    
    # If we didn't save at the final time, compute it
    if (step_count-1) % save_every != 0
        I[save_idx] = imbalance_domainwall_fast(ψ, precomp, L)
    end
    
    t = collect(range(0, tmax, length=t_points))
    return t, I, μ, dim_red
end

# ----------------------------- Parallel disorder average ---------------------

function run_disorder_average_parallel(; L::Int, J::Float64=1.0, g::Float64=0.5,
                                       W::Float64=4.5, nreal::Int=30,
                                       tmax::Float64=50.0, dt::Float64=0.1,
                                       seed::Int=1, krylovdim::Int=30, tol::Float64=1e-9,
                                       save_every::Int=1)

    println("Running $nreal realizations for L=$L, W=$W on $(Threads.nthreads()) threads")
    
    # Preallocate results array
    results = Vector{Any}(undef, nreal)
    
    # Run in parallel using Threads.@threads
    Threads.@threads for r in 1:nreal
        local_seed = seed + 10_000 * r  # Make seed local to avoid race conditions
        println("Starting realization $r / $nreal (thread $(Threads.threadid()))")
        
        # Each thread calls the function independently
        t, I, μ, dim_red = run_one_realization_fixedN_optimized(
            L=L, J=J, g=g, W=W, tmax=tmax, dt=dt,
            seed=local_seed, krylovdim=krylovdim, tol=tol,
            save_every=save_every
        )
        results[r] = (t, I, dim_red)
    end
    
    # Extract results
    t_ref = results[1][1]
    I_all = [res[2] for res in results]
    dim_red = results[1][3]
    
    # Average
    I_avg = zeros(Float64, length(t_ref))
    for I_curr in I_all
        I_avg .+= I_curr
    end
    I_avg ./= nreal
    
    return t_ref, I_avg, dim_red
end

# ----------------------------- Main -------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    
    # Check if we have threads available
    println("Number of threads: $(Threads.nthreads())")
    
    L_list = [12]  # Start with smaller sizes
    J = 1.0
    g = 0.5
    W_list = [2.5, 4.5, 6.0]  # Start with one W value for testing
    
    tmax = 200.0  # Shorter for testing
    dt   = 0.1
    
    nreal_list = [60, 120, 200]  # Fewer realizations for testing
    seed  = 1
    
    krylovdim = 30
    tol = 1e-9
    save_every = 1  # Save every 2 time steps
    L = L_list[1]
    for (W, nr) in zip(W_list, nreal_list)
        println("\n" * "="^60)
        println("Fixed-N Krylov ED: L=$(L), W=$W, nreal=$(nr)")
        println("="^60)
        
        @time t, I, dim_red = run_disorder_average_parallel(
            L=L, J=J, g=g, W=W, nreal=nr,
            tmax=tmax, dt=dt, seed=seed,
            krylovdim=krylovdim, tol=tol,
            save_every=save_every
        )
        
        outfile = "ED_Krylov_fixedN_domainwall_Z3_L$(L)_W$(W)_g$(g)_new.jld2"
        @save outfile t I L W g J tmax dt nr krylovdim tol dim_red save_every
        println("Saved → $outfile (dim_red=$dim_red)")
        
        # Quick plot to verify
        println("\nFirst few imbalance values:")
        for i in 1:min(5, length(t))
            println("t=$(round(t[i], digits=2)), I=$(round(I[i], digits=4))")
        end
    end
end