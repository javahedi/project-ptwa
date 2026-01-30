###############################################################################
# Exact dynamics (ED/Krylov) for the long-range Z₃ clock/parafermion chain
# using ParafermionDynamic.jl — adapted for direct comparison with pTWA.
#
# Target Hamiltonian (open boundary conditions):
#   H = -∑_{r=1}^{L-1} J_r ∑_{j=1}^{L-r} ( Z_j Z_{j+r}† + Z_j† Z_{j+r} )
#       -∑_{j=1}^{L} ( f X_j + f* X_j† )
#
# This script:
#   1) builds the long-range model with J_r = J / r^α,
#   2) prepares product initial states (single excitation, Néel, domain wall),
#   3) evolves exactly in time using Krylov,
#   4) computes observables compatible with pTWA:
#        - ⟨Z_j(t)⟩ (complex)
#        - populations P_a(j,t) = ⟨|a⟩⟨a|⟩  (real, sums to 1)
#   5) saves results to JLD2 and produces heatmaps.
#
# Usage:
#   julia --project=. run_exact_Z3_longrange.jl
#
# Notes:
# - This assumes the ParafermionDynamic.jl API provides:
#     build_model, long_range_hopping, apply_H!, krylov_time_evolve,
#     polarized_state, set_digit, expectation_Z, local_populations
#   If any of these differ slightly in your version, tell me the method names
#   (from your README/examples) and I’ll patch the script accordingly.
###############################################################################

using LinearAlgebra
using ParafermionDynamic
using JLD2
using Plots

# ----------------------------- Helpers: initial states ------------------------

"""
    init_state_single_excitation(model, L, n; background=0, excitation=1, pos=nothing)

Return a computational-basis product state with all sites in `background` except
a single site `pos` in `excitation`.

- `background`, `excitation` ∈ {0,1,2}
- `pos` is 1-based. If not provided, chooses center (L+1)/2 for odd L.
Returns:
- `ψ0::Vector{ComplexF64}` normalized state vector in the model basis
- `s0` integer-encoded basis label used by `model.idxmap`
"""
function init_state_single_excitation(model, L::Int, n::Int;
                                      background::Int=0, excitation::Int=1, pos=nothing)
    (0 <= background <= n-1) || error("background must be in 0..$(n-1)")
    (0 <= excitation <= n-1) || error("excitation must be in 0..$(n-1)")
    if pos === nothing
        isodd(L) || error("Default center pos requires odd L; provide pos explicitly.")
        pos = (L + 1) ÷ 2
    end
    (1 <= pos <= L) || error("pos must be in 1..L")

    # ParafermionDynamic example uses 0-based digit positions for set_digit
    s0 = polarized_state(L, n, background)
    s0 = set_digit(s0, pos - 1, excitation, n)

    ψ0 = zeros(ComplexF64, length(model.states))
    ψ0[model.idxmap[s0]] = 1.0 + 0im
    return ψ0, s0
end

"""
    init_state_neel01(model, L, n)

"Néel-like" product state in the {0,1} subspace: |0,1,0,1,...⟩.
"""
function init_state_neel01(model, L::Int, n::Int)
    n == 3 || @warn "init_state_neel01 designed for n=3; continuing anyway."
    s0 = polarized_state(L, n, 0)
    for j in 1:L
        digit = isodd(j) ? 0 : 1
        s0 = set_digit(s0, j - 1, digit, n)
    end
    ψ0 = zeros(ComplexF64, length(model.states))
    ψ0[model.idxmap[s0]] = 1.0 + 0im
    return ψ0, s0
end

"""
    init_state_domainwall(model, L, n; left=0, right=1)

Domain wall product state: left half in |left⟩, right half in |right⟩.
"""
function init_state_domainwall(model, L::Int, n::Int; left::Int=0, right::Int=1)
    (0 <= left <= n-1) || error("left must be in 0..$(n-1)")
    (0 <= right <= n-1) || error("right must be in 0..$(n-1)")
    s0 = polarized_state(L, n, left)
    mid = L ÷ 2
    for j in (mid+1):L
        s0 = set_digit(s0, j - 1, right, n)
    end
    ψ0 = zeros(ComplexF64, length(model.states))
    ψ0[model.idxmap[s0]] = 1.0 + 0im
    return ψ0, s0
end



function local_populations(ψ, model)
    L, n = model.L, model.n
    P = zeros(Float64, L, n)
    for (idx, state) in enumerate(model.states)
        w = abs2(ψ[idx])
        for j in 0:(L-1)
            a = digit_at(state, j, n)
            P[j+1, a+1] += w
        end
    end
    return P
end


# ----------------------------- Measurements for Fig. A & C -------------------

"""
    compute_R2(P1, j0)

Mean-square radius R^2 = Σ_j (j-j0)^2 P1(j) / Σ_j P1(j)
"""
function compute_R2(P1::AbstractVector{<:Real}, j0::Int)
    norm = sum(P1)
    norm ≈ 0 && return 0.0
    return sum((j - j0)^2 * P1[j] for j in eachindex(P1)) / norm
end

# ----------------------------- Core: run ED/Krylov ----------------------------

"""
    run_exact_Z3_longrange(; L=13, α=3.0, J=1.0, f=0.8, dt=0.05, steps=80,
                            init=:single, save_prefix="ED")

Run exact Krylov time evolution for the long-range Z₃ chain and save observables.

Returns:
- times :: Vector{Float64} length steps+1
- Zt    :: Matrix{ComplexF64} (steps+1, L)    where Zt[tidx, j] = ⟨Z_j(t)⟩
- Pt    :: Array{Float64,3}   (steps+1, L, n) where Pt[tidx, j, a] = P_a(j,t)
- meta  :: Dict with run parameters
"""
function run_exact_Z3_longrange(; L::Int=13, α::Float64=3.0, J::Float64=1.0,
                                f::Float64=0.8, dt::Float64=0.05, steps::Int=80,
                                init::Symbol=:single, save_prefix::String="ED")

    n = 3
    μ = zeros(L)

    # Long-range couplings J_r = J / r^α (ParafermionDynamic helper)
    Jr = long_range_hopping(L, J, α)

    # Build model (open boundaries)
    model = build_model(L; n=n,
                        hopping=Jr,
                        pair_hopping=Jr,
                        mu=μ)

    # Prepare initial state
    ψ0 = nothing
    s0 = nothing
    if init == :single
        ψ0, s0 = init_state_single_excitation(model, L, n; background=0, excitation=1,
                                              pos=isodd(L) ? (L+1)÷2 : (L÷2))
    elseif init == :neel01
        ψ0, s0 = init_state_neel01(model, L, n)
    elseif init == :domainwall
        ψ0, s0 = init_state_domainwall(model, L, n; left=0, right=1)
    else
        error("Unknown init=$init. Use :single, :neel01, or :domainwall.")
    end

    # Time grid
    times = dt .* collect(0:steps)

    # Storage
    Zt  = zeros(ComplexF64, steps+1, L)
    Pt  = zeros(Float64, steps+1, L, n)   # P0,P1,P2
    P1t = zeros(Float64, steps+1, L)      # P1 only
    S   = zeros(Float64, steps+1)         # survival prob at center
    R2  = zeros(Float64, steps+1)         # mean-square radius of P1

    # Initialize
    ψt = copy(ψ0)
    j0 = (L + 1) ÷ 2

    # t=0 observables
    Zt[1, :] .= local_Z(ψt, model)

    P = local_populations(ψt, model)
    Pt[1, :, :] .= P
    P1t[1, :]   .= P[:,2]     # state |1⟩ population

    S[1]  = P1t[1, j0]
    R2[1] = compute_R2(P1t[1, :], j0)

    # Time evolution
    @time for k in 1:steps
        ψt = krylov_time_evolve(ψt, dt, apply_H!, model; kry_m=20)

        # Z expectation
        Zt[k+1, :] .= local_Z(ψt, model)

        # populations + derived measures
        P = local_populations(ψt, model)
        Pt[k+1, :, :] .= P
        P1t[k+1, :]   .= P[:,2]

        S[k+1]  = P1t[k+1, j0]
        R2[k+1] = compute_R2(P1t[k+1, :], j0)

        if k % 10 == 0
            println("ED step $k / $steps")
        end
    end

    meta = Dict(
        "L" => L,
        "n" => n,
        "J" => J,
        "alpha" => α,
        "f" => f,
        "dt" => dt,
        "steps" => steps,
        "init" => String(init),
        "state_label" => s0,
        "boundary" => "open"
    )

    outfile = "$(save_prefix)_Z3_L$(L)_alpha$(α)_init$(init).jld2"
    @save outfile times Zt Pt P1t S R2 meta
    println("Saved ED data → $outfile")

    return times, Zt, Pt, P1t, S, R2, meta
end


# ----------------------------- Plotting utilities -----------------------------

"""
    plot_heatmaps(times, Zt, Pt; α, L, init, prefix="ED")

Generate PDF heatmaps for Re⟨Z⟩ and populations P_a.
"""
function plot_heatmaps(times, Zt, Pt; α, L, init, prefix="ED")
    # Re⟨Z⟩ heatmap
    pltZ = heatmap(1:L, times, real.(Zt);
                   xlabel="site j", ylabel="time t",
                   title="ED Re⟨Z_j(t)⟩ (L=$L, α=$α, init=$init)",
                   aspect_ratio=:auto,
                   colorbar_title="Re⟨Z⟩")
    savefig(pltZ, "$(prefix)_ReZ_heatmap_L$(L)_alpha$(α)_init$(init).pdf")

    # Populations
    for a in 1:3
        pltP = heatmap(1:L, times, Pt[:,:,a];
                       xlabel="site j", ylabel="time t",
                       title="ED population P_$(a-1)(j,t) (L=$L, α=$α, init=$init)",
                       aspect_ratio=:auto,
                       colorbar_title="P")
        savefig(pltP, "$(prefix)_P$(a-1)_heatmap_L$(L)_alpha$(α)_init$(init).pdf")
    end

    println("Saved heatmaps (PDF).")
end

# ----------------------------- Main: configure run ----------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    # Choose L=10/12 for ED comparison with your pTWA; here we use L=13 test.
    L = 15
    α = 0.5
    J = 1.0
    f = 0.8

    dt = 0.025
    steps = 80

    # Initial state options: :single, :neel01, :domainwall
    init = :single

    times, Zt, Pt, P1t, S, R2, meta =
    run_exact_Z3_longrange(L=L, α=α, J=J, f=f,
                           dt=dt, steps=steps,
                           init=init, save_prefix="ED")


    plot_heatmaps(times, Zt, Pt; α=α, L=L, init=String(init), prefix="ED")

    # Quick sanity prints
    j0 = (L + 1) ÷ 2
    println("t=0: Re⟨Z_center⟩ = ", real(Zt[1, j0]), "  Re⟨Z_edge⟩ = ", real(Zt[1, 1]))
end
