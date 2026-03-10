using LinearAlgebra
using Base.Threads
using ParafermionDynamic  # Use the main module instead of individual submodules
using Plots, JLD2
# -----------------------------
# Model parameters
# -----------------------------
L = 15
n = 3
J  = 1.0
g  = 0.0
α = 10.0
μ = zeros(L)

# Build model in specific sector
model = build_model(L; n=n, 
                        hopping=long_range_hopping(L,J,α),
                        pair_hopping=long_range_hopping(L,J,α), 
                        mu=μ)

# -----------------------------
# Initial state: excite middle site
# -----------------------------
middle = div(L,2)                  # middle site index (0-based)
s0 = polarized_state(L, n, 0)      # all zeros
s0 = set_digit(s0, middle, 1, n)    # excite middle site to 1

ψ0 = zeros(ComplexF64, length(model.states))
ψ0[model.idxmap[s0]] = 1.0 + 0im   # normalized initial vector

# -----------------------------
# Time evolution parameters
# -----------------------------
dt = 0.05
steps = 40
ψt = copy(ψ0)

# Store occupations
occ_time = zeros(Float64, L, steps+1)
occ_time[:,1] = local_occupation(ψt, model)  # Removed ObservablesZn.

# -----------------------------
# Time evolution loop (Krylov)
# -----------------------------
@time for t in 1:steps
    global ψt = krylov_time_evolve(ψt, dt, apply_H!, model; kry_m=20)
    occ_time[:, t+1] = local_occupation(ψt, model)
    println("Time step $t")
end



# -----------------------------
# Optional: visualize spread
# -----------------------------

heatmap(1:L, 0:steps, occ_time', 
        xlabel="Time step", ylabel="Site", 
        title="Excitation spreading")
savefig("example1.png")

