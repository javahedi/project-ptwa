using LinearAlgebra
using ParafermionDynamic
using Plots, JLD2

# -----------------------------
# Model parameters with sector
# -----------------------------
L = 15
n = 3
sector = 1
reflection_parity = 1
cc_parity = 1
J  = 1.0
g  = 0.0
α = 10.0
mu=zeros(L)

# Build model in specific sector
model = build_model(L; n=n,  sector=sector, reflection_parity=nothing, cc_parity=nothing,
                    hopping=long_range_hopping(L,J,α),
                    pair_hopping=long_range_hopping(L,g,α), mu=mu)

println("Q=$sector + refl(+1) + CC(+1) dim = ", length(model.states))
println("Full basis size would be: ", n^L)

# -----------------------------
# Initial state in the same sector
# -----------------------------
middle = div(L,2)
s0 = polarized_state(L, n, 0)


# Check initial state sector
initial_sector = sum(digit_at(s0, i, n) for i in 0:L-1) % n
println("Initial state sector: $initial_sector")

# Modify to stay in desired sector
s0 = set_digit(s0, middle, 1, n)

# Verify final state sector
final_sector = sum(BasisZn.digit_at(s0, i, n) for i in 0:L-1) % n
println("Final state sector: $final_sector")

ψ0 = zeros(ComplexF64, length(model.states))
ψ0[model.idxmap[s0]] = 1.0 + 0im

# -----------------------------
# Time evolution (stays in sector)
# -----------------------------
dt = 0.05
steps = 40
ψt = copy(ψ0)

occ_time = zeros(Float64, L, steps+1)
occ_time[:,1] = local_occupation(ψt, model)

for t in 1:steps
    global ψt = krylov_time_evolve(ψt, dt, apply_H!, model; kry_m=20)
    occ_time[:, t+1] = local_occupation(ψt, model)
    
    # Verify we stay in sector
    current_sector = sum(local_occupation(ψt, model)) % n
    println("Time step $t: sector = $current_sector")
end


heatmap(1:L, 0:steps, occ_time', 
        xlabel="Site", ylabel="Time step", 
        title="Excitation spreading (Sector $sector)")
savefig("example2.png")


# NOTE: Since we are using reflection symmetry (reflection_parity = ±1),
# the Hilbert space is reduced by identifying each state with its mirror.
# As a result, local observables like occupation only show "half" the system.
# To visualize the full chain, we need to "unfold" the result by mirroring
# the left half onto the right (or vice versa). This does not affect the dynamics
# but restores the full spatial picture for plotting.

function unfold_reflection(occ_folded::Array{Float64,2}, L::Int)
    
    middle = div(L, 2)
    if isodd(L)
        left = occ_folded[1:middle, :]
        center = occ_folded[middle+1, :]'  # middle site
        right = reverse(left, dims=1)
        return vcat(left, center, right)
    else
        left = occ_folded[1:middle, :]
        right = reverse(left, dims=1)
        return vcat(left, right)
    end
end


occ_full = unfold_reflection(occ_time, L)

heatmap(1:L, 0:steps, occ_full', 
        xlabel="Site", ylabel="Time step", 
        title="Excitation spreading (Sector $sector + refl(+1))")
savefig("example2_unfolded.png")

 
JLD2.@save "data_exact_L15.jld2" occ_full