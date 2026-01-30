
using LinearAlgebra
include("../src/BasisZn.jl")
using .BasisZn

println("==== Z_n Basis & Sector Test ====")

# ----------------------------
# Parameters
# ----------------------------
N = 4       # number of sites
n = 3       # Z_n, e.g., n=3 or 4
sector = 2  # desired "particle number modulo n"

# ----------------------------
# 1. Build full base-n basis
# ----------------------------
full_states, _ = build_full_basis(N, n)
println("Total number of states (full basis): ", length(full_states))
println("First 5 states (base-n): ", full_states[1:min(5,end)])

# Print digits of first state
println("Digits of first state:")
for pos in 0:N-1
    print(digit_at(full_states[1], pos, n), " ")
end
println("\n")

# ----------------------------
# 2. Build sector basis
# ----------------------------
sector_states, idxmap = build_sector_basis(N, n, sector)
println("Number of states in sector ", sector, ": ", length(sector_states))
println("First 5 sector states: ", sector_states[1:min(5,end)])

# Verify sector property
println("Verifying sector property (sum of digits mod n):")
for s in sector_states[1:min(5,end)]
    total = sum(digit_at(s, pos, n) for pos in 0:N-1) % n
    println("State=", s, " sum mod n=", total)
end

println("==== Z_n Basis & Sector Test Completed ====")
