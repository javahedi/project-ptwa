using LinearAlgebra
using Random
include("sampling.jl")
using .Sampling


# ----------------------------
# Choose sampling method
# ----------------------------
#sampling = :gaussian
sampling = :discrete

# Discrete cache (if needed)
cache = nothing
if sampling == :discrete
    cache = build_discrete_cache()
end


function check_initial_means(sampling, L, j0; cache=cache, Ntest=2000, seed=1)
    rng = MersenneTwister(seed)
    m_center11 = 0.0
    m_other00  = 0.0
    m_offdiag  = 0.0
    for _ in 1:Ntest
        x0 = sample_initial_state(sampling, L, j0; cache=cache, aexc=1, rng=rng)
        m_center11 += real(x0[j0][2,2])
        m_other00  += real(x0[1][1,1])
        m_offdiag  += abs(x0[j0][1,2])
    end
    println("⟨x_center(11)⟩ = ", m_center11/Ntest, "  (should be 1)")
    println("⟨x_other(00)⟩  = ", m_other00/Ntest,  "  (should be 1)")
    println("⟨|x_center(01)|⟩ = ", m_offdiag/Ntest, "  (should be ~0 for discrete, >0 for Gaussian)")
end

L=11
j0 = (L + 1) ÷ 2
println("Checking initial means for sampling = ", sampling)
check_initial_means(sampling, L, j0; cache=cache, Ntest=2000, seed=1)