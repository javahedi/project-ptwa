module Sampling

using LinearAlgebra
using Random

export sample_initial_state
export sample_initial_state_domainwall

############################################################
# Gaussian Wigner sampling (general Z_n)
############################################################

"""
Sample a single site in basis state |a0⟩ using Gaussian Wigner approximation.

n  : local Hilbert dimension
a0 : physical state index (0 ... n-1)
"""
function sample_site_gaussian(n::Int, a0::Int; rng=Random.default_rng())

    x = zeros(ComplexF64, n, n)

    # occupation vector
    μ = zeros(Float64, n)
    μ[a0 + 1] = 1.0

    # mean value
    x[a0 + 1, a0 + 1] = 1.0

    # Gaussian fluctuations
    for a in 1:n, b in a+1:n

        σ = sqrt((μ[a] + μ[b]) / 4)

        if σ > 0
            re = σ * randn(rng)
            im = σ * randn(rng)

            x[a,b] = re + 1im*im
            x[b,a] = re - 1im*im
        end
    end

    # enforce Hermiticity
    x = (x + x') / 2

    # normalize trace for stability
    tr_x = real(tr(x))
    if abs(tr_x) > 1e-12
        x ./= tr_x
    end

    return x
end


############################################################
# Single excitation initial state
############################################################

"""
Single excitation initial state.

All sites in |0⟩ except site j0 in |aexc⟩.
"""
function sample_initial_state_gaussian(n::Int, L::Int, j0::Int;
                                       aexc::Int=1,
                                       rng=Random.default_rng())

    x0 = Vector{Matrix{ComplexF64}}(undef, L)

    for j in 1:L
        if j == j0
            x0[j] = sample_site_gaussian(n, aexc; rng=rng)
        else
            x0[j] = sample_site_gaussian(n, 0; rng=rng)
        end
    end

    return x0
end


############################################################
# Domain wall initial state
############################################################

"""
Domain wall:

|a_left ... a_left | a_right ... a_right|
"""
function sample_initial_state_domainwall_gaussian(n::Int, L::Int;
                                                  a_left::Int=0,
                                                  a_right::Int=n-1,
                                                  rng=Random.default_rng())

    x0 = Vector{Matrix{ComplexF64}}(undef, L)

    half = L ÷ 2

    for j in 1:L
        a0 = (j <= half) ? a_left : a_right
        x0[j] = sample_site_gaussian(n, a0; rng=rng)
    end

    return x0
end


############################################################
# Unified interface
############################################################

function sample_initial_state(method::Symbol, n::Int, L::Int, j0::Int;
                              aexc::Int=1,
                              rng=Random.default_rng())

    if method == :gaussian
        return sample_initial_state_gaussian(n, L, j0;
                                             aexc=aexc,
                                             rng=rng)

    elseif method == :discrete
        error("Discrete Wigner sampling not implemented yet.")

    else
        error("Unknown sampling method")
    end
end


function sample_initial_state_domainwall(method::Symbol, n::Int, L::Int;
                                         a_left::Int=0,
                                         a_right::Int=n-1,
                                         rng=Random.default_rng())

    if method == :gaussian
        return sample_initial_state_domainwall_gaussian(n, L;
                                                        a_left=a_left,
                                                        a_right=a_right,
                                                        rng=rng)

    elseif method == :discrete
        error("Discrete Wigner sampling not implemented yet.")

    else
        error("Unknown sampling method")
    end
end

end # module Sampling