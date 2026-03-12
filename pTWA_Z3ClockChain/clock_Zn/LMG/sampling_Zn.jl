module Sampling

using LinearAlgebra
using Random

export sample_site_gaussian
export sample_initial_state_fully_polarized
export sample_initial_state_single_excitation
export sample_initial_state_domainwall
export sample_initial_state
export sample_initial_state_domainwall

############################################################
# Gaussian Wigner sampling (general Z_n)
############################################################

"""
    sample_site_gaussian(n, a0; rng=Random.default_rng())

Sample a single site prepared in the basis state |a0⟩ using a Gaussian
Wigner approximation.

Arguments
---------
n  : local Hilbert dimension
a0 : physical state index (0, ..., n-1)
"""
function sample_site_gaussian(n::Int, a0::Int; rng=Random.default_rng())

    x = zeros(ComplexF64, n, n)

    # Occupation probabilities of the reference basis state
    μ = zeros(Float64, n)
    μ[a0 + 1] = 1.0

    # Mean value
    x[a0 + 1, a0 + 1] = 1.0 + 0im

    # Gaussian fluctuations in off-diagonal entries
    for a in 1:n, b in a+1:n
        σ = sqrt((μ[a] + μ[b]) / 4)
        if σ > 0
            re = σ * randn(rng)
            im = σ * randn(rng)
            x[a, b] = re + 1im * im
            x[b, a] = re - 1im * im
        end
    end

    # Enforce Hermiticity
    x = (x + x') / 2

    # Normalize trace for numerical stability
    tr_x = real(tr(x))
    if abs(tr_x) > 1e-12
        x ./= tr_x
    end

    return x
end

############################################################
# Fully polarized initial state
############################################################

"""
    sample_initial_state_fully_polarized(n, N; a0=0, rng=Random.default_rng())

Sample a fully polarized product state

    |a0 a0 ... a0⟩

for N sites, represented as a vector of N local phase-space matrices.
"""
function sample_initial_state_fully_polarized(n::Int, N::Int;
                                              a0::Int=0,
                                              rng=Random.default_rng())

    x0 = Vector{Matrix{ComplexF64}}(undef, N)

    for j in 1:N
        x0[j] = sample_site_gaussian(n, a0; rng=rng)
    end

    return x0
end

############################################################
# Single excitation initial state
############################################################

"""
    sample_initial_state_single_excitation(n, N, j0; aexc=1, rng=...)

All sites in |0⟩ except site j0 in |aexc⟩.
"""
function sample_initial_state_single_excitation(n::Int, N::Int, j0::Int;
                                                aexc::Int=1,
                                                rng=Random.default_rng())

    x0 = Vector{Matrix{ComplexF64}}(undef, N)

    for j in 1:N
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
    sample_initial_state_domainwall_gaussian(n, N; a_left=0, a_right=n-1, rng=...)

Domain wall state:

    |a_left ... a_left | a_right ... a_right|
"""
function sample_initial_state_domainwall_gaussian(n::Int, N::Int;
                                                  a_left::Int=0,
                                                  a_right::Int=n-1,
                                                  rng=Random.default_rng())

    x0 = Vector{Matrix{ComplexF64}}(undef, N)

    half = N ÷ 2

    for j in 1:N
        a0 = (j <= half) ? a_left : a_right
        x0[j] = sample_site_gaussian(n, a0; rng=rng)
    end

    return x0
end

############################################################
# Unified interfaces
############################################################

"""
    sample_initial_state(method, n, N; state=:polarized, a0=0, j0=1, aexc=1, rng=...)

Unified interface for Gaussian initial states.
"""
function sample_initial_state(method::Symbol, n::Int, N::Int;
                              state::Symbol=:polarized,
                              a0::Int=0,
                              j0::Int=1,
                              aexc::Int=1,
                              rng=Random.default_rng())

    if method != :gaussian
        error("Only :gaussian sampling is implemented.")
    end

    if state == :polarized
        return sample_initial_state_fully_polarized(n, N; a0=a0, rng=rng)

    elseif state == :single_excitation
        return sample_initial_state_single_excitation(n, N, j0;
                                                      aexc=aexc,
                                                      rng=rng)
    else
        error("Unknown state symbol: $state")
    end
end

function sample_initial_state_domainwall(method::Symbol, n::Int, N::Int;
                                         a_left::Int=0,
                                         a_right::Int=n-1,
                                         rng=Random.default_rng())

    if method == :gaussian
        return sample_initial_state_domainwall_gaussian(n, N;
                                                        a_left=a_left,
                                                        a_right=a_right,
                                                        rng=rng)
    else
        error("Only :gaussian sampling is implemented.")
    end
end

end # module Sampling