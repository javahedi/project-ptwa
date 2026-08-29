module Sampling

using Random

export sample_site_gaussian, sample_initial_state_fully_polarized

function sample_site_gaussian(n::Int, a0::Int; rng=Random.default_rng())
    0 <= a0 < n || throw(ArgumentError("a0 must satisfy 0 <= a0 < n"))
    x = zeros(ComplexF64, n, n)
    μ = zeros(Float64, n)
    μ[a0 + 1] = 1.0
    x[a0 + 1, a0 + 1] = 1.0 + 0im

    @inbounds for a in 1:n-1, b in a+1:n
        σ = sqrt((μ[a] + μ[b]) / 4)
        if σ > 0
            q = σ * randn(rng)
            p = σ * randn(rng)
            x[a,b] = q + 1im*p
            x[b,a] = q - 1im*p
        end
    end
    return x
end

function sample_initial_state_fully_polarized(n::Int, N::Int;
                                              a0::Int=0,
                                              rng=Random.default_rng())
    [sample_site_gaussian(n, a0; rng=rng) for _ in 1:N]
end

end
