module Sampling

using LinearAlgebra
using Random

export build_discrete_cache, sample_initial_state

const n = 3
const ω = cis(2π/n)

# ============================================================
# Gaussian sampling
# ============================================================

function sample_site_gaussian(a0::Int; rng=Random.default_rng())
    0 <= a0 <= 2 || throw(ArgumentError("a0 must be 0, 1, or 2"))

    x = zeros(ComplexF64,3,3)
    μ = zeros(Float64,3)
    μ[a0+1] = 1.0

    x[a0+1,a0+1] = 1.0

    for a in 1:2, b in a+1:3
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

function sample_initial_state_gaussian(L,j0; aexc=1, rng=Random.default_rng())
    x0 = Vector{Matrix{ComplexF64}}(undef,L)
    for j in 1:L
        a0 = j == j0 ? aexc : 0
        x0[j] = sample_site_gaussian(a0; rng=rng)
    end
    x0
end

# ============================================================
# Discrete Wigner sampling
# ============================================================

struct DiscreteCache
    A::Vector{Matrix{ComplexF64}}
    W::NTuple{3,Vector{Float64}}
    x_lookup::Vector{Matrix{ComplexF64}}
end

function build_Aqp()
    Z = Diagonal(ComplexF64[1,ω,ω^2])
    X = ComplexF64[
        0 1 0
        0 0 1
        1 0 0
    ]

    A = Matrix{ComplexF64}[]

    for q in 0:2, p in 0:2
        Aqp = zeros(ComplexF64,3,3)

        for m in 0:2, k in 0:2
            phase = ω^(p*k - q*m + 2*m*k)
            Aqp .+= phase .* (Z^m * X^k)
        end

        Aqp ./= 3
        Aqp .= 0.5 .* (Aqp .+ Aqp')
        push!(A,Aqp)
    end

    A
end

function build_wigner_prob(A,a)
    ρ = zeros(ComplexF64,3,3)
    ρ[a+1,a+1] = 1.0

    W = [real(tr(ρ*Aq))/3 for Aq in A]

    minimum(W) < -1e-12 &&
        error("Negative discrete Wigner weight encountered: $(minimum(W))")

    W = max.(W,0.0)
    W ./= sum(W)

    W
end

function phasepoint_to_x(Aqp)
    x = zeros(ComplexF64,3,3)

    # x^{ab} = Tr(A X^{ab}) = A_{ba}
    for a in 1:3, b in 1:3
        x[a,b] = Aqp[b,a]
    end

    x
end

function build_discrete_cache()
    A = build_Aqp()

    W = (
        build_wigner_prob(A,0),
        build_wigner_prob(A,1),
        build_wigner_prob(A,2)
    )

    x_lookup = [phasepoint_to_x(Aq) for Aq in A]

    DiscreteCache(A,W,x_lookup)
end

@inline function sample_index(W,rng)
    r = rand(rng)
    s = 0.0

    for i in eachindex(W)
        s += W[i]
        r <= s && return i
    end

    return lastindex(W)
end

function sample_initial_state_discrete(L,j0,cache; aexc=1, rng=Random.default_rng())
    x0 = Vector{Matrix{ComplexF64}}(undef,L)

    for j in 1:L
        a0 = j == j0 ? aexc : 0
        idx = sample_index(cache.W[a0+1],rng)
        x0[j] = copy(cache.x_lookup[idx])
    end

    x0
end

# ============================================================
# Unified interface
# ============================================================

function sample_initial_state(method::Symbol,L,j0;
                              cache=nothing,
                              aexc=1,
                              rng=Random.default_rng())

    if method == :gaussian
        return sample_initial_state_gaussian(
            L,j0;
            aexc=aexc,
            rng=rng
        )

    elseif method == :discrete
        isnothing(cache) && error("Discrete sampling requires a cache.")
        return sample_initial_state_discrete(
            L,j0,cache;
            aexc=aexc,
            rng=rng
        )

    else
        error("Unknown sampling method: $method")
    end
end

end
