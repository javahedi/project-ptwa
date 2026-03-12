module Sampling

using LinearAlgebra
using Random

export sample_initial_state, build_discrete_cache
export sample_initial_state_domainwall

const n = 3
const ω = cis(2π/n)

# -------------------------------------------------
# Gaussian sampling
# -------------------------------------------------

function sample_site_gaussian(a0::Int; rng=Random.default_rng())

    x = zeros(ComplexF64,3,3)

    μ = zeros(Float64,3)
    μ[a0+1] = 1.0

    x[a0+1,a0+1] = 1

    for a in 1:3, b in a+1:3
        σ = sqrt((μ[a] + μ[b]) / 4)

        if σ > 0
            re = σ * randn(rng)
            im = σ * randn(rng)

            x[a,b] = re + 1im*im
            x[b,a] = re - 1im*im
        end
    end

    return x
end


function sample_initial_state_gaussian(L,j0;aexc=1,rng=Random.default_rng())

    x0 = Vector{Matrix{ComplexF64}}(undef,L)

    for j in 1:L
        if j == j0
            x0[j] = sample_site_gaussian(aexc;rng=rng)
        else
            x0[j] = sample_site_gaussian(0;rng=rng)
        end
    end

    return x0
end


# -------------------------------------------------
# Discrete Wigner sampling
# -------------------------------------------------

struct DiscreteCache
    A::Vector{Matrix{ComplexF64}}
    W0::Vector{Float64}
    W1::Vector{Float64}
    x_lookup::Vector{Matrix{ComplexF64}}
end


function build_Aqp()

    Z = Diagonal([1,ω,ω^2])
    X = [0 1 0;
         0 0 1;
         1 0 0]

    A = Matrix{ComplexF64}[]

    for q in 0:2, p in 0:2

        Aqp = zeros(ComplexF64,3,3)

        for m in 0:2, k in 0:2
            phase = ω^(p*k - q*m + 2*m*k)
            Aqp += phase * (Z^m * X^k)
        end

        push!(A, Aqp/3)
    end

    return A
end


function build_wigner_prob(A,a)

    ρ = zeros(ComplexF64,3,3)
    ρ[a+1,a+1] = 1

    W = zeros(Float64,9)

    for i in 1:9
        W[i] = real(tr(ρ*A[i]))/3
    end

    W ./= sum(W)

    return W
end


function phasepoint_to_x(Aqp)

    x = zeros(ComplexF64,3,3)

    for a in 1:3, b in 1:3
        Xab = zeros(ComplexF64,3,3)
        Xab[a,b] = 1
        x[a,b] = tr(Aqp * Xab)
    end

    return x
end


function build_discrete_cache()

    A = build_Aqp()

    x_lookup = [phasepoint_to_x(A[i]) for i in 1:9]

    W0 = build_wigner_prob(A,0)
    W1 = build_wigner_prob(A,1)

    return DiscreteCache(A,W0,W1,x_lookup)

end


function sample_index(W,rng)

    r = rand(rng)
    s = 0.0

    for i in eachindex(W)
        s += W[i]
        if r < s
            return i
        end
    end

    return length(W)
end


function sample_initial_state_discrete(L,j0,cache;rng=Random.default_rng())

    x0 = Vector{Matrix{ComplexF64}}(undef,L)

    for j in 1:L

        if j == j0
            idx = sample_index(cache.W1,rng)
        else
            idx = sample_index(cache.W0,rng)
        end

        x0[j] = cache.x_lookup[idx]

    end

    return x0
end


# -------------------------------------------------
# Unified interface
# -------------------------------------------------

function sample_initial_state(method::Symbol,L,j0;
                              cache=nothing,
                              aexc=1,
                              rng=Random.default_rng())

    if method == :gaussian
        return sample_initial_state_gaussian(L,j0;aexc=aexc,rng=rng)

    elseif method == :discrete
        return sample_initial_state_discrete(L,j0,cache;rng=rng)

    else
        error("Unknown sampling method")
    end
end



function sample_initial_state_domainwall_gaussian(L; rng=Random.default_rng(), a_left=1, a_right=0)
    x0 = Vector{Matrix{ComplexF64}}(undef, L)
    half = L ÷ 2
    for j in 1:L
        a0 = (j <= half) ? a_left : a_right
        x0[j] = sample_site_gaussian(a0; rng=rng)
    end
    return x0
end

function sample_initial_state_domainwall_discrete(L, cache; rng=Random.default_rng(), a_left=1, a_right=0)
    # Your cache currently has W0 and W1 only. Domain wall needs 0 and 1, so OK.
    @assert a_left in (0,1) && a_right in (0,1) "cache currently supports only a=0,1 (W0/W1)"
    x0 = Vector{Matrix{ComplexF64}}(undef, L)
    half = L ÷ 2
    for j in 1:L
        a0 = (j <= half) ? a_left : a_right
        W = (a0 == 1) ? cache.W1 : cache.W0
        idx = sample_index(W, rng)
        x0[j] = cache.x_lookup[idx]
    end
    return x0
end

function sample_initial_state_domainwall(method::Symbol, L;
                                         cache=nothing,
                                         rng=Random.default_rng(),
                                         a_left::Int=1,
                                         a_right::Int=0)
    if method == :gaussian
        return sample_initial_state_domainwall_gaussian(L; rng=rng, a_left=a_left, a_right=a_right)
    elseif method == :discrete
        return sample_initial_state_domainwall_discrete(L, cache; rng=rng, a_left=a_left, a_right=a_right)
    else
        error("Unknown sampling method")
    end
end

end