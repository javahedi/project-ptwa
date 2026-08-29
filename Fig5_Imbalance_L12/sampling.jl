module Sampling
using LinearAlgebra, Random
export DiscreteCache, build_discrete_cache, sample_site_gaussian, sample_site_discrete
const ω = cis(2π/3)

function sample_site_gaussian(a0::Int; rng=Random.default_rng())
    x=zeros(ComplexF64,3,3); μ=zeros(3); μ[a0+1]=1; x[a0+1,a0+1]=1
    for a in 1:2, b in a+1:3
        σ2=(μ[a]+μ[b])/4
        if σ2>0
            q=sqrt(σ2)*randn(rng); p=sqrt(σ2)*randn(rng)
            x[a,b]=q+1im*p; x[b,a]=q-1im*p
        end
    end
    x
end

struct DiscreteCache
    W::NTuple{3,Vector{Float64}}
    x_lookup::Vector{Matrix{ComplexF64}}
end

function build_discrete_cache()
    Z=Diagonal(ComplexF64[1,ω,ω^2])
    X=ComplexF64[0 1 0; 0 0 1; 1 0 0]
    A=Matrix{ComplexF64}[]
    for q in 0:2, p in 0:2
        Aq=zeros(ComplexF64,3,3)
        for m in 0:2, k in 0:2
            Aq .+= ω^(p*k-q*m+2*m*k).*(Z^m*X^k)
        end
        Aq./=3; Aq.=0.5.*(Aq.+Aq'); push!(A,Aq)
    end
    function weights(a0)
        ρ=zeros(ComplexF64,3,3); ρ[a0+1,a0+1]=1
        W=[real(tr(ρ*Aq))/3 for Aq in A]
        minimum(W)<-1e-12 && error("negative discrete-Wigner weight")
        W=max.(W,0); W./=sum(W); W
    end
    xlookup=[Matrix{ComplexF64}(permutedims(Aq)) for Aq in A]
    DiscreteCache((weights(0),weights(1),weights(2)),xlookup)
end

@inline function sample_index(W,rng)
    r=rand(rng); s=0.0
    for i in eachindex(W)
        s+=W[i]; r<=s && return i
    end
    lastindex(W)
end

function sample_site_discrete(a0::Int,cache::DiscreteCache; rng=Random.default_rng())
    copy(cache.x_lookup[sample_index(cache.W[a0+1],rng)])
end
end
