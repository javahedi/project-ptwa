module Sampling
using LinearAlgebra, Random
export build_discrete_cache, sample_initial_state
const ω = cis(2π/3)

function sample_site_gaussian(a0::Int; rng=Random.default_rng())
    x=zeros(ComplexF64,3,3); μ=zeros(3); μ[a0+1]=1; x[a0+1,a0+1]=1
    for a in 1:2, b in a+1:3
        σ=sqrt((μ[a]+μ[b])/4)
        if σ>0
            q=σ*randn(rng); p=σ*randn(rng)
            x[a,b]=q+1im*p; x[b,a]=q-1im*p
        end
    end
    x
end

struct DiscreteCache
    A::Vector{Matrix{ComplexF64}}
    W::NTuple{3,Vector{Float64}}
    x_lookup::Vector{Matrix{ComplexF64}}
end

function build_Aqp()
    Z=Diagonal(ComplexF64[1,ω,ω^2]); X=ComplexF64[0 1 0;0 0 1;1 0 0]
    A=Matrix{ComplexF64}[]
    for q in 0:2, p in 0:2
        Aq=zeros(ComplexF64,3,3)
        for m in 0:2, k in 0:2
            Aq .+= ω^(p*k-q*m+2*m*k) .* (Z^m*X^k)
        end
        Aq ./= 3; Aq .= 0.5 .* (Aq .+ Aq')
        push!(A,Aq)
    end
    A
end

function build_wigner_prob(A,a)
    ρ=zeros(ComplexF64,3,3); ρ[a+1,a+1]=1
    W=[real(tr(ρ*Aq))/3 for Aq in A]
    minimum(W)<-1e-12 && error("negative Wigner weight")
    W=max.(W,0); W./=sum(W); W
end

phasepoint_to_x(Aq)=permutedims(Aq)

function build_discrete_cache()
    A=build_Aqp(); W=(build_wigner_prob(A,0),build_wigner_prob(A,1),build_wigner_prob(A,2))
    DiscreteCache(A,W,[phasepoint_to_x(Aq) for Aq in A])
end

function sample_index(W,rng)
    r=rand(rng); s=0.0
    for i in eachindex(W)
        s+=W[i]; r<=s && return i
    end
    lastindex(W)
end

function sample_initial_state(method::Symbol,L,j0; cache=nothing,aexc=1,rng=Random.default_rng())
    x0=Vector{Matrix{ComplexF64}}(undef,L)
    for j in 1:L
        a0=j==j0 ? aexc : 0
        if method==:gaussian
            x0[j]=sample_site_gaussian(a0;rng=rng)
        elseif method==:discrete
            idx=sample_index(cache.W[a0+1],rng); x0[j]=copy(cache.x_lookup[idx])
        else
            error("unknown sampling method")
        end
    end
    x0
end
end
