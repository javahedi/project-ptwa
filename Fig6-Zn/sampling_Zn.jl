module Sampling
using LinearAlgebra, Random
export sample_site_gaussian, sample_initial_state_domainwall
function sample_site_gaussian(n::Int,a0::Int; rng=Random.default_rng())
    x=zeros(ComplexF64,n,n); μ=zeros(Float64,n); μ[a0+1]=1.0; x[a0+1,a0+1]=1
    for a in 1:n, b in a+1:n
        σ2=(μ[a]+μ[b])/4
        if σ2>0
            q=sqrt(σ2)*randn(rng); p=sqrt(σ2)*randn(rng)
            x[a,b]=q+1im*p; x[b,a]=q-1im*p
        end
    end
    x .= 0.5 .* (x .+ x')
    x
end
function sample_initial_state_domainwall(n::Int,L::Int; a_left::Int=0,a_right::Int=n-1,rng=Random.default_rng())
    x0=Vector{Matrix{ComplexF64}}(undef,L); half=L÷2
    for j in 1:L
        x0[j]=sample_site_gaussian(n,j<=half ? a_left : a_right; rng=rng)
    end
    x0
end
end
