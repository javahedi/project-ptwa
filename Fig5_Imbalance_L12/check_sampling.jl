using LinearAlgebra, Random
include(joinpath(@__DIR__,"sampling.jl")); using .Sampling
N=5000; cache=build_discrete_cache()
for method in (:gaussian,:discrete), a0 in (0,1)
    avg=zeros(ComplexF64,3,3); maxh=0.0; rng=MersenneTwister(100+10a0+(method==:discrete))
    for _ in 1:N
        x=method==:gaussian ? sample_site_gaussian(a0;rng=rng) : sample_site_discrete(a0,cache;rng=rng)
        avg .+= x; maxh=max(maxh,maximum(abs.(x-x')))
    end
    avg./=N
    println("$method |$a0>: diag=",real.(diag(avg)),"  <x01>=",avg[1,2],"  herm=",maxh)
end
