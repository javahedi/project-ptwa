using LinearAlgebra, Random
include(joinpath(@__DIR__,"sampling_Zn.jl")); using .Sampling
for n in 3:7
    rng=MersenneTwister(100+n)
    for a0 in (0,n-1)
        avg=zeros(ComplexF64,n,n); mh=0.0
        for _ in 1:5000
            x=sample_site_gaussian(n,a0; rng=rng); avg .+= x; mh=max(mh,maximum(abs.(x-x')))
        end
        avg ./= 5000
        println("n=$n |$a0>: diag mean = ",round.(real.(diag(avg));digits=4),", max Hermiticity error = ",mh)
    end
end
