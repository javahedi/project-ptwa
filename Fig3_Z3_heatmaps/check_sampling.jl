using Random, LinearAlgebra
include(joinpath(@__DIR__,"sampling.jl")); using .Sampling
L=13; j0=7; Ntest=5000
for method in (:gaussian,:discrete)
    rng=MersenneTwister(1); cache=method==:discrete ? build_discrete_cache() : nothing
    c11=0.0; o00=0.0; c01=0.0+0im; herr=0.0
    for _ in 1:Ntest
        x0=sample_initial_state(method,L,j0;cache=cache,aexc=1,rng=rng)
        c11+=real(x0[j0][2,2]); o00+=real(x0[1][1,1]); c01+=x0[j0][1,2]
        herr=max(herr,maximum(abs.(x0[j0]-x0[j0]')))
    end
    println("\nmethod=$method")
    println("<x_center11> = ",c11/Ntest," target 1")
    println("<x_other00>  = ",o00/Ntest," target 1")
    println("<x_center01> = ",c01/Ntest," target 0")
    println("max Hermiticity error = ",herr)
end
