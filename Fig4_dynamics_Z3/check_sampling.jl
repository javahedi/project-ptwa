using Random
using LinearAlgebra

include(joinpath(@__DIR__,"sampling.jl"))
using .Sampling

L = 13
j0 = 7
Ntest = 5000

function check(method)

    rng =
        MersenneTwister(1)

    cache =
        method == :discrete ?
        build_discrete_cache() :
        nothing

    center11 =
        0.0

    other00 =
        0.0

    offdiag =
        0.0 + 0.0im

    maxherm =
        0.0

    for _ in 1:Ntest

        x0 =
            sample_initial_state(
                method,
                L,
                j0;
                cache=cache,
                aexc=1,
                rng=rng
            )

        center11 +=
            real(x0[j0][2,2])

        other00 +=
            real(x0[1][1,1])

        offdiag +=
            x0[j0][1,2]

        maxherm =
            max(
                maxherm,
                maximum(
                    abs.(
                        x0[j0] -
                        x0[j0]'
                    )
                )
            )
    end

    println()
    println("method = $method")
    println(
        "<x_center^(11)> = ",
        center11/Ntest
    )
    println(
        "<x_other^(00)>  = ",
        other00/Ntest
    )
    println(
        "<x_center^(01)> = ",
        offdiag/Ntest
    )
    println(
        "max Hermiticity error = ",
        maxherm
    )
end

check(:gaussian)
check(:discrete)
