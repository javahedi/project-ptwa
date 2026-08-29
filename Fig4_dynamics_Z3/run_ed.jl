using LinearAlgebra
using KrylovKit
using JLD2

const L = 13
const j0 = 7

const J = 1.0
const g = 0.5

const alphas =
    [3.0,1.5,0.5]

const tmax = 5.0
const Nt = 101

const d = 3^L
const ω = cis(2π/3)

const OUTDIR =
    joinpath(@__DIR__,"data","ed")

function basis_digits()

    digits =
        Matrix{UInt8}(
            undef,
            L,
            d
        )

    for s in 0:d-1

        q = s

        for j in 1:L
            digits[j,s+1] =
                UInt8(q % 3)

            q ÷= 3
        end
    end

    digits
end

function shift_tables(digits)

    pow3 =
        [3^(j-1) for j in 1:L]

    xp =
        Matrix{Int32}(undef,L,d)

    xm =
        Matrix{Int32}(undef,L,d)

    for s in 0:d-1

        for j in 1:L

            a =
                Int(digits[j,s+1])

            ap =
                mod(a+1,3)

            am =
                mod(a-1,3)

            xp[j,s+1] =
                Int32(
                    s +
                    (ap-a)*pow3[j] +
                    1
                )

            xm[j,s+1] =
                Int32(
                    s +
                    (am-a)*pow3[j] +
                    1
                )
        end
    end

    xp,xm
end

function interaction_diagonal(digits,alpha)

    diag =
        zeros(Float64,d)

    Jij =
        [
            i == j ?
            0.0 :
            J / abs(i-j)^alpha
            for i in 1:L,
                j in 1:L
        ]

    for s in 1:d

        E = 0.0

        for i in 1:L-1,
            j in i+1:L

            ai =
                Int(digits[i,s])

            aj =
                Int(digits[j,s])

            E +=
                -2 *
                Jij[i,j] *
                real(
                    ω^(ai-aj)
                )
        end

        diag[s] = E
    end

    diag
end

struct ClockLinearMap

    diag::Vector{Float64}

    xp::Matrix{Int32}

    xm::Matrix{Int32}
end

function (H::ClockLinearMap)(v)

    y =
        ComplexF64.(H.diag) .* v

    for s in 1:d

        vs = v[s]

        for j in 1:L

            y[Int(H.xp[j,s])] +=
                -g * vs

            y[Int(H.xm[j,s])] +=
                -g * vs
        end
    end

    y
end

function initial_state()

    s0 =
        3^(j0-1)

    ψ0 =
        zeros(ComplexF64,d)

    ψ0[s0+1] = 1.0

    ψ0
end

function measure_observables(
    ψ,
    digits
)

    prob =
        abs2.(ψ)

    Pexc =
        zeros(Float64,L)

    for j in 1:L

        p = 0.0

        for s in 1:d

            digits[j,s] != 0 &&
                (p += prob[s])
        end

        Pexc[j] = p
    end

    center_exc =
        Pexc[j0]

    avg_disp =
        sum(
            abs(j-j0) *
            Pexc[j]
            for j in 1:L
        )

    center_exc,avg_disp
end

mkpath(OUTDIR)

digits =
    basis_digits()

xp,xm =
    shift_tables(digits)

ψ0 =
    initial_state()

times =
    collect(
        range(
            0.0,
            tmax,
            length=Nt
        )
    )

for alpha in alphas

    println()
    println(
        "ED alpha=$alpha, dimension=$d"
    )

    H =
        ClockLinearMap(
            interaction_diagonal(
                digits,
                alpha
            ),
            xp,
            xm
        )

    ψ =
        copy(ψ0)

    center_exc =
        zeros(Float64,Nt)

    avg_disp =
        zeros(Float64,Nt)

    normψ =
        zeros(Float64,Nt)

    center_exc[1],
    avg_disp[1] =
        measure_observables(
            ψ,
            digits
        )

    normψ[1] =
        norm(ψ)

    for it in 2:Nt

        dt =
            times[it] -
            times[it-1]

        ψ,info =
            exponentiate(
                H,
                -1im*dt,
                ψ
            )

        center_exc[it],
        avg_disp[it] =
            measure_observables(
                ψ,
                digits
            )

        normψ[it] =
            norm(ψ)

        println(
            "t=$(times[it])  norm=$(normψ[it])"
        )
    end

    outfile =
        joinpath(
            OUTDIR,
            "ed_Z3_L$(L)_alpha$(alpha).jld2"
        )

    @save outfile L j0 alpha J g times center_exc avg_disp normψ

    println("saved -> $outfile")
end
