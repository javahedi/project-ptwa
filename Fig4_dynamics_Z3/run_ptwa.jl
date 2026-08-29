using LinearAlgebra
using Random
using JLD2
using Base.Threads

include(joinpath(@__DIR__,"sampling.jl"))
using .Sampling

# ============================================================
# Published parameters
# ============================================================

const L = 13
const j0 = 7
const J = 1.0
const g = 0.5

const alphas = [3.0,1.5,0.5]
const samplings = [:gaussian,:discrete]

const tmax = 5.0
const Nt = 101

const Ntraj = 10_000
const seed = 1234

const ω = cis(2π/3)

const OUTDIR = joinpath(@__DIR__,"data","ptwa")

struct PTWAParams
    L::Int
    alpha::Float64
    J::Float64
    g::Float64
    Jij::Matrix{Float64}
end

function build_Jij(L,alpha,J)
    M = zeros(Float64,L,L)

    for i in 1:L, j in 1:L
        if i != j
            M[i,j] = J / abs(i-j)^alpha
        end
    end

    M
end

@inline Z_of(x) =
    x[1,1] + ω*x[2,2] + ω^2*x[3,3]

function build_h!(h,x,p)

    Z = [Z_of(x[k]) for k in 1:p.L]

    for j in 1:p.L

        fill!(h[j],0)

        for a in 0:2

            wa = ω^a
            s = 0.0

            for k in 1:p.L
                k == j && continue

                s += p.Jij[j,k] *
                     real(wa * conj(Z[k]))
            end

            h[j][a+1,a+1] = -2s
        end

        # -g (X + X†)
        h[j][2,1] -= p.g
        h[j][3,2] -= p.g
        h[j][1,3] -= p.g

        h[j][1,2] -= p.g
        h[j][2,3] -= p.g
        h[j][3,1] -= p.g
    end

    nothing
end

# Convention:
# (h_j)_{ba} = ∂H_W / ∂x_j^{ab}
# => dx_j/dt = i[x_j,h_j]

function rhs!(dx,x,h,p)

    build_h!(h,x,p)

    for j in 1:p.L
        dx[j] .= 1im .* (
            x[j]*h[j] -
            h[j]*x[j]
        )
    end

    nothing
end

function evolve_and_measure(x0,p,times)

    x = [copy(A) for A in x0]

    h  = [zeros(ComplexF64,3,3) for _ in 1:p.L]
    k1 = [zeros(ComplexF64,3,3) for _ in 1:p.L]
    k2 = [zeros(ComplexF64,3,3) for _ in 1:p.L]
    k3 = [zeros(ComplexF64,3,3) for _ in 1:p.L]
    k4 = [zeros(ComplexF64,3,3) for _ in 1:p.L]
    xt = [zeros(ComplexF64,3,3) for _ in 1:p.L]

    Ntloc = length(times)

    center_exc = zeros(Float64,Ntloc)
    avg_disp   = zeros(Float64,Ntloc)

    function measure!(it)

        ce = 0.0
        disp = 0.0

        for j in 1:p.L

            Pexc_j = real(
                x[j][2,2] +
                x[j][3,3]
            )

            if j == j0
                ce = Pexc_j
            end

            disp += abs(j-j0) * Pexc_j
        end

        center_exc[it] = ce
        avg_disp[it] = disp
    end

    measure!(1)

    for it in 2:Ntloc

        dt = times[it] - times[it-1]

        rhs!(k1,x,h,p)

        for j in 1:p.L
            @. xt[j] =
                x[j] + (dt/2)*k1[j]
        end

        rhs!(k2,xt,h,p)

        for j in 1:p.L
            @. xt[j] =
                x[j] + (dt/2)*k2[j]
        end

        rhs!(k3,xt,h,p)

        for j in 1:p.L
            @. xt[j] =
                x[j] + dt*k3[j]
        end

        rhs!(k4,xt,h,p)

        for j in 1:p.L

            @. x[j] +=
                (dt/6) *
                (
                    k1[j] +
                    2k2[j] +
                    2k3[j] +
                    k4[j]
                )

            # numerical roundoff cleanup only
            x[j] .= 0.5 .* (
                x[j] .+
                x[j]'
            )
        end

        measure!(it)
    end

    return center_exc, avg_disp
end

function run_case(alpha,sampling_method)

    println()
    println("------------------------------------------")
    println("alpha = $alpha")
    println("sampling = $sampling_method")
    println("------------------------------------------")

    mkpath(OUTDIR)

    times =
        collect(
            range(
                0.0,
                tmax,
                length=Nt
            )
        )

    p = PTWAParams(
        L,
        alpha,
        J,
        g,
        build_Jij(L,alpha,J)
    )

    cache =
        sampling_method == :discrete ?
        build_discrete_cache() :
        nothing

    nt = Threads.nthreads(:default)

    println("Using $nt default threads")

    center_sum =
        [zeros(Float64,Nt) for _ in 1:nt]

    center_sum2 =
        [zeros(Float64,Nt) for _ in 1:nt]

    disp_sum =
        [zeros(Float64,Nt) for _ in 1:nt]

    disp_sum2 =
        [zeros(Float64,Nt) for _ in 1:nt]

    Threads.@threads :static for worker in 1:nt

        c1 = center_sum[worker]
        c2 = center_sum2[worker]

        d1 = disp_sum[worker]
        d2 = disp_sum2[worker]

        for tr in worker:nt:Ntraj

            sampling_offset =
                sampling_method == :gaussian ?
                10_000_000 :
                20_000_000

            rng = MersenneTwister(
                seed +
                sampling_offset +
                round(Int,1000*alpha) +
                tr
            )

            x0 = sample_initial_state(
                sampling_method,
                L,
                j0;
                cache=cache,
                aexc=1,
                rng=rng
            )

            center,disp =
                evolve_and_measure(
                    x0,
                    p,
                    times
                )

            c1 .+= center
            c2 .+= center.^2

            d1 .+= disp
            d2 .+= disp.^2

            if tr % 500 == 0
                println(
                    "alpha=$alpha  sampling=$sampling_method  trajectory $tr / $Ntraj"
                )
            end
        end
    end

    center_total =
        reduce(+,center_sum)

    center_total2 =
        reduce(+,center_sum2)

    disp_total =
        reduce(+,disp_sum)

    disp_total2 =
        reduce(+,disp_sum2)

    center_exc =
        center_total ./ Ntraj

    avg_disp =
        disp_total ./ Ntraj

    center_var =
        max.(
            center_total2 ./ Ntraj -
            center_exc.^2,
            0.0
        )

    disp_var =
        max.(
            disp_total2 ./ Ntraj -
            avg_disp.^2,
            0.0
        )

    center_err =
        sqrt.(
            center_var ./ Ntraj
        )

    disp_err =
        sqrt.(
            disp_var ./ Ntraj
        )

    outfile =
        joinpath(
            OUTDIR,
            "ptwa_Z3_L$(L)_alpha$(alpha)_Ntraj$(Ntraj)_$(sampling_method).jld2"
        )

    sampling = sampling_method

    @save outfile L j0 alpha J g times sampling Ntraj seed center_exc center_err avg_disp disp_err

    println("saved -> $outfile")
end

for alpha in alphas
    for sampling_method in samplings
        run_case(alpha,sampling_method)
    end
end
