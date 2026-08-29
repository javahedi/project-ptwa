using LinearAlgebra, Random, JLD2, Base.Threads
include(joinpath(@__DIR__,"sampling.jl")); using .Sampling

# const L=13, j0=7, J=1.0, g=0.5, tmax=5.0, Nt=101, Ntraj=10_000, seed=1234
# const alphas=[3.0,1.5,0.5]; const sampling=:discrete; const ω=cis(2π/3)

const L = 13
const j0 = 7
const J = 1.0
const g = 0.5
const tmax = 5.0
const Nt = 101
const Ntraj=10_000
const alphas = [3.0, 1.5, 0.5]
const SAMPLING = :discrete
const ω = cis(2π/3)
const d = 3^L
const seed=1234

const OUTDIR=joinpath(@__DIR__,"data","ptwa")
struct PTWAParams; L::Int; alpha::Float64; J::Float64; g::Float64; Jij::Matrix{Float64}; end
function build_Jij(L,a,J)
    M=zeros(L,L); for i in 1:L, j in 1:L; i!=j && (M[i,j]=J/abs(i-j)^a); end; M
end
Z_of(x)=x[1,1]+ω*x[2,2]+ω^2*x[3,3]
function build_h!(h,x,p)
    Z=[Z_of(x[k]) for k in 1:p.L]
    for j in 1:p.L
        fill!(h[j],0)
        for a in 0:2
            s=0.0; wa=ω^a
            for k in 1:p.L
                k==j && continue; s += p.Jij[j,k]*real(wa*conj(Z[k]))
            end
            h[j][a+1,a+1]=-2s
        end
        h[j][2,1]-=p.g; h[j][3,2]-=p.g; h[j][1,3]-=p.g
        h[j][1,2]-=p.g; h[j][2,3]-=p.g; h[j][3,1]-=p.g
    end
end
function rhs!(dx,x,h,p)
    build_h!(h,x,p)
    for j in 1:p.L; dx[j].=1im.*(x[j]*h[j]-h[j]*x[j]); end
end
function evolve_and_measure(x0,p,times)
    x=[copy(A) for A in x0]; h=[zeros(ComplexF64,3,3) for _ in 1:p.L]
    k1=deepcopy(h); k2=deepcopy(h); k3=deepcopy(h); k4=deepcopy(h); xt=deepcopy(h)
    P=zeros(Float64,p.L,length(times))
    measure!(k)=(@inbounds for j in 1:p.L; P[j,k]=real(x[j][2,2]+x[j][3,3]); end)
    measure!(1)
    for it in 2:length(times)
        dt=times[it]-times[it-1]; rhs!(k1,x,h,p)
        for j in 1:p.L; @. xt[j]=x[j]+(dt/2)*k1[j]; end; rhs!(k2,xt,h,p)
        for j in 1:p.L; @. xt[j]=x[j]+(dt/2)*k2[j]; end; rhs!(k3,xt,h,p)
        for j in 1:p.L; @. xt[j]=x[j]+dt*k3[j]; end; rhs!(k4,xt,h,p)
        for j in 1:p.L
            @. x[j]+=(dt/6)*(k1[j]+2k2[j]+2k3[j]+k4[j]); x[j].=0.5.*(x[j].+x[j]')
        end
        measure!(it)
    end
    P
end


function run_alpha(alpha)

    mkpath(OUTDIR)

    times = collect(range(0.0, tmax, length=Nt))

    p = PTWAParams(
        L,
        alpha,
        J,
        g,
        build_Jij(L, alpha, J)
    )

    sampling_method = SAMPLING

    cache = sampling_method == :discrete ?
            build_discrete_cache() : nothing

    # Number of workers used for our own accumulation
    nt = Threads.nthreads(:default)

    println("Using $nt default Julia threads")

    sums  = [zeros(Float64, L, Nt) for _ in 1:nt]
    sums2 = [zeros(Float64, L, Nt) for _ in 1:nt]

    # ---------------------------------------------------------
    # Important:
    # worker = 1:nt is OUR index.
    # Never use threadid() as an array index.
    # ---------------------------------------------------------

    Threads.@threads :static for worker in 1:nt

        local_sum  = sums[worker]
        local_sum2 = sums2[worker]

        # worker 1: 1, 1+nt, 1+2nt, ...
        # worker 2: 2, 2+nt, ...
        for tr in worker:nt:Ntraj

            rng = MersenneTwister(
                seed +
                round(Int, 1000 * alpha) +
                tr
            )

            x0 = sample_initial_state(
                sampling_method,
                L,
                j0;
                cache = cache,
                aexc = 1,
                rng = rng
            )

            P = evolve_and_measure(
                x0,
                p,
                times
            )

            local_sum  .+= P
            local_sum2 .+= P.^2

            if tr % 500 == 0
                println(
                    "alpha = $alpha   trajectory $tr / $Ntraj"
                )
            end
        end
    end

    # ---------------------------------------------------------
    # Ensemble statistics
    # ---------------------------------------------------------

    Psum  = reduce(+, sums)
    Psum2 = reduce(+, sums2)

    Pexc = Psum ./ Ntraj

    Pexc_var = max.(
        Psum2 ./ Ntraj .- Pexc.^2,
        0.0
    )

    Pexc_err = sqrt.(
        Pexc_var ./ Ntraj
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    outfile = joinpath(
        OUTDIR,
        "ptwa_Z3_L$(L)_alpha$(alpha)_Ntraj$(Ntraj)_$(sampling_method).jld2"
    )

    sampling = sampling_method

    @save outfile L j0 alpha J g times sampling Ntraj seed Pexc Pexc_err

    println("saved -> $outfile")
end

for a in alphas
    println("\npTWA alpha=$a")
    run_alpha(a)
end