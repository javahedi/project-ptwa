using LinearAlgebra
using Random
using Statistics
using JLD2
using Base.Threads

include(joinpath(@__DIR__,"sampling_Zn.jl"))
using .Sampling

const J = 1.0
const g = 0.5
const tmax = 5.0
const Nt = 101
const Ntraj = 2000
const seed = 1234
const N_top = 30
const ns_top = collect(3:7)
const n_bottom = 5
const Ns_bottom = collect(10:10:80)
const OUTDIR = joinpath(@__DIR__,"data","ptwa")

struct PTWAParams
    N::Int; n::Int; J::Float64; g::Float64; ω::ComplexF64
end

@inline function Z_of(xj,ω,n)
    sum(ω^a*xj[a+1,a+1] for a in 0:n-1)
end

function build_h!(h,x,p)
    z = [Z_of(x[j],p.ω,p.n) for j in 1:p.N]
    zsum = sum(z)
    for j in 1:p.N
        fill!(h[j],0)
        zex = zsum-z[j]
        for a in 0:p.n-1
            wa = p.ω^a
            h[j][a+1,a+1] = -(p.J/p.N)*(wa*conj(zex)+conj(wa)*zex)
        end
        for a in 0:p.n-1
            b = mod(a+1,p.n)
            h[j][b+1,a+1] += -p.g
            h[j][a+1,b+1] += -p.g
        end
    end
end

function rhs!(dx,x,h,p)
    build_h!(h,x,p)
    for j in 1:p.N
        dx[j] .= 1im .* (x[j]*h[j] - h[j]*x[j])
    end
end

function evolve_and_measure(x0,p,times)
    N,n = p.N,p.n
    x = [copy(A) for A in x0]
    h  = [zeros(ComplexF64,n,n) for _ in 1:N]
    k1 = [zeros(ComplexF64,n,n) for _ in 1:N]
    k2 = [zeros(ComplexF64,n,n) for _ in 1:N]
    k3 = [zeros(ComplexF64,n,n) for _ in 1:N]
    k4 = [zeros(ComplexF64,n,n) for _ in 1:N]
    xt = [zeros(ComplexF64,n,n) for _ in 1:N]
    mZ = zeros(ComplexF64,length(times))

    mZ[1] = sum(Z_of(x[j],p.ω,n) for j in 1:N)/N

    for k in 2:length(times)
        dt = times[k]-times[k-1]
        rhs!(k1,x,h,p)

        for j in 1:N
            @. xt[j] = x[j] + (dt/2)*k1[j]
        end
        rhs!(k2,xt,h,p)

        for j in 1:N
            @. xt[j] = x[j] + (dt/2)*k2[j]
        end
        rhs!(k3,xt,h,p)

        for j in 1:N
            @. xt[j] = x[j] + dt*k3[j]
        end
        rhs!(k4,xt,h,p)

        for j in 1:N
            @. x[j] += (dt/6)*(k1[j]+2k2[j]+2k3[j]+k4[j])
            x[j] .= 0.5 .* (x[j] .+ x[j]')
        end
        mZ[k] = sum(Z_of(x[j],p.ω,n) for j in 1:N)/N
    end
    mZ
end

function run_case(N,n)
    mkpath(OUTDIR)
    times = collect(range(0.0,tmax,length=Nt))
    p = PTWAParams(N,n,J,g,cis(2π/n))
    mZ_traj = Matrix{ComplexF64}(undef,Ntraj,Nt)

    @threads for tr in 1:Ntraj
        rng = MersenneTwister(seed+tr)
        x0 = sample_initial_state_fully_polarized(n,N;a0=0,rng=rng)
        mZ_traj[tr,:] .= evolve_and_measure(x0,p,times)
    end

    mZ_avg = vec(mean(mZ_traj,dims=1))
    abs_mZ = abs.(mZ_avg)
    mean_abs_mZ_traj = vec(mean(abs.(mZ_traj),dims=1))

    outfile = joinpath(OUTDIR,"ptwa_LMG_N$(N)_n$(n)_Ntraj$(Ntraj).jld2")
    @save outfile N n J g Ntraj seed times mZ_avg abs_mZ mean_abs_mZ_traj mZ_traj
    println("saved -> $outfile")
end

cases = Set{Tuple{Int,Int}}()
foreach(n -> push!(cases,(N_top,n)), ns_top)
foreach(N -> push!(cases,(N,n_bottom)), Ns_bottom)
for (N,n) in sort!(collect(cases))
    println("pTWA: N=$N n=$n")
    run_case(N,n)
end
