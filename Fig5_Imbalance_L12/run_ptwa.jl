using LinearAlgebra, Random, JLD2, Base.Threads
include(joinpath(@__DIR__,"sampling.jl")); using .Sampling

const L=12; const J=1.0; const gpair=0.3
const W_list=[0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
const Ndis=100; const Nmc=100; const tmax=100.0; const Nt=1001
const times=collect(range(0.0,tmax,length=Nt)); const samplings=[:gaussian,:discrete]
const master_seed=1234; const OUTDIR=joinpath(@__DIR__,"data","ptwa")

const ω=cis(2π/3)
const B=ComplexF64[0 1 0; 0 0 1; 0 0 0]
const Bd=adjoint(B); const B2=B*B; const Bd2=Bd*Bd
const U=Diagonal(ComplexF64[1,ω,ω^2]); const U2=U*U
const Nop=Diagonal(ComplexF64[0,1,2])
const A1=Bd*Matrix(U); const C1=adjoint(A1)
const A2=Bd2*Matrix(U2); const C2=adjoint(A2)

@inline function trAB(A,Bm)
    s=0.0+0im
    @inbounds for i in 1:3, k in 1:3; s+=A[i,k]*Bm[k,i]; end
    s
end

function sample_domainwall(method,cache,rng)
    x0=Vector{Matrix{ComplexF64}}(undef,L)
    for j in 1:L
        a0=j<=L÷2 ? 1 : 0
        x0[j]=method==:gaussian ? sample_site_gaussian(a0;rng=rng) : sample_site_discrete(a0,cache;rng=rng)
    end
    x0
end

struct Params
    mu::Vector{Float64}
end

function build_h!(h,x,p)
    t1=-J*(1-gpair); t2=-J*gpair
    mB=Vector{ComplexF64}(undef,L); mBd=similar(mB); mA1=similar(mB); mC1=similar(mB)
    mB2=similar(mB); mBd2=similar(mB); mA2=similar(mB); mC2=similar(mB)
    @inbounds for j in 1:L
        fill!(h[j],0); h[j].+=p.mu[j].*Matrix(Nop)
        mB[j]=trAB(B,x[j]); mBd[j]=trAB(Bd,x[j]); mA1[j]=trAB(A1,x[j]); mC1[j]=trAB(C1,x[j])
        mB2[j]=trAB(B2,x[j]); mBd2[j]=trAB(Bd2,x[j]); mA2[j]=trAB(A2,x[j]); mC2[j]=trAB(C2,x[j])
    end
    @inbounds for j in 1:L-1
        jp=j+1
        h[j]  .+= t1.*(mB[jp].*A1 + mBd[jp].*C1)
        h[jp] .+= t1.*(mA1[j].*B + mC1[j].*Bd)
        h[j]  .+= t2.*(mB2[jp].*A2 + mBd2[jp].*C2)
        h[jp] .+= t2.*(mA2[j].*B2 + mC2[j].*Bd2)
    end
    @inbounds for j in 1:L; h[j].=0.5.*(h[j].+h[j]'); end
    nothing
end

function rhs!(dx,x,h,p)
    build_h!(h,x,p)
    @inbounds for j in 1:L; dx[j].=1im.*(x[j]*h[j]-h[j]*x[j]); end
    nothing
end

function evolve(x0,p)
    x=[copy(A) for A in x0]
    h=[zeros(ComplexF64,3,3) for _ in 1:L]; k1=deepcopy(h); k2=deepcopy(h); k3=deepcopy(h); k4=deepcopy(h); xt=deepcopy(h)
    I=zeros(Float64,Nt)
    function measure!(it)
        NL=0.0; NR=0.0
        @inbounds for j in 1:L
            nj=real(trAB(Matrix(Nop),x[j]))
            if j<=L÷2; NL+=nj; else; NR+=nj; end
        end
        I[it]=2*(NL-NR)/L
    end
    measure!(1)
    for it in 2:Nt
        dt=times[it]-times[it-1]
        rhs!(k1,x,h,p)
        @inbounds for j in 1:L; @. xt[j]=x[j]+(dt/2)*k1[j]; end; rhs!(k2,xt,h,p)
        @inbounds for j in 1:L; @. xt[j]=x[j]+(dt/2)*k2[j]; end; rhs!(k3,xt,h,p)
        @inbounds for j in 1:L; @. xt[j]=x[j]+dt*k3[j]; end; rhs!(k4,xt,h,p)
        @inbounds for j in 1:L
            @. x[j]+=(dt/6)*(k1[j]+2k2[j]+2k3[j]+k4[j])
            x[j].=0.5.*(x[j].+x[j]')
        end
        measure!(it); isfinite(I[it]) || error("non-finite imbalance at t=$(times[it])")
    end
    I
end

function run_case(W,sampling_method,iw)
    println("\npTWA W=$W sampling=$sampling_method")
    mkpath(OUTDIR); cache=sampling_method==:discrete ? build_discrete_cache() : nothing
    nt=Threads.nthreads(:default)
    S=[zeros(Float64,Nt) for _ in 1:nt]; S2=[zeros(Float64,Nt) for _ in 1:nt]
    Threads.@threads :static for worker in 1:nt
        s=S[worker]; s2=S2[worker]
        for r in worker:nt:Ndis
            drng=MersenneTwister(master_seed+1_000_000*iw+r)
            mu=(2W).*rand(drng,L).-W; p=Params(mu); Imc=zeros(Float64,Nt)
            for tr in 1:Nmc
                off=sampling_method==:gaussian ? 10_000_000 : 20_000_000
                rng=MersenneTwister(master_seed+off+1_000_000*iw+10_000*r+tr)
                Imc .+= evolve(sample_domainwall(sampling_method,cache,rng),p)
            end
            Ir=Imc./Nmc; s.+=Ir; s2.+=Ir.^2
            r%10==0 && println("W=$W $sampling_method disorder $r/$Ndis")
        end
    end
    I_sum=reduce(+,S); I_sum2=reduce(+,S2); I_mean=I_sum./Ndis
    I_var=max.(I_sum2./Ndis.-I_mean.^2,0); I_err=sqrt.(I_var./Ndis); sampling=sampling_method
    outfile=joinpath(OUTDIR,"pTWA_Z3_parafermion_L$(L)_g$(gpair)_W$(W)_Ndis$(Ndis)_Nmc$(Nmc)_$(sampling).jld2")
    @save outfile L J gpair W Ndis Nmc times sampling I_mean I_err master_seed
    println("saved -> $outfile")
end

for (iw,W) in enumerate(W_list), sampler in samplings; run_case(W,sampler,iw); end
