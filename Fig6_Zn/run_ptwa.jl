using LinearAlgebra, Random, FFTW, JLD2, Base.Threads
include(joinpath(@__DIR__,"sampling_Zn.jl")); using .Sampling
const L=24
const J=1.0 
const dt=0.05
const  tmax=10.0
const  Ntraj=2000
const  seed=1234
const times=collect(0.0:dt:tmax); const Nt=length(times); const ns=collect(3:7)
const OUTDIR=joinpath(@__DIR__,"data","ptwa")
struct PTWAParams; L::Int; n::Int; J::Float64; end
function zn_ops(n)
    ω=cis(2π/n); U=Diagonal(ComplexF64[ω^a for a in 0:n-1]) |> Matrix
    B=zeros(ComplexF64,n,n); for a in 2:n; B[a-1,a]=1; end
    B,Matrix(adjoint(B)),U
end
@inline function trAB(A,B)
    s=0.0+0im
    @inbounds for i in axes(A,1), k in axes(A,2); s += A[i,k]*B[k,i]; end
    s
end
function build_h!(h,x,p,B,Bd,U)
    C=Bd*U; Cd=Matrix(adjoint(C)); b=Vector{ComplexF64}(undef,p.L); bd=similar(b); c=similar(b); cd=similar(b)
    @inbounds for j in 1:p.L
        b[j]=trAB(B,x[j]); bd[j]=trAB(Bd,x[j]); c[j]=trAB(C,x[j]); cd[j]=trAB(Cd,x[j]); fill!(h[j],0)
    end
    @inbounds for j in 1:p.L-1
        jp=j+1
        h[j]  .+= -p.J .* (b[jp].*C + bd[jp].*Cd)
        h[jp] .+= -p.J .* (c[j].*B + cd[j].*Bd)
    end
    @inbounds for j in 1:p.L; h[j] .= 0.5 .* (h[j] .+ h[j]'); end
end
function rhs!(dx,x,h,p,B,Bd,U)
    build_h!(h,x,p,B,Bd,U)
    @inbounds for j in 1:p.L; dx[j] .= 1im .* (x[j]*h[j]-h[j]*x[j]); end
end
function evolve_and_measure(x0,p,B,Bd,U)
    x=[copy(A) for A in x0]; h=[zeros(ComplexF64,p.n,p.n) for _ in 1:p.L]
    k1=deepcopy(h); k2=deepcopy(h); k3=deepcopy(h); k4=deepcopy(h); xt=deepcopy(h)
    P0=zeros(Float64,p.L,Nt); I=zeros(Float64,Nt); half=p.L÷2
    function measure!(it)
        left=0.0; right=0.0
        @inbounds for j in 1:p.L
            v=real(x[j][1,1]); P0[j,it]=v
            j<=half ? (left+=v) : (right+=v)
        end
        I[it]=(2/p.L)*(left-right)
    end
    measure!(1)
    for it in 2:Nt
        rhs!(k1,x,h,p,B,Bd,U)
        @inbounds for j in 1:p.L; @. xt[j]=x[j]+(dt/2)*k1[j]; end; rhs!(k2,xt,h,p,B,Bd,U)
        @inbounds for j in 1:p.L; @. xt[j]=x[j]+(dt/2)*k2[j]; end; rhs!(k3,xt,h,p,B,Bd,U)
        @inbounds for j in 1:p.L; @. xt[j]=x[j]+dt*k3[j]; end; rhs!(k4,xt,h,p,B,Bd,U)
        @inbounds for j in 1:p.L
            @. x[j]+=(dt/6)*(k1[j]+2k2[j]+2k3[j]+k4[j]); x[j].=0.5.*(x[j].+x[j]')
        end
        measure!(it)
    end
    P0,I
end
function run_n(n)
    println("pTWA n=$n"); mkpath(OUTDIR); B,Bd,U=zn_ops(n); p=PTWAParams(L,n,J); nt=Threads.nthreads(:default)
    Ps=[zeros(Float64,L,Nt) for _ in 1:nt]; Is=[zeros(Float64,Nt) for _ in 1:nt]; I2s=[zeros(Float64,Nt) for _ in 1:nt]
    Threads.@threads :static for worker in 1:nt
        ps=Ps[worker]; is=Is[worker]; i2=I2s[worker]
        for tr in worker:nt:Ntraj
            rng=MersenneTwister(seed+1_000_000*n+tr)
            x0=sample_initial_state_domainwall(n,L;a_left=0,a_right=n-1,rng=rng)
            P0,I=evolve_and_measure(x0,p,B,Bd,U); ps .+= P0; is .+= I; i2 .+= I.^2
            tr%200==0 && println("n=$n trajectory $tr / $Ntraj")
        end
    end
    P0_avg=reduce(+,Ps)./Ntraj; I_avg=reduce(+,Is)./Ntraj; I2_avg=reduce(+,I2s)./Ntraj
    I_err=sqrt.(max.(I2_avg.-I_avg.^2,0)./Ntraj)
    qs=2π.*(0:L-1)./L; Sa_avg=zeros(Float64,L,Nt)
    for it in 1:Nt; Sa_avg[:,it].=abs2.(fft(view(P0_avg,:,it)))./L; end
    sampling=:gaussian; a_left=0; a_right=n-1
    outfile=joinpath(OUTDIR,"pTWA_Zn_domainwall_singlehop_n$(n)_L$(L)_gaussian_N$(Ntraj).jld2")
    @save outfile n L J dt tmax times qs Ntraj seed sampling a_left a_right P0_avg Sa_avg I_avg I_err
    println("saved -> $outfile")
end
for n in ns; run_n(n); end
