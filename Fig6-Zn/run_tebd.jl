using LinearAlgebra, FFTW, JLD2
BLAS.set_num_threads(1)
const L=24
const J=1.0
const  dt=0.05
const  tmax=10.0
const  chiMax=80
const times=collect(0.0:dt:tmax); const Nt=length(times); const ns=collect(3:7)
const OUTDIR=joinpath(@__DIR__,"data","tebd")
function zn_ops(n)
    ω=cis(2π/n); U=Diagonal(ComplexF64[ω^a for a in 0:n-1]) |> Matrix
    B=zeros(ComplexF64,n,n); for a in 2:n; B[a-1,a]=1; end
    B,Matrix(adjoint(B)),U
end
function bond_gate(n)
    B,Bd,U=zn_ops(n); hop=kron(Bd*U,B); H=-J.*(hop+hop'); exp(-1im*dt*H)
end
function domain_wall_mps(n)
    G=Vector{Array{ComplexF64,3}}(undef,L); λ=Vector{Vector{ComplexF64}}(undef,L+1)
    for j in 1:L; G[j]=zeros(ComplexF64,n,1,1); λ[j]=ComplexF64[1]; end; λ[L+1]=ComplexF64[1]
    for j in 1:L; a0=j<=L÷2 ? 0 : n-1; G[j][a0+1,1,1]=1; end
    λ,G
end
@inline function safe_inv_vec(v; eps=1e-12)
    out=similar(v); @inbounds for i in eachindex(v); out[i]=abs(v[i])>eps ? inv(v[i]) : 0.0+0im; end; out
end
function build_bond_matrix(G1,G2,λL,λM,λR,gate)
    n=size(G1,1); χL=size(G1,2); χM=size(G1,3); χR=size(G2,3)
    M=zeros(ComplexF64,χL*n,n*χR); v=zeros(ComplexF64,n*n); w=similar(v)
    @inbounds for a in 1:χL, b in 1:χR
        fill!(v,0)
        for m in 1:χM
            fac=λL[a]*λM[m]*λR[b]
            for s in 1:n, t in 1:n; v[(s-1)*n+t]+=fac*G1[s,a,m]*G2[t,m,b]; end
        end
        mul!(w,gate,v)
        for s in 1:n, t in 1:n; M[(a-1)*n+s,(t-1)*χR+b]=w[(s-1)*n+t]; end
    end
    M
end
function update_bond!(λ,G,gate,i)
    G1,G2=G[i],G[i+1]; λL,λM,λR=λ[i],λ[i+1],λ[i+2]; n=size(G1,1); χL=length(λL); χR=length(λR)
    F=svd(build_bond_matrix(G1,G2,λL,λM,λR,gate);alg=LinearAlgebra.QRIteration()); keep=min(length(F.S),chiMax)
    U=F.U[:,1:keep]; S=F.S[1:keep]; Vh=F.Vt[1:keep,:]; norm(S)>0 && (S./=norm(S)); λ[i+1]=ComplexF64.(S)
    il=safe_inv_vec(λL); ir=safe_inv_vec(λR); Gnew1=zeros(ComplexF64,n,χL,keep); Gnew2=zeros(ComplexF64,n,keep,χR)
    @inbounds for a in 1:χL, s in 1:n, m in 1:keep; Gnew1[s,a,m]=il[a]*U[(a-1)*n+s,m]; end
    @inbounds for m in 1:keep, s in 1:n, b in 1:χR; Gnew2[s,m,b]=Vh[m,(s-1)*χR+b]*ir[b]; end
    G[i]=Gnew1; G[i+1]=Gnew2
end
function tebd_step!(λ,G,gate)
    for parity in 0:1
        for j in parity+1:2:L-1; update_bond!(λ,G,gate,j); end
    end
end
function local_P0(λ,G,pos)
    λL=λ[pos]; λR=λ[pos+1]; A=G[pos]; num=0.0; den=0.0
    @inbounds for a in eachindex(λL), s in 1:size(A,1), b in eachindex(λR)
        p=abs2(λL[a]*A[s,a,b]*λR[b]); den+=p; s==1 && (num+=p)
    end
    num/den
end
function run_n(n)
    println("TEBD n=$n"); mkpath(OUTDIR); gate=bond_gate(n); λ,G=domain_wall_mps(n); P0=zeros(Float64,L,Nt); I=zeros(Float64,Nt); half=L÷2
    function measure!(it)
        for j in 1:L; P0[j,it]=local_P0(λ,G,j); end
        I[it]=(2/L)*(sum(view(P0,1:half,it))-sum(view(P0,half+1:L,it)))
    end
    measure!(1)
    for it in 2:Nt; tebd_step!(λ,G,gate); measure!(it); (it-1)%20==0 && println("n=$n t=$(times[it])"); end
    qs=2π.*(0:L-1)./L; Sa=zeros(Float64,L,Nt)
    for it in 1:Nt; Sa[:,it].=abs2.(fft(view(P0,:,it)))./L; end
    outfile=joinpath(OUTDIR,"TEBD_Zn_domainwall_singlehop_n$(n)_L$(L)_chi$(chiMax)_dt$(dt).jld2")
    @save outfile n L J dt tmax times qs chiMax P0 Sa I; println("saved -> $outfile")
end
for n in ns; run_n(n); end
