using LinearAlgebra, SparseArrays, KrylovKit, JLD2

const L=13
const j0=7
const J=1.0
const g=0.5
const tmax=5.0
const Nt=101
const alphas=[3.0,1.5,0.5]; const ω=cis(2π/3); const d=3^L
const OUTDIR=joinpath(@__DIR__,"data","ed")

function basis_digits()
    D=Matrix{UInt8}(undef,L,d)
    for s in 0:d-1
        q=s
        for j in 1:L; D[j,s+1]=UInt8(q%3); q÷=3; end
    end
    D
end
function build_sparse_H(D,alpha)
    rows=Int[]; cols=Int[]; vals=ComplexF64[]; pow3=[3^(j-1) for j in 1:L]
    sizehint!(rows,d*(1+2L)); sizehint!(cols,d*(1+2L)); sizehint!(vals,d*(1+2L))
    for s in 0:d-1
        col=s+1; E=0.0
        for i in 1:L-1, j in i+1:L
            ai=Int(D[i,col]); aj=Int(D[j,col]); E += -2J/abs(i-j)^alpha*real(ω^(ai-aj))
        end
        push!(rows,col); push!(cols,col); push!(vals,E)
        for j in 1:L
            a=Int(D[j,col]); ap=mod(a+1,3); am=mod(a-1,3)
            push!(rows,s+(ap-a)*pow3[j]+1); push!(cols,col); push!(vals,-g)
            push!(rows,s+(am-a)*pow3[j]+1); push!(cols,col); push!(vals,-g)
        end
    end
    sparse(rows,cols,vals,d,d)
end
function excitation_profile!(out,ψ,D)
    prob=abs2.(ψ)
    for j in 1:L
        s=0.0; @inbounds for q in 1:d; D[j,q]!=0 && (s+=prob[q]); end; out[j]=s
    end
end
D=basis_digits(); times=collect(range(0.0,tmax,length=Nt)); s0=3^(j0-1); ψ0=zeros(ComplexF64,d); ψ0[s0+1]=1
for alpha in alphas
    println("ED alpha=$alpha, dim=$d"); H=build_sparse_H(D,alpha); ψ=copy(ψ0)
    Pexc=zeros(Float64,L,Nt); normψ=zeros(Float64,Nt); excitation_profile!(view(Pexc,:,1),ψ,D); normψ[1]=1
    for k in 2:Nt
        dt=times[k]-times[k-1]; ψ,=exponentiate(H,-1im*dt,ψ); excitation_profile!(view(Pexc,:,k),ψ,D); normψ[k]=norm(ψ)
        println("  t=$(times[k]) norm=$(normψ[k])")
    end
    outfile=joinpath(OUTDIR,"ed_Z3_L$(L)_alpha$(alpha).jld2"); @save outfile L j0 alpha J g times Pexc normψ; println("saved -> $outfile")
end
