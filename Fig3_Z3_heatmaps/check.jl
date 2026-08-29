using JLD2, Printf
L=13; j0=7; alphas=[3.0,1.5,0.5]; Ntraj=10_000; sampling=:discrete
for a in alphas
    ed=load(joinpath(@__DIR__,"data","ed","ed_Z3_L$(L)_alpha$(a).jld2"))
    pt=load(joinpath(@__DIR__,"data","ptwa","ptwa_Z3_L$(L)_alpha$(a)_Ntraj$(Ntraj)_$(sampling).jld2"))
    @assert ed["times"]==pt["times"]; @assert maximum(abs.(ed["normψ"].-1))<1e-8
    exact0=zeros(L); exact0[j0]=1; @assert maximum(abs.(ed["Pexc"][:,1].-exact0))<1e-12
    initerr=maximum(abs.(pt["Pexc"][:,1].-exact0)); Δ=maximum(abs.(ed["Pexc"].-pt["Pexc"]))
    @printf("alpha=%3.1f  init err=%.3e  max|ED-pTWA|=%.6e  norm drift=%.3e\n",a,initerr,Δ,maximum(abs.(ed["normψ"].-1)))
end
println("All checks passed.")
