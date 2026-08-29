using JLD2, Printf
const L=24
const Ntraj=2000
const chi=80
const dt=0.05
for n in 3:7
    pt=load(joinpath(@__DIR__,"data","ptwa","pTWA_Zn_domainwall_singlehop_n$(n)_L$(L)_gaussian_N$(Ntraj).jld2"))
    te=load(joinpath(@__DIR__,"data","tebd","TEBD_Zn_domainwall_singlehop_n$(n)_L$(L)_chi$(chi)_dt$(dt).jld2"))
    @assert pt["times"]==te["times"]; @assert pt["qs"]==te["qs"]
    @assert abs(pt["I_avg"][1]-1)<1e-10; @assert abs(te["I"][1]-1)<1e-10
    @printf("n=%d  pTWA I(tf)=%.6f ± %.6f   TEBD I(tf)=%.6f   max|ΔI|=%.6f\n",n,pt["I_avg"][end],pt["I_err"][end],te["I"][end],maximum(abs.(pt["I_avg"]-te["I"])))
end
println("All Fig4 structural checks passed.")
