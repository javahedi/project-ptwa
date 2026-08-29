using JLD2, Statistics, Printf
const L=12; const g=0.3; const W_list=[0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
const seed_list=0:8; const Ndis=100; const Nmc=100; const chi=100; const dt=0.1
for (seed,W) in zip(seed_list,W_list)
    ed=load(joinpath(@__DIR__,"data","ed","ED_Z3_L$(L)_g$(g)_W$(W)_Ndis$(Ndis)_seed$(seed)_chi$(chi)_dt$(dt).jld2"))
    Ied=ed["imbalance_avg"]; t=ed["times"]
    @assert abs(Ied[1]-1)<1e-10
    @assert all(isfinite,Ied)
    println("\nW=$W")
    for sampler in (:gaussian,:discrete)
        pt=load(joinpath(@__DIR__,"data","ptwa","pTWA_Z3_parafermion_L$(L)_g$(g)_W$(W)_Ndis$(Ndis)_Nmc$(Nmc)_$(sampler).jld2"))
        @assert pt["times"]==t
        @assert all(isfinite,pt["I_mean"])
        mask=(t.>=2).&(t.<=20)
        a=mean(pt["I_mean"][mask]); b=mean(Ied[mask])
        @printf("  %-8s I(0)=%.8f Ibar=%.6f EDbar=%.6f Δ=%+.3e\n",string(sampler),pt["I_mean"][1],a,b,a-b)
    end
end
println("\nAll structural checks passed.")
