using JLD2
using Printf

N_top=30; ns_top=collect(3:7); n_bottom=5; Ns_bottom=collect(10:10:80); Ntraj=2000
ed_dir=joinpath(@__DIR__,"data","ed")
pt_dir=joinpath(@__DIR__,"data","ptwa")

cases=Set{Tuple{Int,Int}}()
foreach(n->push!(cases,(N_top,n)),ns_top)
foreach(N->push!(cases,(N,n_bottom)),Ns_bottom)

for (N,n) in sort!(collect(cases))
    ed=load(joinpath(ed_dir,"ed_LMG_N$(N)_n$(n).jld2"))
    pt=load(joinpath(pt_dir,"ptwa_LMG_N$(N)_n$(n)_Ntraj$(Ntraj).jld2"))
    @assert ed["times"] == pt["times"]
    @assert maximum(abs.(ed["normψ"].-1)) < 1e-8
    @assert abs(abs(ed["mZ"][1])-1) < 1e-12
    @assert abs(pt["abs_mZ"][1]-1) < 5e-2
    Δ=maximum(abs.(abs.(ed["mZ"]).-pt["abs_mZ"]))
    @printf("N=%3d n=%2d  max |ED-pTWA| = %.6e\n",N,n,Δ)
end
println("All checks passed.")
