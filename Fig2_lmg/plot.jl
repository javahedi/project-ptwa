using CairoMakie
using JLD2
using LaTeXStrings

N_top=30; 
ns_top=collect(3:7); 
n_bottom=5; 
Ns_bottom=collect(10:10:80); 
Ntraj=2000
ed_dir=joinpath(@__DIR__,"data","ed")
pt_dir=joinpath(@__DIR__,"data","ptwa")

set_theme!(Theme(fontsize=21,Axis=(xticklabelsize=17,yticklabelsize=17,xlabelsize=23,ylabelsize=23)))
fig=Figure(size=(650,720))
ax1=Axis(fig[1,1],ylabel=L"|m(t)|")
ax2=Axis(fig[2,1],xlabel=L"Jt",ylabel=L"|m(t)|")

ct=cgrad(:tab10,length(ns_top),categorical=true)
cb=cgrad(:tab10,length(Ns_bottom),categorical=true)

for (i,n) in enumerate(ns_top)
    ed=load(joinpath(ed_dir,"ed_LMG_N$(N_top)_n$(n).jld2"))
    pt=load(joinpath(pt_dir,"ptwa_LMG_N$(N_top)_n$(n)_Ntraj$(Ntraj).jld2"))
    lines!(ax1,ed["times"],abs.(ed["mZ"]),linewidth=2,color=ct[i])
    lines!(ax1,pt["times"],pt["abs_mZ"],linewidth=2,linestyle=:dash,color=ct[i])
end

for (i,N) in enumerate(Ns_bottom)
    ed=load(joinpath(ed_dir,"ed_LMG_N$(N)_n$(n_bottom).jld2"))
    pt=load(joinpath(pt_dir,"ptwa_LMG_N$(N)_n$(n_bottom)_Ntraj$(Ntraj).jld2"))
    lines!(ax2,ed["times"],abs.(ed["mZ"]),linewidth=2,color=cb[i])
    lines!(ax2,pt["times"],pt["abs_mZ"],linewidth=2,linestyle=:dash,color=cb[i])
end

# axislegend(
#     ax1,
#     [LineElement(linewidth=2),
#      LineElement(linewidth=2, linestyle=:dash)],
#     ["ED", "pTWA"],
#     position = :rt,
#     framevisible = false
# )
# text!(ax1,0.04,0.08,text=L"N=30",space=:relative,align=(:left,:bottom))
# text!(ax2,0.04,0.08,text=L"n=5",space=:relative,align=(:left,:bottom))
# text!(ax1,0.97,0.93,text=L"n=3,\ldots,7",space=:relative,align=(:right,:top),fontsize=18)
# text!(ax2,0.97,0.93,text=L"N=10,20,\ldots,80",space=:relative,align=(:right,:top),fontsize=18)

# ------------------------------------------------------------
# Legend inside top panel
# ------------------------------------------------------------

axislegend(
    ax1,
    [LineElement(linewidth=2),
     LineElement(linewidth=2, linestyle=:dash)],
    ["ED", "pTWA"],
    position = :rt,
    framevisible = false
)

# ------------------------------------------------------------
# Fixed-parameter labels
# ------------------------------------------------------------

text!(ax1, 0.04, 0.08,
    text = L"N=30",
    space = :relative,
    align = (:left, :bottom)
)

text!(ax2, 0.04, 0.08,
    text = L"n=5",
    space = :relative,
    align = (:left, :bottom)
)

# ------------------------------------------------------------
# Arrows indicating increasing n and N
# ------------------------------------------------------------

# Top panel: increasing n = 3 -> 7
arrows2d!(
    ax1,
    [0.75], [0.60], # x0,y0
    [1.35], [0.35],
    color = :black,
    shaftwidth = 1.5,
    lengthscale = 0.55
)

text!(
    ax1,
    2.05, 0.90,
    text = L"n=3,\ldots,7",
    fontsize = 18,
    align = (:center, :bottom)
)

# Bottom panel: increasing N = 10 -> 80
arrows2d!(
    ax2,
    [2.05], [0.10],
    [1.75], [0.35],
    color = :black,
    shaftwidth = 1.5,
    lengthscale = 0.55
)

text!(
    ax2,
    3.65, 0.34,
    text = L"N=10,20,\ldots,80",
    fontsize = 18,
    align = (:center, :bottom)
)

save(joinpath(@__DIR__,"Fig_LMG.pdf"),fig)
save(joinpath(@__DIR__,"Fig_LMG.png"),fig,px_per_unit=2)
