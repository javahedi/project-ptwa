using CairoMakie
using DelimitedFiles
using JLD2

set_theme!(Theme(
    fontsize = 20,
    Axis = (
        xlabelsize = 22,
        ylabelsize = 22,
        xticklabelsize = 18,
        yticklabelsize = 18,
    )
))

############################################################
# Load TEBD heatmap
############################################################

function load_tebd_heatmap(n)
    file = "../tebd/structure_a_singlehop_domainwall_heatmap_n$(n)_L24.dat"
    data = readdlm(file)
    q  = data[:,1]
    Sq = data[:,2:end]
    return q, Sq
end

############################################################
# Load TEBD imbalance
############################################################

function load_tebd_imbalance(n)
    file = "../tebd/imbalance_a_singlehop_domainwall_n$(n)_L24.dat"
    data = readdlm(file)
    t = data[:,1]
    I = data[:,2]
    return t, I
end

############################################################
# Load pTWA heatmap
############################################################

function load_ptwa_heatmap(n)
    file = "../pTWA_Z3ClockChain/parafermion_Z3/structure_factor/pTWA_Zn_domainwall_singlehop_n$(n)_L24_gaussian_N2000.jld2"
    data = load(file)
    q  = data["qs"]
    Sq = data["Sa_avg"]
    return q, Sq
end

############################################################
# Load pTWA imbalance
############################################################

function load_ptwa_imbalance(n)
    file = "../pTWA_Z3ClockChain/parafermion_Z3/structure_factor/pTWA_Zn_domainwall_singlehop_n$(n)_L24_gaussian_N2000.jld2"
    data = load(file)
    t = data["times"]
    I = data["I_avg"]
    return t, I
end

############################################################
# Load data
############################################################

ns = 3:7

q_tebd = Dict()
S_tebd = Dict()
t_tebd = Dict()
I_tebd = Dict()

q_ptwa = Dict()
S_ptwa = Dict()
t_ptwa = Dict()
I_ptwa = Dict()

for n in ns
    q_tebd[n], S_tebd[n] = load_tebd_heatmap(n)
    t_tebd[n], I_tebd[n] = load_tebd_imbalance(n)

    q_ptwa[n], S_ptwa[n] = load_ptwa_heatmap(n)
    t_ptwa[n], I_ptwa[n] = load_ptwa_imbalance(n)
end

############################################################
# Figure layout
############################################################

fig = Figure(size = (1000, 800))

colgap!(fig.layout, 5)
rowgap!(fig.layout, 5)

############################################################
# Row 1 — TEBD heatmaps
############################################################

ax1 = Axis(fig[1,1], title=L"TEBD,\;n=3", ylabel="time")
ax2 = Axis(fig[1,2], title=L"TEBD,\;n=5")
ax3 = Axis(fig[1,3], title=L"TEBD,\;n=7")

hm = heatmap!(ax1, q_tebd[3], t_tebd[3], S_tebd[3], colormap=:tofino25)
heatmap!(ax2, q_tebd[5], t_tebd[5], S_tebd[5], colormap=:tofino25)
heatmap!(ax3, q_tebd[7], t_tebd[7], S_tebd[7], colormap=:tofino25)

############################################################
# Row 2 — pTWA heatmaps
############################################################

ax4 = Axis(fig[2,1], title=L"pTWA,\;n=3", xlabel=L"q", ylabel="time")
ax5 = Axis(fig[2,2], title=L"pTWA,\;n=5", xlabel=L"q")
ax6 = Axis(fig[2,3], title=L"pTWA,\;n=7", xlabel=L"q")

heatmap!(ax4, q_ptwa[3], t_ptwa[3], S_ptwa[3], colormap=:tofino25)
heatmap!(ax5, q_ptwa[5], t_ptwa[5], S_ptwa[5], colormap=:tofino25)
heatmap!(ax6, q_ptwa[7], t_ptwa[7], S_ptwa[7], colormap=:tofino25)

############################################################
# Shared colorbar
############################################################

cb = Colorbar(fig[1:2,4], hm, label = L"S(q)")
colsize!(fig.layout, 4, Auto(0.2))

############################################################
# Bottom panel — imbalance
############################################################

ax7 = Axis(fig[3,1:3],
    xlabel = "time",
    ylabel = "Imbalance",
    title  = "Domain-wall imbalance"
)

colors = [:blue, :orange, :green, :red, :purple]

# Exact (TEBD) — solid
for (i,n) in enumerate(ns)
    lines!(ax7, t_tebd[n], I_tebd[n],
        color = colors[i],
        linewidth = 1)
end

# pTWA — dashed
for (i,n) in enumerate(ns)
    lines!(ax7, t_ptwa[n], I_ptwa[n],
        color = colors[i],
        linestyle = :dash,
        linewidth = 1)
end

############################################################
# Clean legend (method + n)
############################################################

Legend(fig[3,4],
[
    LineElement(color=:black, linewidth=1),
    LineElement(color=:black, linestyle=:dash, linewidth=1),
    LineElement(color=colors[1], linewidth=1),
    LineElement(color=colors[2], linewidth=1),
    LineElement(color=colors[3], linewidth=1),
    LineElement(color=colors[4], linewidth=1),
    LineElement(color=colors[5], linewidth=1)
],
[
    "TEBD",
    "pTWA",
    L"n=3",
    L"n=4",
    L"n=5",
    L"n=6",
    L"n=7"
],
framevisible=false
)

############################################################
# Final layout adjustments
############################################################

resize_to_layout!(fig)

save("Fig4_Zn.png", fig)
save("Fig4_Zn.pdf", fig)

