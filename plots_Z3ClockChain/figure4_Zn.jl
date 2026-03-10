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
    file = "../pTWA_Z3ClockChain/structure_factor/pTWA_Zn_domainwall_singlehop_n$(n)_L24_gaussian_N2000.jld2"
    data = load(file)
    q  = data["qs"]
    Sq = data["Sa_avg"]
    return q, Sq
end

############################################################
# Load pTWA imbalance
############################################################

function load_ptwa_imbalance(n)
    file = "../pTWA_Z3ClockChain/structure_factor/pTWA_Zn_domainwall_singlehop_n$(n)_L24_gaussian_N2000.jld2"
    data = load(file)
    t = data["times"]
    I = data["I_avg"]
    return t, I
end

############################################################
# Load all systems
############################################################

q3_tebd, S3_tebd = load_tebd_heatmap(3)
q4_tebd, S4_tebd = load_tebd_heatmap(4)
q5_tebd, S5_tebd = load_tebd_heatmap(5)

q3_ptwa, S3_ptwa = load_ptwa_heatmap(3)
q4_ptwa, S4_ptwa = load_ptwa_heatmap(4)
q5_ptwa, S5_ptwa = load_ptwa_heatmap(5)

t3_tebd, I3_tebd = load_tebd_imbalance(3)
t4_tebd, I4_tebd = load_tebd_imbalance(4)
t5_tebd, I5_tebd = load_tebd_imbalance(5)

t3_ptwa, I3_ptwa = load_ptwa_imbalance(3)
t4_ptwa, I4_ptwa = load_ptwa_imbalance(4)
t5_ptwa, I5_ptwa = load_ptwa_imbalance(5)

############################################################
# Figure layout
############################################################

fig = Figure(size = (1000, 800))   # Slightly reduced size

# Adjust gaps between subplots
colgap!(fig.layout, 5)
rowgap!(fig.layout, 5)

############################################################
# Row 1 — TEBD heatmaps (use TEBD time vectors)
############################################################

ax1 = Axis(fig[1,1], title=L"TEBD,\;n=3", ylabel="time")
ax2 = Axis(fig[1,2], title=L"TEBD,\;n=4")
ax3 = Axis(fig[1,3], title=L"TEBD,\;n=5")

hm = heatmap!(ax1, q3_tebd, t3_tebd, S3_tebd, colormap=:tofino25)
heatmap!(ax2, q4_tebd, t4_tebd, S4_tebd, colormap=:tofino25)
heatmap!(ax3, q5_tebd, t5_tebd, S5_tebd, colormap=:tofino25)

############################################################
# Row 2 — pTWA heatmaps (use pTWA time vectors)
############################################################

ax4 = Axis(fig[2,1], title=L"pTWA,\;n=3", xlabel=L"q", ylabel="time")
ax5 = Axis(fig[2,2], title=L"pTWA,\;n=4", xlabel=L"q")
ax6 = Axis(fig[2,3], title=L"pTWA,\;n=5", xlabel=L"q")

heatmap!(ax4, q3_ptwa, t3_ptwa, S3_ptwa, colormap=:tofino25)
heatmap!(ax5, q4_ptwa, t4_ptwa, S4_ptwa, colormap=:tofino25)
heatmap!(ax6, q5_ptwa, t5_ptwa, S5_ptwa, colormap=:tofino25)

############################################################
# Shared colorbar (placed in column 4, rows 1-2)
############################################################

cb = Colorbar(fig[1:2,4], hm, label = L"S(q)")
colsize!(fig.layout, 4, Auto(0.2))   # Make colorbar column narrower

############################################################
# Bottom panel — imbalance
############################################################

ax7 = Axis(fig[3,1:3],
    xlabel = "time",
    ylabel = "Imbalance",
    title  = "Domain-wall imbalance"
)

l1 = lines!(ax7, t3_tebd, I3_tebd, color=:blue, linewidth=3)
l2 = lines!(ax7, t3_ptwa, I3_ptwa, color=:blue, linestyle=:dash, linewidth=3)

l3 = lines!(ax7, t4_tebd, I4_tebd, color=:orange, linewidth=3)
l4 = lines!(ax7, t4_ptwa, I4_ptwa, color=:orange, linestyle=:dash, linewidth=3)

l5 = lines!(ax7, t5_tebd, I5_tebd, color=:green, linewidth=3)
l6 = lines!(ax7, t5_ptwa, I5_ptwa, color=:green, linestyle=:dash, linewidth=3)

############################################################
# Legend (placed in column 4, row 3)
############################################################

Legend(fig[3,4],
    [l1, l2, l3, l4, l5, l6],
    [
        L"n=3\;TEBD",
        L"n=3\;pTWA",
        L"n=4\;TEBD",
        L"n=4\;pTWA",
        L"n=5\;TEBD",
        L"n=5\;pTWA"
    ],
    framevisible=false
)

############################################################
# Final layout adjustments and saving
############################################################

# Tighten layout to avoid clipping
resize_to_layout!(fig)

save("Fig4_Zn.png", fig)
save("Fig4_Zn.pdf", fig)

