using CairoMakie
using DelimitedFiles

set_theme!(Theme(
    fontsize = 18,
    Axis = (
        xlabelsize = 20,
        ylabelsize = 20,
        xticklabelsize = 16,
        yticklabelsize = 16,
    )
))

############################################################
# Load heatmap data S_a(q,t)
############################################################

function load_heatmap(n)

    file = "structure_a_singlehop_domainwall_heatmap_n$(n)_L24.dat"

    data = readdlm(file)

    q  = data[:,1]
    Sq = data[:,2:end]

    return q, Sq
end

############################################################
# Load imbalance dynamics
############################################################

function load_dynamics(n)

    file = "imbalance_a_singlehop_domainwall_n$(n)_L24.dat"

    data = readdlm(file)

    t  = data[:,1]
    I  = data[:,2]

    return t, I
end

############################################################
# Load systems
############################################################

q3,S3 = load_heatmap(3)
q4,S4 = load_heatmap(4)
q5,S5 = load_heatmap(5)

t3,I3 = load_dynamics(3)
t4,I4 = load_dynamics(4)
t5,I5 = load_dynamics(5)

############################################################
# time grid
############################################################

steps = size(S3,2)
dt = t3[2] - t3[1]
times = (0:steps-1) .* dt

############################################################
# Figure
############################################################

fig = Figure(size=(900,900))

############################################################
# Heatmap n=3
############################################################

ax1 = Axis(fig[1,1],
    title = L"S_a(q,t),\; n=3",
    xlabel = L"q",
    ylabel = L"Jt"
)

hm1 = heatmap!(ax1, q3, times, S3, colormap=:tofino25)

Colorbar(fig[1,2], hm1)

############################################################
# Heatmap n=4
############################################################

ax2 = Axis(fig[2,1],
    title = L"S_a(q,t),\; n=4",
    xlabel = L"q",
    ylabel = L"Jt"
)

hm2 = heatmap!(ax2, q4, times, S4, colormap=:tofino25)

Colorbar(fig[2,2], hm2)

############################################################
# Heatmap n=5
############################################################

ax3 = Axis(fig[3,1],
    title = L"S_a(q,t),\; n=5",
    xlabel = L"q",
    ylabel = L"Jt"
)

hm3 = heatmap!(ax3, q5, times, S5, colormap=:tofino25)

Colorbar(fig[3,2], hm3)

############################################################
# Domain-wall imbalance
############################################################

ax4 = Axis(fig[4,1:2],
    xlabel = L"Jt",
    ylabel = L"\mathcal{I}_a(t)",
    title = "Domain-wall imbalance"
)

lines!(ax4, t3, I3, label = L"n=3", linewidth=3)
lines!(ax4, t4, I4, label = L"n=4", linewidth=3)
lines!(ax4, t5, I5, label = L"n=5", linewidth=3)

axislegend(ax4)

save("tebd_domainwall_heatmap_imbalance.png", fig)
