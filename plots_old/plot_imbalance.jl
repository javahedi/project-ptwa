using CairoMakie
using JLD2
using Statistics

# ===================== PARAMETERS =====================

L       = 12
α       = 6.0
sampler = "discrete"

W_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

# ===================== FIGURE SETUP =====================

fig = Figure(resolution = (600, 420))

ax = Axis(
    fig[1, 1],
    xlabel = "time t",
    ylabel = L"Imbalance $\mathcal{I}(t)$",
    xgridvisible = true,
    ygridvisible = true,
    xlabelsize = 16,
    ylabelsize = 16

)

# ===================== COLORMAP =====================

cmap = :viridis
crange = (minimum(W_list), maximum(W_list))

# ===================== MAIN CURVES =====================

Iinf = Float64[]

for W in W_list
    filename = "imbalance_pTWA_Z3_L$(L)_alpha$(α)_W$(W)_$(sampler)_safe.jld2"
    @load filename t I

    lines!(
        ax,
        t,
        I;
        linewidth = 2,
        color = W,
        colormap = cmap,
        colorrange = crange,
    )

    # Long-time average (last 30%)
    tcut = 0.7 * maximum(t)
    push!(Iinf, mean(I[t .>= tcut]))
end

# ===================== COLORBAR =====================

Colorbar(
    fig[1, 2],
    colormap = cmap,
    limits = crange,
    label = L"Disorder strength $W$",
    width = 14,
    labelsize = 16,

)

# ===================== INSET: I_infty vs W =====================

ax_in = Axis(
    fig[1, 1],
    width = Relative(0.38),
    height = Relative(0.38),
    halign = 0.97,
    valign = 0.97,
    xlabel = L"$W$",
    ylabel = L"$\mathcal{I}_\infty$",
    xticklabelsize = 9,
    yticklabelsize = 9,
    xlabelsize = 16,
    ylabelsize = 16,
)

scatter!(
    ax_in,
    W_list,
    Iinf;
    color = :black,
    markersize = 7,
)

lines!(
    ax_in,
    W_list,
    Iinf;
    color = :black,
    linewidth = 1.5,
)

# ===================== SAVE =====================

save("Fig_Imbalance_L$(L)_alpha$(α)_vs_W.pdf", fig)
save("Fig_Imbalance_L$(L)_alpha$(α)_vs_W.png", fig)

println("Saved Makie figure successfully.")
