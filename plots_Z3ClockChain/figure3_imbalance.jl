using CairoMakie
using JLD2
using Statistics
using LaTeXStrings
using DelimitedFiles

# ===================== PARAMETERS =====================

L       = 12
sampler = "gaussian"

W_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
          5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0]

# TEBD reference data
data05 = "../tebd/imbalance_fk_z3_L12_g0.3_W0.5_U2.dat"
data20 = "../tebd/imbalance_fk_z3_L12_g0.3_W2.0_U2.dat"
data40 = "../tebd/imbalance_fk_z3_L12_g0.3_W4.0_U2.dat"

data80 = "../tebd/imbalance_fk_z3_L12_g0.3_W8.0_U2.dat"

# ===================== LOAD TEBD DATA =====================

t05, I05 = eachcol(readdlm(data05, skipstart=1))
t20, I20 = eachcol(readdlm(data20, skipstart=1))
t40, I40 = eachcol(readdlm(data40, skipstart=1))
t80, I80 = eachcol(readdlm(data80, skipstart=1))

# ===================== FIGURE SETUP =====================

fig = Figure(size = (600, 420))

ax = Axis(
    fig[1, 1],
    xlabel = "time",
    ylabel = "Imbalance",#L"\mathcal{I}(t)",
    xlabelsize = 28,
    ylabelsize = 28,
    xticklabelsize = 18,
    yticklabelsize = 18,
    xgridvisible = true,
    ygridvisible = true,
)

# ===================== COLORMAP =====================

cmap = :viridis
crange = (minimum(W_list), maximum(W_list))

# ===================== MAIN pTWA CURVES =====================

Iinf = Float64[]

for W in W_list

    filename = "../pTWA_Z3ClockChain/output/pTWA_Z3_parafermion_L$(L)_g0.3_W$(W)_Ndis100_Nmc100_discrete.jld2"

    @load filename J gpair W Ndis Nmc seed tmax times sampling I_mean I_err n_mean

    lines!(
        ax,
        times,
        I_mean;
        linewidth = 2,
        color = W,
        colormap = cmap,
        colorrange = crange,
    )

    # Long-time average (last 30%)
    tcut = 0.7 * maximum(times)
    push!(Iinf, mean(I_mean[times .>= tcut]))

end

# ===================== TEBD DATA (SYMBOLS) =====================

scatter!(
    ax, t05[1:10:end], I05[1:10:end];
    markersize = 7,
    color = 0.5,
    colormap = cmap,
    colorrange = crange,
    marker = :circle,
    strokewidth = 1.,
    strokecolor = :black
)

scatter!(
    ax, t20[1:10:end], I20[1:10:end];
    markersize = 7,
    color = 2.0,
    colormap = cmap,
    colorrange = crange,
    marker = :circle,
    strokewidth = 1.,
    strokecolor = :black
)

scatter!(
    ax, t40[1:10:end], I40[1:10:end];
    markersize = 7,
    color = 4.0,
    colormap = cmap,
    colorrange = crange,
    marker = :circle,
    strokewidth = 1.,
    strokecolor = :black
)

scatter!(
    ax, t80[1:10:end], I80[1:10:end];
    markersize = 7,
    color = 8.0,
    colormap = cmap,
    colorrange = crange,
    marker = :circle,
    strokewidth = 1.,
    strokecolor = :black
)

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
    ylabel = "Imbalance[t→∞]",#L"\mathcal{I}_\infty",
    xticklabelsize = 9,
    yticklabelsize = 9,
    xlabelsize = 16,
    ylabelsize = 14,
)

scatter!(
    ax_in,
    W_list,
    Iinf;
    marker = :xcross,
    color = :black,
    markersize = 9,
)

lines!(
    ax_in,
    W_list,
    Iinf;
    color = :black,
    linewidth = 1.5,
)

# ===================== SAVE =====================

save("Fig3_Imbalance_L$(L)_g0.3_vs_W.pdf", fig)
save("Fig3_Imbalance_L$(L)_g0.3_vs_W.png", fig)

println("Saved Makie figure successfully.")


