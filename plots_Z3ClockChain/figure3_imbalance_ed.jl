using CairoMakie
using JLD2
using Statistics
using LaTeXStrings
using DelimitedFiles

# ===================== PARAMETERS =====================

L       = 10
sampler = "gaussian"

W_list = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
seed_list = 0:8#[9, 10, 11, 12, 13, 14, 15, 16, 17]
# averaging window
tmin = 2.0
tcut = 20.0



# ===================== WINDOW AVERAGE FUNCTION =====================

function window_average(t, I, tmin, tcut)
    mask = (t .>= tmin) .& (t .<= tcut)
    return mean(I[mask])
end



# ===================== FIGURE SETUP =====================

fig = Figure(size = (600, 420))

ax = Axis(
    fig[1, 1],
    xlabel = "time",
    ylabel = "Imbalance",
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

# ===================== pTWA CURVES =====================

Iavg = Float64[]

for W in W_list
                   
    filename = "../pTWA_Z3ClockChain/parafermion_Z3/output/pTWA_Z3_parafermion_L$(L)_g0.3_W$(W)_Ndis100_Nmc100_gaussian.jld2"

    @load filename times I_mean

    lines!(
        ax,
        times,
        I_mean;
        linewidth = 2,
        color = W,
        colormap = cmap,
        colorrange = crange,
    )

    push!(Iavg, window_average(times, I_mean, tmin, tcut))
end

# ===================== ED DATA (SYMBOLS) =====================
I_ed_avg = Float64[]

for (s, W) in zip(seed_list, W_list)
        filename = "../ED_Z3ClockChain/Z3/ED_Z3_parafermion_L$(L)_g0.3_W$(W)_seed$(s).jld2"
        #data = load(filename)
        #println(keys(data))
        @load filename times I_mean
        #@load filename times I_sum r_completed
        #I_mean = I_sum ./ r_completed

         lines!(
            ax,
            times,
            I_mean;
            linewidth = 2,
            color = W,
            colormap = cmap,
            colorrange = crange,
            linestyle = :dash
        )

        # scatter!(
        #     ax, times, I_mean;
        #     markersize = 7,
        #     color = W,
        #     colormap = cmap,
        #     colorrange = crange,
        #     marker = :circle,
        #     strokewidth = 1,
        #     strokecolor = :black
        # )

        push!(I_ed_avg, window_average(times, I_mean, tmin, tcut))
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

# ===================== INSET =====================

ax_in = Axis(
    fig[1, 1],
    width = Relative(0.38),
    height = Relative(0.38),
    halign = 0.97,
    valign = 0.97,
    xlabel = L"W",
    ylabel = L"\bar{\mathcal{I}}",
    xticklabelsize = 9,
    yticklabelsize = 9,
    xlabelsize = 16,
    ylabelsize = 14,
)

# pTWA averages
scatter!(
    ax_in,
    W_list,
    Iavg;
    marker = :xcross,
    color = :black,
    markersize = 9,
)

lines!(
    ax_in,
    W_list,
    Iavg;
    color = :black,
    linewidth = 1.5,
)

# ED averages
scatter!(
    ax_in,
    W_list,
    I_ed_avg;
    marker = :circle,
    markersize = 8,
    color = :red
)

# ===================== SAVE =====================

save("Fig3_Imbalance_L$(L)_g0.3_vs_W_windowavg.pdf", fig)
save("Fig3_Imbalance_L$(L)_g0.3_vs_W_windowavg.png", fig)

println("Saved Makie figure successfully.")