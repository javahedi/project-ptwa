using CairoMakie
using JLD2
using LaTeXStrings

# ----------------------------
# Load data
# ----------------------------
#sampling = :gaussian
sampling = :discrete

α = 1.5
@load "pTWA_Z3_L13_alpha$(α)_g0.5_single_excitation_$(sampling)_N2000.jld2" times Pexc_avg center_exc center_err avg_disp disp_err j0 L

Nt = length(times)

# ----------------------------
# Makie theme
# ----------------------------
set_theme!(Theme(
    fontsize = 20,
    Axis = (
        xticklabelsize = 18,
        yticklabelsize = 18
    )
))

# ----------------------------
# Figure
# ----------------------------
fig = Figure(size = (600, 900))

# ---- Heatmap (light cone) ----
ax1 = Axis(fig[1,1],
    xlabel = "site j",
    ylabel = L"time"
)

hm = heatmap!(ax1, 1:L, times, Pexc_avg,
    colormap = :gnuplot,
    colorrange = (0.0, 1.0)
)

Colorbar(fig[1,2], hm)

# ---- Center decay with error bars ----
ax2 = Axis(fig[2,1],
    xlabel = L"time",
    ylabel = L"P_{exc}(j_0,t)"
)

lines!(ax2, times, center_exc, linewidth = 3, color = :blue)

errorbars!(
    ax2,
    times,
    center_exc,
    center_err,
    color = range(0, 1, length = length(times)),
    whiskerwidth = 5
)

# ---- Average displacement with error bars ----
ax3 = Axis(fig[3,1],
    xlabel = L"time",
    ylabel = L"\langle |j-j_0| \rangle"
)

lines!(ax3, times, avg_disp, linewidth = 3, color = :blue)

errorbars!(
    ax3,
    times,
    avg_disp,
    disp_err,
    color = range(0, 1, length = length(times)),
    whiskerwidth = 5

)

# ----------------------------
# Save figure
# ----------------------------
save("pTWA_Z3_lightcone_L13_alpha$(α)_$(sampling).png", fig)