using CairoMakie
using JLD2
using LaTeXStrings

# ----------------------------
# Load data
# ----------------------------
α = 0.5
@load "ED_Z3_L13_alpha$(α)_g0.5_single_excitation.jld2" times P1 P2 Pexc center_exc avg_disp j0 L

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

hm = heatmap!(ax1, 1:L, times, Pexc,
    colormap = :gnuplot, colorrange = (0.0, 1.0)
)

Colorbar(fig[1,2], hm)

# ---- Center decay ----
ax2 = Axis(fig[2,1],
    xlabel = L"time",
    ylabel = L"P_1(j_0,t)"
)

lines!(ax2, times, center_exc, linewidth = 3, color = :blue)

# ---- Average displacement ----
ax3 = Axis(fig[3,1],
    xlabel = L"time",
    ylabel = L"\langle |j-j_0| \rangle"
)

lines!(ax3, times, avg_disp, linewidth = 3, color = :blue)

# ----------------------------
# Save figure
# ----------------------------
save("ED_Z3_lightcone_L13_alpha$(α).png", fig)