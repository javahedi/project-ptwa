
using CairoMakie
using JLD2
using LaTeXStrings

set_theme!(Theme(
    fontsize = 23,
    Axis = (
        xticklabelsize = 18,
        yticklabelsize = 18,
        xlabelsize = 28,
        ylabelsize = 28,
    )
))

fig = Figure(size = (1000,900))

# ====================================================
# Axes
# ====================================================

# α = 3.0
ax11 = Axis(fig[1,1], ylabel=L"P_{\mathrm{exc}}(j_0,t)")
ax12 = Axis(fig[1,2], ylabel=L"\langle |j-j_0| \rangle")

# α = 1.5
ax21 = Axis(fig[2,1], ylabel=L"P_{\mathrm{exc}}(j_0,t)", limits=(0,nothing,0.9,1.001))
ax22 = Axis(fig[2,2], ylabel=L"\langle |j-j_0| \rangle")

# α = 0.5
ax31 = Axis(fig[3,1],
    xlabel="time",
    ylabel=L"P_{\mathrm{exc}}(j_0,t)",
    limits=(0,nothing,0.99,1.001)
)

ax32 = Axis(fig[3,2],
    xlabel="time",
    ylabel=L"\langle |j-j_0| \rangle"
)

# ====================================================
# Labels inside panels
# ====================================================

text!(ax11,0.05,0.1,text=L"\alpha=3.0",space=:relative,fontsize=28)
text!(ax12,0.05,0.6,text=L"\alpha=3.0",space=:relative,fontsize=28)

text!(ax21,0.05,0.1,text=L"\alpha=1.5",space=:relative,fontsize=28)
text!(ax22,0.4,0.1,text=L"\alpha=1.5",space=:relative,fontsize=28)

text!(ax31,0.4,0.1,text=L"\alpha=0.5",space=:relative,fontsize=28)
text!(ax32,0.4,0.1,text=L"\alpha=0.5",space=:relative,fontsize=28)

# ====================================================
# Function to load and plot a given α
# ====================================================

function plot_alpha!(ax_exc, ax_disp, α; legend=false)

    # ----- ED
    @load "../ED_Z3ClockChain/ED_Z3_L13_alpha$(α)_g0.5_single_excitation.jld2" times center_exc avg_disp

    lines!(ax_exc, times, center_exc,
        color=:black, linewidth=3, label=legend ? "ED" : nothing)

    lines!(ax_disp, times, avg_disp,
        color=:black, linewidth=3)

    # ----- Gaussian
    @load "../pTWA_Z3ClockChain/pTWA_Z3_L13_alpha$(α)_g0.5_single_excitation_gaussian_N10000.jld2" times center_exc center_err avg_disp disp_err

    errorbars!(ax_exc, times, center_exc, center_err,
        color=:red, whiskerwidth=4)

    lines!(ax_exc, times, center_exc,
        color=:red, linestyle=:dash,
        label=legend ? "Gaussian" : nothing)

    errorbars!(ax_disp, times, avg_disp, disp_err,
        color=:red, whiskerwidth=4)

    lines!(ax_disp, times, avg_disp,
        color=:red, linestyle=:dash)

    # # ----- Discrete
    @load "../pTWA_Z3ClockChain/pTWA_Z3_L13_alpha$(α)_g0.5_single_excitation_discrete_N10000.jld2" times center_exc center_err avg_disp disp_err

    errorbars!(ax_exc, times, center_exc, center_err,
        color=:blue, whiskerwidth=4)

    lines!(ax_exc, times, center_exc,
        color=:blue, label=legend ? "Discrete" : nothing)

    errorbars!(ax_disp, times, avg_disp, disp_err,
        color=:blue, whiskerwidth=4)

    lines!(ax_disp, times, avg_disp,
        color=:blue)

end

# ====================================================
# Plot all α values
# ====================================================

plot_alpha!(ax11, ax12, 3.0, legend=true)
plot_alpha!(ax21, ax22, 1.5)
plot_alpha!(ax31, ax32, 0.5)

axislegend(ax11)

# ====================================================
# Save
# ====================================================

save("Fig2_dynamics_Z3.png", fig)
save("Fig2_dynamics_Z3.pdf", fig)

