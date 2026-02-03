###############################################################################
# Plot robustness of pTWA results to string-symbol discretization
# (Appendix-quality figure)
###############################################################################

using JLD2
using CairoMakie
using LaTeXStrings

# ----------------------- Load saved data -------------------------------------

filename = "string_robustness_Z3_L15_alpha0.5.jld2"
@load filename t S_cont R2_cont S_disc R2_disc

# ----------------------- Create figure ---------------------------------------

fig = Figure(size=(600, 250), fontsize=14, font="Computer Modern")

# ----------------------- Panel 1: S(t) ---------------------------------------

ax1 = Axis(fig[1, 1],
    xlabel=L"t",
    ylabel=L"S(t)",
    xgridvisible=true,
    ygridvisible=true,
    xgridcolor=(:gray, 0.2),
    ygridcolor=(:gray, 0.2)
)

# Continuous string
lines!(ax1, t, S_cont,
    linewidth=2.5,
    color=:blue,
    linestyle=:solid,
    label="continuous string"
)

# Discrete string  
lines!(ax1, t, S_disc,
    linewidth=2.5,
    color=:red,
    linestyle=:dash,
    label="discrete string"
)

# Legend inside the plot (top right)
axislegend(ax1, position=:rt, framevisible=true, bgcolor=(:white, 0.8))

# ----------------------- Panel 2: R²(t) -------------------------------------

ax2 = Axis(fig[1, 2],
    xlabel=L"t",
    ylabel=L"R^2(t)",
    xgridvisible=true,
    ygridvisible=true,
    xgridcolor=(:gray, 0.2),
    ygridcolor=(:gray, 0.2)
)

# Continuous string
lines!(ax2, t, R2_cont,
    linewidth=2.5,
    color=:blue,
    linestyle=:solid,
    label="continuous string"
)

# Discrete string
lines!(ax2, t, R2_disc,
    linewidth=2.5,
    color=:red,
    linestyle=:dash,
    label="discrete string"
)

# Legend inside the plot (top right)
axislegend(ax2, position=:rt, framevisible=true, bgcolor=(:white, 0.8))

# ----------------------- Adjust layout --------------------------------------

# Add some spacing between panels
rowgap!(fig.layout, 30)

# Set same x-limits for both panels
xlims!(ax1, (minimum(t), maximum(t)))
xlims!(ax2, (minimum(t), maximum(t)))

# Adjust y-limits to be reasonable
ylims!(ax1, (0, 1))
#ylims!(ax2, (0, 1))

# ----------------------- Save figure ----------------------------------------

save("appendix_string_robustness.pdf", fig)
save("appendix_string_robustness.png", fig)

println("Saved appendix figure → appendix_string_robustness_cairo.pdf")

# ###############################################################################
# # Plot robustness of pTWA results to string-symbol discretization
# # (Appendix-quality figure)
# ###############################################################################

# using JLD2
# using Plots
# using LaTeXStrings

# # ----------------------- Load saved data -------------------------------------

# filename = "string_robustness_Z3_L15_alpha0.5.jld2"
# @load filename t S_cont R2_cont S_disc R2_disc

# # ----------------------- Plot styling ----------------------------------------

# default(
#     fontfamily = "Computer Modern",
#     linewidth = 2.5,
#     legendfontsize = 11,
#     guidefontsize = 13,
#     tickfontsize = 11,
#     titlefontsize = 13,
#     framestyle = :box,
# )

# line_cont = (:solid, :blue)
# line_disc = (:dash, :red)

# # ----------------------- Panel 1: S(t) ---------------------------------------

# pS = plot(
#     t, S_cont;
#     label = "continuous string",
#     linestyle = line_cont[1],
#     color = line_cont[2],
#     xlabel = L"t",
#     ylabel = L"S(t)",
# )

# plot!(
#     pS,
#     t, S_disc;
#     label = "discrete string",
#     linestyle = line_disc[1],
#     color = line_disc[2],
# )

# # ----------------------- Panel 2: R^2(t) -------------------------------------

# pR = plot(
#     t, R2_cont;
#     label = "continuous string",
#     linestyle = line_cont[1],
#     color = line_cont[2],
#     xlabel = L"t",
#     ylabel = L"R^2(t)",
# )

# plot!(
#     pR,
#     t, R2_disc;
#     label = "discrete string",
#     linestyle = line_disc[1],
#     color = line_disc[2],
# )

# # ----------------------- Combine panels --------------------------------------

# plt = plot(
#     pS, pR;
#     layout = (2, 1),
#     size = (500, 600),
#     margin = Plots.mm,
# )

# savefig(plt, "appendix_string_robustness.pdf")
# savefig(plt, "appendix_string_robustness.png")

# println("Saved appendix figure → appendix_string_robustness.pdf")
