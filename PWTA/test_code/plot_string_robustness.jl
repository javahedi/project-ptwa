###############################################################################
# Plot robustness of pTWA results to string-symbol discretization
# (Appendix-quality figure)
###############################################################################

using JLD2
using Plots
using LaTeXStrings

# ----------------------- Load saved data -------------------------------------

filename = "string_robustness_Z3_L15_alpha0.5.jld2"
@load filename t S_cont R2_cont S_disc R2_disc

# ----------------------- Plot styling ----------------------------------------

default(
    fontfamily = "Computer Modern",
    linewidth = 2.5,
    legendfontsize = 11,
    guidefontsize = 13,
    tickfontsize = 11,
    titlefontsize = 13,
    framestyle = :box,
)

line_cont = (:solid, :blue)
line_disc = (:dash, :red)

# ----------------------- Panel 1: S(t) ---------------------------------------

pS = plot(
    t, S_cont;
    label = "continuous string",
    linestyle = line_cont[1],
    color = line_cont[2],
    xlabel = L"t",
    ylabel = L"S(t)",
)

plot!(
    pS,
    t, S_disc;
    label = "discrete string",
    linestyle = line_disc[1],
    color = line_disc[2],
)

# ----------------------- Panel 2: R^2(t) -------------------------------------

pR = plot(
    t, R2_cont;
    label = "continuous string",
    linestyle = line_cont[1],
    color = line_cont[2],
    xlabel = L"t",
    ylabel = L"R^2(t)",
)

plot!(
    pR,
    t, R2_disc;
    label = "discrete string",
    linestyle = line_disc[1],
    color = line_disc[2],
)

# ----------------------- Combine panels --------------------------------------

plt = plot(
    pS, pR;
    layout = (2, 1),
    size = (500, 600),
    margin = Plots.mm,
)

savefig(plt, "appendix_string_robustness.pdf")
savefig(plt, "appendix_string_robustness.png")

println("Saved appendix figure → appendix_string_robustness.pdf")
