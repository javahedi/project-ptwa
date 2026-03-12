using CairoMakie, JLD2, LaTeXStrings

# ====================================================
# Global theme
# ====================================================
set_theme!(Theme(
    fontsize = 23,
    Axis = (
        xticklabelsize = 18,
        yticklabelsize = 18,
        xlabelsize = 28,
        ylabelsize = 28,
    )
))

# ====================================================
# Parameters
# ====================================================
N_fixed_top = 30
ns_top      = 3:7

n_fixed_bottom = 5
Ns_bottom      = [10, 20, 30, 40, 50, 60, 70, 80]

ed_dir   = "../ED_Z3ClockChain/Zn/LMG/clock_LMG_scan"
ptwa_dir = "../pTWA_Z3ClockChain/clock_Zn/LMG/pTWA_LMG_Zn"

colors_top   = cgrad(:tab10, length(ns_top))
colors_bottom = cgrad(:tab10, length(Ns_bottom))

# ====================================================
# Figure and axes (two rows, one column)
# ====================================================
fig = Figure(size = (600, 700))

ax_top = Axis(fig[1, 1],
    ylabel = L"|m(t)|"
)

ax_bottom = Axis(fig[2, 1],
    xlabel = "time",
    ylabel = L"|m(t)|"
)

# ====================================================
# Panel labels (fixed parameter info)
# ====================================================
text!(ax_top, 0.05, 0.15, text = L"N = 30\ \text{(fixed)}",
    space = :relative, fontsize = 28, align = (:left, :top))
text!(ax_bottom, 0.05, 0.15, text = L"n = 5\ \text{(fixed)}",
    space = :relative, fontsize = 28, align = (:left, :top))

# ====================================================
# Top panel: fixed N, varying n
# ====================================================
for (i, n) in enumerate(ns_top)
    ed_file = joinpath(ed_dir, "clock_LMG_N$(N_fixed_top)_n$(n).jld2")
    pt_file = joinpath(ptwa_dir, "pTWA_LMG_Zn_N$(N_fixed_top)_n$(n)_g0.5_Ntraj2000.jld2")
    isfile(ed_file) && isfile(pt_file) || continue

    ed = load(ed_file)
    pt = load(pt_file)

    # ED: solid line
    lines!(ax_top, ed["times"], abs.(ed["mz_t"]),
        color = colors_top[i], linewidth = 1,
        label = i == 1 ? "ED" : nothing)

    # pTWA: dashed line + error bars
    lines!(ax_top, pt["times"], pt["abs_mZ_avg"],
        color = colors_top[i], linestyle = :dash, linewidth = 1,
        label = i == 1 ? "pTWA" : nothing)

    errorbars!(ax_top, pt["times"], pt["abs_mZ_avg"], pt["abs_mZ_err"],
        color = colors_top[i], whiskerwidth = 4)
end
axislegend(ax_top, position = :rt, framevisible = true, labelsize = 20)

# ====================================================
# Bottom panel: fixed n, varying N
# ====================================================
for (i, N) in enumerate(Ns_bottom)
    ed_file = joinpath(ed_dir, "clock_LMG_N$(N)_n$(n_fixed_bottom).jld2")
    pt_file = joinpath(ptwa_dir, "pTWA_LMG_Zn_N$(N)_n$(n_fixed_bottom)_g0.5_Ntraj2000.jld2")
    isfile(ed_file) && isfile(pt_file) || continue

    ed = load(ed_file)
    pt = load(pt_file)

    lines!(ax_bottom, ed["times"], abs.(ed["mz_t"]),
        color = colors_bottom[i], linewidth = 1,
        label = i == 1 ? "ED" : nothing)

    lines!(ax_bottom, pt["times"], pt["abs_mZ_avg"],
        color = colors_bottom[i], linestyle = :dash, linewidth = 1,
        label = i == 1 ? "pTWA" : nothing)

    errorbars!(ax_bottom, pt["times"], pt["abs_mZ_avg"], pt["abs_mZ_err"],
        color = colors_bottom[i], whiskerwidth = 4)
end
axislegend(ax_bottom, position = :rt, framevisible = true, labelsize = 20)

# ====================================================
# Arrows indicating increasing parameter values
# ====================================================





arrows2d!(ax_top, [0.6], [0.6], [2], [0.8], 
        color = :black, lengthscale = 0.4, shaftwidth = 1)
text!(ax_top, 2.1, 0.9, text = L"n=3,4,…7",
    fontsize = 22, color = :black, align = (:center, :bottom))

# Bottom panel: arrow from N=10 curve to N=80 curve at same time

arrows2d!(ax_bottom, [3.0], [0.75], [3.8], [0.4 - 1.55], 
        color = :black, lengthscale = 0.25, shaftwidth = 1)
text!(ax_bottom, 4.3, 0.35, text = L"N=10,20,…80",
    fontsize = 22, color = :black, align = (:center, :bottom))

# ====================================================
# Save figure
# ====================================================
save("Fig5_LMG.pdf", fig)
save("Fig5_LMG.png", fig)



# using CairoMakie, JLD2, LaTeXStrings

# # ====================================================
# # Global theme (matches the first code)
# # ====================================================
# set_theme!(Theme(
#     fontsize = 23,
#     Axis = (
#         xticklabelsize = 18,
#         yticklabelsize = 18,
#         xlabelsize = 28,
#         ylabelsize = 28,
#     )
# ))

# # ====================================================
# # Parameters
# # ====================================================
# N_fixed_top = 30
# ns_top      = 3:7

# n_fixed_bottom = 5
# Ns_bottom      = [10, 20, 30, 40, 50, 60, 70, 80]

# ed_dir   = "../ED_Z3ClockChain/Zn/LMG/clock_LMG_scan"
# ptwa_dir = "../pTWA_Z3ClockChain/clock_Zn/LMG/pTWA_LMG_Zn"

# colors_top   = cgrad(:tab10, length(ns_top))
# colors_bottom = cgrad(:tab10, length(Ns_bottom))

# # ====================================================
# # Figure and axes (two rows, one column)
# # ====================================================
# fig = Figure(size = (600, 700))

# ax_top = Axis(fig[1, 1],
#     #xlabel = L"t",
#     ylabel = L"|m_Z(t)|"
# )

# ax_bottom = Axis(fig[2, 1],
#     xlabel = "time",
#     ylabel = L"|m_Z(t)|"
# )

# # ====================================================
# # Panel labels (like the α labels in the first code)
# # ====================================================
# text!(ax_top, 0.05, 0.15, text = L"N = 30\ \text{(fixed)}",
#     space = :relative, fontsize = 28, align = (:left, :top))
# text!(ax_bottom, 0.05, 0.15, text = L"n = 5\ \text{(fixed)}",
#     space = :relative, fontsize = 28, align = (:left, :top))

# # ====================================================
# # Top panel: fixed N, varying n
# # ====================================================
# for (i, n) in enumerate(ns_top)
#     ed_file = joinpath(ed_dir, "clock_LMG_N$(N_fixed_top)_n$(n).jld2")
#     pt_file = joinpath(ptwa_dir, "pTWA_LMG_Zn_N$(N_fixed_top)_n$(n)_g0.5_Ntraj2000.jld2")
#     isfile(ed_file) && isfile(pt_file) || continue

#     ed = load(ed_file)
#     pt = load(pt_file)

#     # ED: solid line
#     lines!(ax_top, ed["times"], abs.(ed["mz_t"]),
#         color = colors_top[i], linewidth = 1,
#         label = i == 1 ? "ED" : nothing)   # label only once

#     # pTWA: dashed line + error bars
#     lines!(ax_top, pt["times"], pt["abs_mZ_avg"],
#         color = colors_top[i], linestyle = :dash, linewidth = 1,
#         label = i == 1 ? "pTWA" : nothing)

#     errorbars!(ax_top, pt["times"], pt["abs_mZ_avg"], pt["abs_mZ_err"],
#         color = colors_top[i], whiskerwidth = 4)
# end

# # Legend for top panel (two entries: ED and pTWA)
# axislegend(ax_top, position = :rt, framevisible = true, labelsize = 20)

# # ====================================================
# # Bottom panel: fixed n, varying N
# # ====================================================
# for (i, N) in enumerate(Ns_bottom)
#     ed_file = joinpath(ed_dir, "clock_LMG_N$(N)_n$(n_fixed_bottom).jld2")
#     pt_file = joinpath(ptwa_dir, "pTWA_LMG_Zn_N$(N)_n$(n_fixed_bottom)_g0.5_Ntraj2000.jld2")
#     isfile(ed_file) && isfile(pt_file) || continue

#     ed = load(ed_file)
#     pt = load(pt_file)

#     lines!(ax_bottom, ed["times"], abs.(ed["mz_t"]),
#         color = colors_bottom[i], linewidth = 1,
#         label = i == 1 ? "ED" : nothing)

#     lines!(ax_bottom, pt["times"], pt["abs_mZ_avg"],
#         color = colors_bottom[i], linestyle = :dash, linewidth = 1,
#         label = i == 1 ? "pTWA" : nothing)

#     errorbars!(ax_bottom, pt["times"], pt["abs_mZ_avg"], pt["abs_mZ_err"],
#         color = colors_bottom[i], whiskerwidth = 4)
# end

# axislegend(ax_bottom, position = :rt, framevisible = true, labelsize = 20)


# # ------------------------------------------------------------
# # Save figure
# # ------------------------------------------------------------

# save("Fig5_LMG.pdf", fig)
# save("Fig5_LMG.png", fig)