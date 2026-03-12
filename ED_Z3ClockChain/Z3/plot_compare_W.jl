# plot_compare_W.jl
using JLD2
using CairoMakie
using LaTeXStrings

# ----------------------------
# Helper: load one dataset
# ----------------------------
function load_run(path::String)
    @load path times I_mean I_err n_mean L W gpair Ndis
    return (path=path, times=times, I_mean=I_mean, I_err=I_err,
            n_mean=n_mean, L=L, W=W, gpair=gpair, Ndis=Ndis)
end

# ----------------------------
# Provide your two files here
# ----------------------------
file_W05 = "output/ED_Z3_parafermion_L8_g0.3_W0.5_Ndis20.jld2"
file_W1 = "output/ED_Z3_parafermion_L8_g0.3_W1.0_Ndis20.jld2"
file_W2 = "output/ED_Z3_parafermion_L8_g0.3_W2.0_Ndis20.jld2"
file_W4 = "output/ED_Z3_parafermion_L8_g0.3_W4.0_Ndis20.jld2"

d05 = load_run(file_W05)
d1 = load_run(file_W1)
d2 = load_run(file_W2)
d4 = load_run(file_W4)

@assert d2.L == d4.L "L mismatch between files"
@assert length(d2.times) == length(d4.times) "time grid mismatch"

# ----------------------------
# Plot I(t) comparison
# ----------------------------
fig1 = Figure(size=(600, 300))
ax = Axis(fig1[1, 1],
    xlabel=L"t\,[J^{-1}]",
    ylabel=L"I(t)",
)

# curves

lines!(ax, d05.times, d05.I_mean, label="W=$(d05.W), Ndis=$(d05.Ndis)")
lines!(ax, d1.times, d1.I_mean, label="W=$(d1.W), Ndis=$(d1.Ndis)")
lines!(ax, d2.times, d2.I_mean, label="W=$(d2.W), Ndis=$(d2.Ndis)")
lines!(ax, d4.times, d4.I_mean, label="W=$(d4.W), Ndis=$(d4.Ndis)")

# error bands (semi-transparent)
band!(ax, d05.times, d05.I_mean .- d05.I_err, d05.I_mean .+ d05.I_err)
band!(ax, d1.times, d1.I_mean .- d1.I_err, d1.I_mean .+ d1.I_err)
band!(ax, d2.times, d2.I_mean .- d2.I_err, d2.I_mean .+ d2.I_err)
band!(ax, d4.times, d4.I_mean .- d4.I_err, d4.I_mean .+ d4.I_err)

axislegend(ax, position=:rb)

save("compare_imbalance_W05_W1_W2_W4.png", fig1)
println("Saved: compare_imbalance_W05_W1_W2_W4.png")

# # ----------------------------
# # Optional: plot n_mean heatmaps
# # n_mean is L x Nt; show as (site, time) heatmap
# # ----------------------------
# function plot_heatmap(d; outname)
#     L = d.L
#     t = d.times
#     nt = length(t)

#     fig = Figure(size=(850, 520))
#     ax = Axis(fig[1, 1],
#         xlabel="site j",
#         ylabel=L"t\,[J^{-1}]",
#         title="⟨n_j(t)⟩ disorder-avg (W=$(d.W), L=$(d.L), g=$(d.gpair))"
#     )

#     # Heatmap expects a matrix with y as rows and x as cols.
#     # We'll show time on y-axis: nt x L
#     hm = heatmap!(ax, 1:L, t, permutedims(d.n_mean, (2, 1)))

#     Colorbar(fig[1, 2], hm, label=L"\langle n_j\rangle")
#     fig
#     save(outname, fig)
#     println("Saved: $outname")
# end

# plot_heatmap(d2; outname="nmean_heatmap_W2.png")
# plot_heatmap(d4; outname="nmean_heatmap_W4.png")