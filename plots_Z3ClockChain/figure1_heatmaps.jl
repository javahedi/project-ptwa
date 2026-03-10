using CairoMakie
using JLD2
using LaTeXStrings

set_theme!(Theme(
    fontsize = 28,
    Axis = (
        xticklabelsize = 18,
        yticklabelsize = 18,
        xlabelsize = 28,
        ylabelsize = 28
    )
))

L = 13
fig = Figure(size = (1100, 1000))

# -------------------------------------------------
# Axes (3 rows now)
# -------------------------------------------------

ax11 = Axis(fig[1,1], ylabel="time")      # α=3 ED
ax12 = Axis(fig[1,2])                     # α=3 pTWA

ax21 = Axis(fig[2,1], ylabel="time")      # α=1.5 ED
ax22 = Axis(fig[2,2])                     # α=1.5 pTWA

ax31 = Axis(fig[3,1], xlabel="site j", ylabel="time")   # α=0.5 ED
ax32 = Axis(fig[3,2], xlabel="site j")                  # α=0.5 pTWA

xt = 1:2:L

for ax in (ax11,ax12,ax21,ax22,ax31,ax32)
    ax.xticks = xt
end

###################################################
# α = 3.0
###################################################

α = 3.0

@load "../ED_Z3ClockChain/ED_Z3_L13_alpha$(α)_g0.5_single_excitation.jld2" times Pexc
hm1 = heatmap!(ax11, 1:L, times, Pexc,
    colormap=:blues,
    colorrange=(0,1)
)

@load "../pTWA_Z3ClockChain/pTWA_Z3_L13_alpha$(α)_g0.5_single_excitation_gaussian_N10000.jld2" times Pexc_avg
heatmap!(ax12, 1:L, times, Pexc_avg,
    colormap=:blues,
    colorrange=(0,1)
)

text!(ax11,0.05,0.2,text=L"\alpha=3.0,~~ED",space=:relative,fontsize=28,color=:white)
text!(ax12,0.05,0.2,text=L"\alpha=3.0,~~pTWA",space=:relative,fontsize=28,color=:white)

###################################################
# α = 1.5  (NEW)
###################################################

α = 1.5

@load "../ED_Z3ClockChain/ED_Z3_L13_alpha$(α)_g0.5_single_excitation.jld2" times Pexc
heatmap!(ax21, 1:L, times, Pexc,
    colormap=:blues,
    colorrange=(0,1)
)

@load "../pTWA_Z3ClockChain/pTWA_Z3_L13_alpha$(α)_g0.5_single_excitation_gaussian_N10000.jld2" times Pexc_avg
heatmap!(ax22, 1:L, times, Pexc_avg,
    colormap=:blues,
    colorrange=(0,1)
)

text!(ax21,0.05,0.2,text=L"\alpha=1.5,~~ED",space=:relative,fontsize=28,color=:white)
text!(ax22,0.05,0.2,text=L"\alpha=1.5,~~pTWA",space=:relative,fontsize=28,color=:white)

###################################################
# α = 0.5
###################################################

α = 0.5

@load "../ED_Z3ClockChain/ED_Z3_L13_alpha$(α)_g0.5_single_excitation.jld2" times Pexc
heatmap!(ax31, 1:L, times, Pexc,
    colormap=:blues,
    colorrange=(0,1)
)

@load "../pTWA_Z3ClockChain/pTWA_Z3_L13_alpha$(α)_g0.5_single_excitation_gaussian_N10000.jld2" times Pexc_avg
heatmap!(ax32, 1:L, times, Pexc_avg,
    colormap=:blues,
    colorrange=(0,1)
)

text!(ax31,0.05,0.2,text=L"\alpha=0.5,~~ED",space=:relative,fontsize=28,color=:white)
text!(ax32,0.05,0.2,text=L"\alpha=0.5,~~pTWA",space=:relative,fontsize=28,color=:white)

###################################################
# Shared colorbar
###################################################

Colorbar(fig[:,3], hm1)

###################################################
# Save
###################################################

save("Fig1_heatmaps_Z3.png", fig)
save("Fig1_heatmaps_Z3.pdf", fig)
