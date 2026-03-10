###############################################################################
# Paper plots for Z3 parafermion dynamics
#
# Fig A:  P1(j,t) heatmaps
# Fig C:  Survival S(t) and mean-square radius R^2(t)
#
# Data sources:
#   - ED (exact diagonalization / Krylov)
#   - pTWA discrete sampling
#   - pTWA Gaussian sampling
#
# Cases:
#   α = 3.0   (short-range)
#   α = 1.5   (long-range)
###############################################################################

using JLD2
using Plots
using Statistics
using Plots.Measures 

default(
    linewidth = 2,
    framestyle = :box,
    guidefontsize = 12,
    tickfontsize = 10,
    legendfontsize = 10,
    size = (600, 420),
)

# ============================= File locations ================================

"""
Adjust these paths if needed.
Assumes filenames exactly as produced by your scripts.
"""

function filenames(α)
    return Dict(
        :ED => "ED_Z3_L15_alpha$(α)_initsingle.jld2",

        :pTWA_discrete =>
            "pTWA_parafermion_discrete_Z3_L15_alpha$(α)_single_AC.jld2",

        :pTWA_gaussian =>
            "pTWA_parafermion_gaussian_Z3_L15_alpha$(α)_single_AC.jld2"
    )
end

# ============================= Load helpers ==================================

"""
Load ED data
"""
function load_ED(fname)
    @load fname times Zt Pt P1t S R2 meta
    return times, P1t, S, R2
end

"""
Load pTWA data (Gaussian or discrete)
"""
function load_pTWA(fname)
    @load fname t P1t S R2 meta
    return t, P1t, S, R2
end

# ============================= Fig A =========================================

"""
Fig A: P1(j,t) heatmap
"""
function plot_figA_P1(t, P1; title_str, outfile)
    L = size(P1, 2)

    plt = heatmap(
        1:L, t, P1;
        xlabel = "site j",
        ylabel = "time t",
        title  = title_str,

        # --- critical fixes ---
        left_margin   = 1mm,
        bottom_margin = 1mm,
        right_margin  = 1mm,
        top_margin    = 1mm,

        guidefontsize = 14,
        tickfontsize  = 11,

        #colorbar_title = "P₁(j,t)",
        colorbar_titlefontsize = 13,
        colorbar_tickfontsize  = 10,
        colorbar_title_location = :right,
    )

    savefig(plt, outfile)
    println("Saved $outfile")
end


# ============================= Fig C =========================================

"""
Fig C: survival probability + mean-square radius
"""
function plot_figC(t,
                   S_ed, R2_ed,
                   S_d,  R2_d,
                   S_g,  R2_g;
                   title_str,

                   outfile)

    plt1 = plot(
        t, S_ed, label="ED", color=:black, linestyle=:solid,
         # --- critical fixes ---
        left_margin   = 3mm,
        bottom_margin = 3mm,
        right_margin  = 1mm,
        top_margin    = 1mm,
        xlabel="time t", ylabel="S(t)"
    )
    plot!(plt1, t, S_d, label="pTWA (discrete)", linestyle=:dash)
    plot!(plt1, t, S_g, label="pTWA (Gaussian)", linestyle=:dot)
    #title!(plt1, title_str)

    plt2 = plot(
        t, R2_ed, label="ED", color=:black, linestyle=:solid,
        xlabel="time t", ylabel="R²(t)"
    )
    plot!(plt2, t, R2_d, label="pTWA (discrete)", linestyle=:dash)
    plot!(plt2, t, R2_g, label="pTWA (Gaussian)", linestyle=:dot)

    plt = plot(plt1, plt2, layout=(1,2), size=(900,350))
    savefig(plt, outfile)
    println("Saved $outfile")
end

# ============================= Main driver ===================================

function make_all_plots(; L=16)

    for α in (3.0, 0.5)

        println("\n=== Plotting α = $α ===")

        files = filenames(α)

        # --- load data ---
        t_ed, P1_ed, S_ed, R2_ed = load_ED(files[:ED])
        t_d,  P1_d,  S_d,  R2_d  = load_pTWA(files[:pTWA_discrete])
        t_g,  P1_g,  S_g,  R2_g  = load_pTWA(files[:pTWA_gaussian])

        # --- Fig A ---
        plot_figA_P1(
            t_ed, P1_ed;
            title_str = "ED  P₁(j,t)  (L=$L, α=$α)",
            outfile   = "FigA_ED_P1_L$(L)_alpha$(α).png"
        )

        plot_figA_P1(
            t_d, P1_d;
            title_str = "pTWA (discrete)  P₁(j,t)  (L=$L, α=$α)",
            outfile   = "FigA_pTWA_discrete_P1_L$(L)_alpha$(α).png"
        )

        plot_figA_P1(
            t_g, P1_g;
            title_str = "pTWA (Gaussian)  P₁(j,t)  (L=$L, α=$α)",
            outfile   = "FigA_pTWA_gaussian_P1_L$(L)_alpha$(α).png"
        )

        # --- Fig C ---
        plot_figC(
            t_ed,
            S_ed, R2_ed,
            S_d,  R2_d,
            S_g,  R2_g;
            title_str = "Single-site excitation (L=$L, α=$α)",
            outfile   = "FigC_AC_L$(L)_alpha$(α).png"
        )
    end
end

# ============================= Run ===========================================

if abspath(PROGRAM_FILE) == @__FILE__
    make_all_plots(L=15)
end
