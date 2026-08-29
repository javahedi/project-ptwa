using CairoMakie
using JLD2
using LaTeXStrings

function make_plot()

    set_theme!(Theme(
        fontsize = 23,
        Axis = (
            xticklabelsize = 18,
            yticklabelsize = 18,
            xlabelsize = 28,
            ylabelsize = 28,
        )
    ))

    L = 13
    Ntraj = 10_000

    ed_dir =
        joinpath(@__DIR__,"data","ed")

    pt_dir =
        joinpath(@__DIR__,"data","ptwa")

    fig =
        Figure(
            size=(1000,900)
        )

    # ====================================================
    # Axes
    # ====================================================

    ax11 =
        Axis(
            fig[1,1],
            ylabel=L"P_{\mathrm{exc}}(j_0,t)"
        )

    ax12 =
        Axis(
            fig[1,2],
            ylabel=L"\bar{r}(t)"
        )

    ax21 =
        Axis(
            fig[2,1],
            ylabel=L"P_{\mathrm{exc}}(j_0,t)",
            limits=(0,nothing,0.9,1.001)
        )

    ax22 =
        Axis(
            fig[2,2],
            ylabel=L"\bar{r}(t)"
        )

    ax31 =
        Axis(
            fig[3,1],
            xlabel=L"Jt",
            ylabel=L"P_{\mathrm{exc}}(j_0,t)",
            limits=(0,nothing,0.99,1.001)
        )

    ax32 =
        Axis(
            fig[3,2],
            xlabel=L"Jt",
            ylabel=L"\bar{r}(t)"
        )

    # ====================================================
    # Labels inside panels
    # ====================================================

    text!(
        ax11,
        0.05,0.10,
        text=L"\alpha=3.0",
        space=:relative,
        fontsize=28
    )

    text!(
        ax12,
        0.05,0.60,
        text=L"\alpha=3.0",
        space=:relative,
        fontsize=28
    )

    text!(
        ax21,
        0.05,0.10,
        text=L"\alpha=1.5",
        space=:relative,
        fontsize=28
    )

    text!(
        ax22,
        0.40,0.10,
        text=L"\alpha=1.5",
        space=:relative,
        fontsize=28
    )

    text!(
        ax31,
        0.40,0.10,
        text=L"\alpha=0.5",
        space=:relative,
        fontsize=28
    )

    text!(
        ax32,
        0.40,0.10,
        text=L"\alpha=0.5",
        space=:relative,
        fontsize=28
    )

    # ====================================================
    # Load + plot
    # ====================================================

    function plot_alpha!(
        ax_exc,
        ax_disp,
        alpha;
        legend=false
    )

        # ---------- ED

        ed =
            load(
                joinpath(
                    ed_dir,
                    "ed_Z3_L$(L)_alpha$(alpha).jld2"
                )
            )

        lines!(
            ax_exc,
            ed["times"],
            ed["center_exc"],
            color=:black,
            linewidth=3,
            label=legend ? "ED" : nothing
        )

        lines!(
            ax_disp,
            ed["times"],
            ed["avg_disp"],
            color=:black,
            linewidth=3
        )

        # ---------- Gaussian

        gauss =
            load(
                joinpath(
                    pt_dir,
                    "ptwa_Z3_L$(L)_alpha$(alpha)_Ntraj$(Ntraj)_gaussian.jld2"
                )
            )

        errorbars!(
            ax_exc,
            gauss["times"],
            gauss["center_exc"],
            gauss["center_err"],
            color=:red,
            whiskerwidth=4
        )

        lines!(
            ax_exc,
            gauss["times"],
            gauss["center_exc"],
            color=:red,
            linestyle=:dash,
            label=legend ? "Gaussian" : nothing
        )

        errorbars!(
            ax_disp,
            gauss["times"],
            gauss["avg_disp"],
            gauss["disp_err"],
            color=:red,
            whiskerwidth=4
        )

        lines!(
            ax_disp,
            gauss["times"],
            gauss["avg_disp"],
            color=:red,
            linestyle=:dash
        )

        # ---------- Discrete

        discrete =
            load(
                joinpath(
                    pt_dir,
                    "ptwa_Z3_L$(L)_alpha$(alpha)_Ntraj$(Ntraj)_discrete.jld2"
                )
            )

        errorbars!(
            ax_exc,
            discrete["times"],
            discrete["center_exc"],
            discrete["center_err"],
            color=:blue,
            whiskerwidth=4
        )

        lines!(
            ax_exc,
            discrete["times"],
            discrete["center_exc"],
            color=:blue,
            linestyle=:dash,
            label=legend ? "Discrete" : nothing
        )

        errorbars!(
            ax_disp,
            discrete["times"],
            discrete["avg_disp"],
            discrete["disp_err"],
            color=:blue,
            whiskerwidth=4
        )

        lines!(
            ax_disp,
            discrete["times"],
            discrete["avg_disp"],
            color=:blue,
            linestyle=:dash
        )
    end

    plot_alpha!(
        ax11,
        ax12,
        3.0;
        legend=true
    )

    plot_alpha!(
        ax21,
        ax22,
        1.5
    )

    plot_alpha!(
        ax31,
        ax32,
        0.5
    )

    axislegend(
        ax11;
        framevisible=false
    )

    save(
        joinpath(
            @__DIR__,
            "Fig4_dynamics_Z3.png"
        ),
        fig,
        px_per_unit=2
    )

    save(
        joinpath(
            @__DIR__,
            "Fig4_dynamics_Z3.pdf"
        ),
        fig
    )

    println(
        "Saved Fig4_dynamics_Z3.png/pdf"
    )
end

make_plot()
