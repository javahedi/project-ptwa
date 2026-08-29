using CairoMakie
using JLD2
using LaTeXStrings

function make_plot()

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
    Ntraj = 10_000
    sampling = :discrete

    ed_dir = joinpath(@__DIR__, "data", "ed")
    pt_dir = joinpath(@__DIR__, "data", "ptwa")

    fig = Figure(size = (1100, 1000))

    # -------------------------------------------------
    # Axes
    # -------------------------------------------------

    ax11 = Axis(fig[1,1], ylabel = L"Jt")
    ax12 = Axis(fig[1,2])

    ax21 = Axis(fig[2,1], ylabel = L"Jt")
    ax22 = Axis(fig[2,2])

    ax31 = Axis(
        fig[3,1],
        xlabel = L"\mathrm{site}\ j",
        ylabel = L"Jt"
    )

    ax32 = Axis(
        fig[3,2],
        xlabel = L"\mathrm{site}\ j"
    )

    xt = 1:2:L

    for ax in (ax11,ax12,ax21,ax22,ax31,ax32)
        ax.xticks = xt
    end

    # -------------------------------------------------
    # alpha = 3.0
    # -------------------------------------------------

    α = 3.0

    ed = load(
        joinpath(
            ed_dir,
            "ed_Z3_L$(L)_alpha$(α).jld2"
        )
    )

    pt = load(
        joinpath(
            pt_dir,
            "ptwa_Z3_L$(L)_alpha$(α)_Ntraj$(Ntraj)_$(sampling).jld2"
        )
    )

    hm1 = heatmap!(
        ax11,
        1:L,
        ed["times"],
        ed["Pexc"],
        colormap = :blues,
        colorrange = (0,1)
    )

    heatmap!(
        ax12,
        1:L,
        pt["times"],
        pt["Pexc"],
        colormap = :blues,
        colorrange = (0,1)
    )

    text!(
        ax11,
        0.05, 0.10,
        text = L"\alpha=3.0",
        space = :relative,
        fontsize = 32,
        color = :black
    )

    text!(
        ax11,
        0.70, 0.10,
        text = L"\mathrm{Exact}",
        space = :relative,
        fontsize = 32,
        color = :black
    )

    text!(
        ax12,
        0.05, 0.10,
        text = L"\alpha=3.0",
        space = :relative,
        fontsize = 32,
        color = :black
    )

    text!(
        ax12,
        0.70, 0.10,
        text = L"p\mathrm{TWA}",
        space = :relative,
        fontsize = 32,
        color = :black
    )

    # -------------------------------------------------
    # alpha = 1.5
    # -------------------------------------------------

    α = 1.5

    ed = load(
        joinpath(
            ed_dir,
            "ed_Z3_L$(L)_alpha$(α).jld2"
        )
    )

    pt = load(
        joinpath(
            pt_dir,
            "ptwa_Z3_L$(L)_alpha$(α)_Ntraj$(Ntraj)_$(sampling).jld2"
        )
    )

    heatmap!(
        ax21,
        1:L,
        ed["times"],
        ed["Pexc"],
        colormap = :blues,
        colorrange = (0,1)
    )

    heatmap!(
        ax22,
        1:L,
        pt["times"],
        pt["Pexc"],
        colormap = :blues,
        colorrange = (0,1)
    )

    text!(
        ax21,
        0.05, 0.10,
        text = L"\alpha=1.5",
        space = :relative,
        fontsize = 32,
        color = :black
    )

    text!(
        ax21,
        0.70, 0.10,
        text = L"\mathrm{Exact}",
        space = :relative,
        fontsize = 32,
        color = :black
    )

    text!(
        ax22,
        0.05, 0.10,
        text = L"\alpha=1.5",
        space = :relative,
        fontsize = 32,
        color = :black
    )

    text!(
        ax22,
        0.70, 0.10,
        text = L"p\mathrm{TWA}",
        space = :relative,
        fontsize = 32,
        color = :black
    )

    # -------------------------------------------------
    # alpha = 0.5
    # -------------------------------------------------

    α = 0.5

    ed = load(
        joinpath(
            ed_dir,
            "ed_Z3_L$(L)_alpha$(α).jld2"
        )
    )

    pt = load(
        joinpath(
            pt_dir,
            "ptwa_Z3_L$(L)_alpha$(α)_Ntraj$(Ntraj)_$(sampling).jld2"
        )
    )

    heatmap!(
        ax31,
        1:L,
        ed["times"],
        ed["Pexc"],
        colormap = :blues,
        colorrange = (0,1)
    )

    heatmap!(
        ax32,
        1:L,
        pt["times"],
        pt["Pexc"],
        colormap = :blues,
        colorrange = (0,1)
    )

    text!(
        ax31,
        0.05, 0.10,
        text = L"\alpha=0.5",
        space = :relative,
        fontsize = 32,
        color = :black
    )

    text!(
        ax31,
        0.70, 0.10,
        text = L"\mathrm{Exact}",
        space = :relative,
        fontsize = 32,
        color = :black
    )

    text!(
        ax32,
        0.05, 0.10,
        text = L"\alpha=0.5",
        space = :relative,
        fontsize = 32,
        color = :black
    )

    text!(
        ax32,
        0.70, 0.10,
        text = L"p\mathrm{TWA}",
        space = :relative,
        fontsize = 32,
        color = :black
    )

    # -------------------------------------------------
    # Shared colorbar
    # -------------------------------------------------

    Colorbar(
        fig[:,3],
        hm1,
        label = L"P_{\mathrm{exc}}(j,t)"
    )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    save(
        joinpath(@__DIR__, "Fig1_heatmaps_Z3.png"),
        fig,
        px_per_unit = 2
    )

    save(
        joinpath(@__DIR__, "Fig1_heatmaps_Z3.pdf"),
        fig
    )

    println("Saved Fig1_heatmaps_Z3.png/pdf")
end

make_plot()