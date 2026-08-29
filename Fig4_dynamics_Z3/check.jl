using JLD2
using Printf

const L = 13
const j0 = 7
const alphas = [3.0,1.5,0.5]
const Ntraj = 10_000
const samplings = [:gaussian,:discrete]

ed_dir =
    joinpath(@__DIR__,"data","ed")

pt_dir =
    joinpath(@__DIR__,"data","ptwa")

for alpha in alphas

    println()
    println("alpha = $alpha")

    ed =
        load(
            joinpath(
                ed_dir,
                "ed_Z3_L$(L)_alpha$(alpha).jld2"
            )
        )

    @assert abs(ed["center_exc"][1]-1) < 1e-12
    @assert abs(ed["avg_disp"][1]) < 1e-12
    @assert maximum(abs.(ed["normψ"].-1)) < 1e-8

    for sampling in samplings

        pt =
            load(
                joinpath(
                    pt_dir,
                    "ptwa_Z3_L$(L)_alpha$(alpha)_Ntraj$(Ntraj)_$(sampling).jld2"
                )
            )

        @assert ed["times"] == pt["times"]

        center0 =
            pt["center_exc"][1]

        disp0 =
            pt["avg_disp"][1]

        Δcenter =
            maximum(
                abs.(
                    ed["center_exc"] -
                    pt["center_exc"]
                )
            )

        Δdisp =
            maximum(
                abs.(
                    ed["avg_disp"] -
                    pt["avg_disp"]
                )
            )

        @printf(
            "  %-8s  center(0)=%.8f  disp(0)=%.3e  maxΔcenter=%.6e  maxΔdisp=%.6e\n",
            string(sampling),
            center0,
            disp0,
            Δcenter,
            Δdisp
        )
    end
end

println()
println("All structural checks passed.")
