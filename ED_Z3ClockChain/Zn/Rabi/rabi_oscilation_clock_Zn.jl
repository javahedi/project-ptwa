# single_site_clock_rabi_compare.jl

using LinearAlgebra
using JLD2
using Printf
using Plots

# ------------------------------------------------------------
# Clock operators
# ------------------------------------------------------------

function clock_X(n::Int)
    X = zeros(ComplexF64, n, n)
    for a in 0:n-1
        ap = mod(a + 1, n)
        X[ap + 1, a + 1] = 1
    end
    return X
end

function single_site_hamiltonian(n::Int, g::Real)
    X = clock_X(n)
    return -g * (X + X')
end

# ------------------------------------------------------------
# Initial state
# ------------------------------------------------------------

function basis_state(n::Int, a0::Int)
    ψ0 = zeros(ComplexF64, n)
    ψ0[a0 + 1] = 1
    return ψ0
end

# ------------------------------------------------------------
# ED evolution
# ------------------------------------------------------------

function evolve_exact(H, ψ0, times)

    F = eigen(H)
    V = F.vectors
    E = F.values

    c0 = V' * ψ0

    ψs = Vector{Vector{ComplexF64}}(undef, length(times))

    for (k,t) in enumerate(times)
        phase = exp.(-1im .* E .* t)
        ψs[k] = V * (phase .* c0)
    end

    return ψs
end

function populations(ψs)

    n = length(ψs[1])
    Nt = length(ψs)

    P = zeros(Float64, n, Nt)

    for k in 1:Nt
        for a in 1:n
            P[a,k] = abs2(ψs[k][a])
        end
    end

    return P
end

# ------------------------------------------------------------
# Analytical solution
# ------------------------------------------------------------

function analytic_amplitude(r, t, n, g)

    s = 0.0 + 0.0im

    for m in 0:n-1
        k = 2π*m/n
        s += exp(1im * 2g * t * cos(k)) * exp(1im * k * r)
    end

    return s / n
end


function analytic_probabilities(times, n, g, a0)

    Nt = length(times)

    P = zeros(Float64, n, Nt)

    for (k,t) in enumerate(times)

        for a in 0:n-1

            r = mod(a - a0, n)

            A = analytic_amplitude(r, t, n, g)

            P[a+1, k] = abs2(A)

        end
    end

    return P
end

# ------------------------------------------------------------
# Plot comparison
# ------------------------------------------------------------

function plot_compare(times, P_ed, P_an, n, a0, g, outname)

    plt = plot(
        xlabel="t",
        ylabel="Probability",
        title="ED vs Analytical  (n=$n)",
        lw=2
    )

    plot!(plt, times, P_ed[a0+1,:], label="ED return")
    plot!(plt, times, P_an[a0+1,:], label="Analytic return", ls=:dash)

    savefig(plt, outname)

end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

function run_scan(; ns=[3,10,100],
                    g=1.0,
                    a0=0,
                    tmax=20,
                    Nt=1001,
                    outdir="Zn_rabi_compare")

    mkpath(outdir)

    times = collect(range(0,tmax,length=Nt))

    for n in ns

        println("\nRunning n=$n")

        # ED
        H = single_site_hamiltonian(n,g)
        ψ0 = basis_state(n,a0)

        ψs = evolve_exact(H,ψ0,times)

        P_ed = populations(ψs)

        # Analytical
        P_an = analytic_probabilities(times,n,g,a0)

        # Error
        err = maximum(abs.(P_ed .- P_an))

        println("Max error ED vs analytic = ", err)

        # Save
        outfile = joinpath(outdir,"Zn_$n.jld2")

        @save outfile n g times P_ed P_an err

        # Plot
        figfile = joinpath(outdir,"compare_n_$n.png")

        plot_compare(times,P_ed,P_an,n,a0,g,figfile)

    end

end


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

run_scan(
    ns=[3,10,100],
    g=1.0,
    a0=0,
    tmax=20,
    Nt=1001
)