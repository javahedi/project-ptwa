# pTWA for fully connected Z_n clock model (LMG-type)
#
# Hamiltonian:
#   H = -(J/N) sum_{i<j} (Z_i Z_j† + Z_i† Z_j) - g sum_i (X_i + X_i†)
#
# Initial state:
#   fully polarized product state |0>^{⊗ N}
#
# Observables saved:
#   mZ(t)        = (1/N) sum_j <Z_j>
#   |mZ(t)|
#   p_a(t)       = (1/N) sum_j <X_j^{aa}>
#   SEM error bars over trajectories

using LinearAlgebra
using JLD2
using Random

include("sampling_Zn.jl")
using .Sampling

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

struct PTWAParams
    N::Int
    n::Int
    J::Float64
    g::Float64
    ω::ComplexF64
end

# ------------------------------------------------------------
# Local observables
# ------------------------------------------------------------

@inline function Z_of(xj::AbstractMatrix{ComplexF64}, ω::ComplexF64, n::Int)
    z = 0.0 + 0.0im
    @inbounds for a in 0:n-1
        z += (ω^a) * xj[a+1, a+1]
    end
    return z
end

# ------------------------------------------------------------
# Build local mean-field matrices h_j
# ------------------------------------------------------------
# Exact finite-N fully connected field:
#   h_j,aa = -(J/N) sum_{k != j} [ ω^a z_k^* + ω^{-a} z_k ]
#
# plus drive term:
#   -g (X + X†)
# ------------------------------------------------------------

function build_h!(h::Vector{Matrix{ComplexF64}},
                  x::Vector{Matrix{ComplexF64}},
                  p::PTWAParams)

    N, n, J, g, ω = p.N, p.n, p.J, p.g, p.ω

    z = Vector{ComplexF64}(undef, N)
    zsum = 0.0 + 0.0im

    @inbounds for j in 1:N
        zj = Z_of(x[j], ω, n)
        z[j] = zj
        zsum += zj
    end

    @inbounds for j in 1:N
        hj = h[j]
        fill!(hj, 0.0 + 0.0im)

        # exclude self-interaction
        zex = zsum - z[j]

        # diagonal interaction term
        for a in 0:n-1
            wa = ω^a
            hj[a+1, a+1] = -(J / N) * (wa * conj(zex) + conj(wa) * zex)
        end

        # drive term: -g (X + X†)
        for a in 0:n-1
            b = mod(a + 1, n)
            hj[b+1, a+1] += -g      # X
            hj[a+1, b+1] += -g      # X†
        end
    end

    return nothing
end

# ------------------------------------------------------------
# RHS: dx_j/dt = i [h_j, x_j]
# ------------------------------------------------------------

function rhs!(dx::Vector{Matrix{ComplexF64}},
              x::Vector{Matrix{ComplexF64}},
              h::Vector{Matrix{ComplexF64}},
              p::PTWAParams)

    build_h!(h, x, p)

    @inbounds for j in 1:p.N
        dx[j] .= 1im .* (h[j] * x[j] - x[j] * h[j])
    end

    return nothing
end

# ------------------------------------------------------------
# RK4 time evolution
# ------------------------------------------------------------

function evolve_times_ptwa(x0::Vector{Matrix{ComplexF64}},
                           p::PTWAParams,
                           times::Vector{Float64})

    N, n = p.N, p.n
    x = [copy(x0[j]) for j in 1:N]

    h  = [zeros(ComplexF64, n, n) for _ in 1:N]
    k1 = [zeros(ComplexF64, n, n) for _ in 1:N]
    k2 = [zeros(ComplexF64, n, n) for _ in 1:N]
    k3 = [zeros(ComplexF64, n, n) for _ in 1:N]
    k4 = [zeros(ComplexF64, n, n) for _ in 1:N]
    xt = [zeros(ComplexF64, n, n) for _ in 1:N]

    out = Vector{Vector{Matrix{ComplexF64}}}(undef, length(times))
    out[1] = [copy(x[j]) for j in 1:N]

    for k in 2:length(times)
        dt = times[k] - times[k-1]

        rhs!(k1, x, h, p)

        @inbounds for j in 1:N
            xt[j] .= x[j] .+ (dt/2) .* k1[j]
        end
        rhs!(k2, xt, h, p)

        @inbounds for j in 1:N
            xt[j] .= x[j] .+ (dt/2) .* k2[j]
        end
        rhs!(k3, xt, h, p)

        @inbounds for j in 1:N
            xt[j] .= x[j] .+ dt .* k3[j]
        end
        rhs!(k4, xt, h, p)

        @inbounds for j in 1:N
            x[j] .+= (dt/6) .* (k1[j] .+ 2 .* k2[j] .+ 2 .* k3[j] .+ k4[j])

            # enforce Hermiticity gently against numerical drift
            x[j] .= 0.5 .* (x[j] .+ x[j]')

            # normalize trace for numerical stability
            trj = real(tr(x[j]))
            if abs(trj) > 1e-12
                x[j] ./= trj
            end
        end

        out[k] = [copy(x[j]) for j in 1:N]
    end

    return out
end

# ------------------------------------------------------------
# Measurements for one trajectory
# ------------------------------------------------------------

function measure_observables_traj(xs::Vector{Vector{Matrix{ComplexF64}}}, N::Int, n::Int, ω::ComplexF64)

    Nt = length(xs)

    mZ_t = zeros(ComplexF64, Nt)
    abs_mZ_t = zeros(Float64, Nt)
    pa_t = zeros(Float64, n, Nt)

    @inbounds for k in 1:Nt
        mz = 0.0 + 0.0im

        for j in 1:N
            xj = xs[k][j]

            # populations
            for a in 0:n-1
                pa_t[a+1, k] += real(xj[a+1, a+1]) / N
            end

            # clock order parameter
            mz += Z_of(xj, ω, n) / N
        end

        mZ_t[k] = mz
        abs_mZ_t[k] = abs(mz)
    end

    return mZ_t, abs_mZ_t, pa_t
end

# ------------------------------------------------------------
# Main scan
# ------------------------------------------------------------

function run_scan(; Ns = [10, 20, 30, 40],
                    ns = [2, 3, 4, 5, 6],
                    J = 1.0,
                    g = 0.5,
                    Ntraj = 2000,
                    seed = 1234,
                    times = collect(range(0.0, 5.0, length=101)),
                    outdir = "pTWA_LMG_Zn")

    mkpath(outdir)
    rng = MersenneTwister(seed)

    for N in Ns, n in ns

        println("\nRunning pTWA for N=$N, n=$n")

        ω = cis(2π / n)
        p = PTWAParams(N, n, J, g, ω)
        Nt = length(times)

        # accumulators
        mZ_sum      = zeros(ComplexF64, Nt)
        mZ_sum2_abs = zeros(Float64, Nt)   # for |mZ| variance estimate

        abs_mZ_sum  = zeros(Float64, Nt)
        abs_mZ_sum2 = zeros(Float64, Nt)

        pa_sum      = zeros(Float64, n, Nt)
        pa_sum2     = zeros(Float64, n, Nt)

        for tr in 1:Ntraj

            # Match ED initial state: fully polarized |0>^{⊗ N}
            x0 = sample_initial_state(:gaussian, n, N;
                                      state = :polarized,
                                      a0 = 0,
                                      rng = rng)

            xs = evolve_times_ptwa(x0, p, times)

            mZ_t, abs_mZ_t, pa_t = measure_observables_traj(xs, N, n, ω)

            mZ_sum .+= mZ_t
            abs_mZ_sum .+= abs_mZ_t
            abs_mZ_sum2 .+= abs_mZ_t .^ 2

            pa_sum .+= pa_t
            pa_sum2 .+= pa_t .^ 2

            if tr % 50 == 0
                println("  traj $tr / $Ntraj")
            end
        end

        # means
        mZ_avg = mZ_sum ./ Ntraj
        abs_mZ_avg = abs_mZ_sum ./ Ntraj
        pa_avg = pa_sum ./ Ntraj

        # standard errors
        abs_mZ_var = abs_mZ_sum2 ./ Ntraj .- abs_mZ_avg .^ 2
        abs_mZ_err = sqrt.(max.(abs_mZ_var, 0.0) ./ Ntraj)

        pa_var = pa_sum2 ./ Ntraj .- pa_avg .^ 2
        pa_err = sqrt.(max.(pa_var, 0.0) ./ Ntraj)

        println("  final |mZ| = ", abs_mZ_avg[end], " ± ", abs_mZ_err[end])

        outfile = joinpath(outdir, "pTWA_LMG_Zn_N$(N)_n$(n)_g$(g)_Ntraj$(Ntraj).jld2")

        @save outfile N n J g Ntraj seed times mZ_avg abs_mZ_avg abs_mZ_err pa_avg pa_err

        println("  saved -> ", outfile)
    end
end

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

run_scan(
    Ns = [60,70,80],
    ns = [5],
    J = 1.0,
    g = 0.5,
    Ntraj = 2000,
    seed = 1234,
    times = collect(range(0.0, 5.0, length=101)),
    outdir = "pTWA_LMG_Zn"
)