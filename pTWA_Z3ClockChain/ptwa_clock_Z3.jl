# pTWA for Z3 clock chain in Hubbard (matrix) variables
using LinearAlgebra
using JLD2
using Random

include("sampling.jl")
using .Sampling

# ----------------------------
# Choose sampling method
# ----------------------------
#sampling = :gaussian
sampling = :discrete

# Discrete cache (if needed)
cache = nothing
if sampling == :discrete
    cache = build_discrete_cache()
end


# ----------------------------
# Check initial means
# ----------------------------
function check_initial_means(sampling, L, j0; cache=cache, Ntest=2000, seed=1)
    rng = MersenneTwister(seed)
    m_center11 = 0.0
    m_other00  = 0.0
    m_offdiag  = 0.0
    for _ in 1:Ntest
        x0 = sample_initial_state(sampling, L, j0; cache=cache, aexc=1, rng=rng)
        m_center11 += real(x0[j0][2,2])
        m_other00  += real(x0[1][1,1])
        m_offdiag  += abs(x0[j0][1,2])
    end
    println("⟨x_center(11)⟩ = ", m_center11/Ntest, "  (should be 1)")
    println("⟨x_other(00)⟩  = ", m_other00/Ntest,  "  (should be 1)")
    println("⟨|x_center(01)|⟩ = ", m_offdiag/Ntest, "  (should be ~0 for discrete, >0 for Gaussian)")
end

# ----------------------------
# pTWA for Z3 clock chain in Hubbard (matrix) variables
# ----------------------------
struct PTWAParams
    L::Int
    alpha::Float64
    J::Float64
    g::Float64
    Jij::Matrix{Float64}  # full matrix with Jij[i,j] (i!=j), open chain
end

const n = 3
const ω = cis(2π / n)

function build_Jij(L::Int, alpha::Float64, J::Float64)
    Jij = zeros(Float64, L, L)
    for i in 1:L, j in 1:L
        if i != j
            Jij[i, j] = J / (abs(i - j)^alpha)
        end
    end
    return Jij
end

@inline function Z_of(xj::Matrix{ComplexF64})
    return (ω^0)*xj[1,1] + (ω^1)*xj[2,2] + (ω^2)*xj[3,3]
end

function build_h!(h::Vector{Matrix{ComplexF64}}, x::Vector{Matrix{ComplexF64}}, p::PTWAParams)
    L = p.L
    Jij = p.Jij

    Z  = Vector{ComplexF64}(undef, L)
    Zd = Vector{ComplexF64}(undef, L)
    for k in 1:L
        Z[k]  = Z_of(x[k])
        Zd[k] = conj(Z[k])
    end

    for j in 1:L
        hj = h[j]
        fill!(hj, 0.0 + 0.0im)

        # ZZ: diagonal only
        for a in 0:2
            s = 0.0
            wa = ω^a
            for k in 1:L
                k == j && continue
                s += Jij[j,k] * real(wa * Zd[k])
            end
            hj[a+1, a+1] += -2s
        end

        # X field: cyclic nearest off-diagonals
        hj[2,1] += -p.g
        hj[3,2] += -p.g
        hj[1,3] += -p.g
        hj[1,2] += -p.g
        hj[2,3] += -p.g
        hj[3,1] += -p.g
    end
    return nothing
end

function rhs!(dx::Vector{Matrix{ComplexF64}}, x::Vector{Matrix{ComplexF64}},
              h::Vector{Matrix{ComplexF64}}, p::PTWAParams)
    build_h!(h, x, p)
    L = p.L
    for j in 1:L
        dx[j] .= 1im * (h[j]*x[j] - x[j]*h[j])
    end
    return nothing
end

function evolve_times_ptwa(x0::Vector{Matrix{ComplexF64}}, p::PTWAParams, times::Vector{Float64})
    L = p.L
    x = [copy(x0[j]) for j in 1:L]

    # workspace
    h  = [zeros(ComplexF64, 3,3) for _ in 1:L]
    k1 = [zeros(ComplexF64, 3,3) for _ in 1:L]
    k2 = [zeros(ComplexF64, 3,3) for _ in 1:L]
    k3 = [zeros(ComplexF64, 3,3) for _ in 1:L]
    k4 = [zeros(ComplexF64, 3,3) for _ in 1:L]
    xt = [zeros(ComplexF64, 3,3) for _ in 1:L]

    out = Vector{Vector{Matrix{ComplexF64}}}(undef, length(times))
    out[1] = [copy(x[j]) for j in 1:L]

    for k in 2:length(times)
        dt = times[k] - times[k-1]

        rhs!(k1, x, h, p)

        for j in 1:L
            xt[j] .= x[j] .+ (dt/2) .* k1[j]
        end
        rhs!(k2, xt, h, p)

        for j in 1:L
            xt[j] .= x[j] .+ (dt/2) .* k2[j]
        end
        rhs!(k3, xt, h, p)

        for j in 1:L
            xt[j] .= x[j] .+ dt .* k3[j]
        end
        rhs!(k4, xt, h, p)

        for j in 1:L
            x[j] .+= (dt/6) .* (k1[j] .+ 2k2[j] .+ 2k3[j] .+ k4[j])
        end

        out[k] = [copy(x[j]) for j in 1:L]
    end

    return out
end

# ----------------------------
# Measurements
# ----------------------------
function measure_P1_matrix_traj(xs::Vector{Vector{Matrix{ComplexF64}}}, L::Int)
    Nt = length(xs)
    P1 = zeros(Float64, L, Nt)
    for k in 1:Nt
        for j in 1:L
            P1[j,k] = real(xs[k][j][2,2])  # a=1 -> index 2
        end
    end
    return P1
end

function measure_Pa_matrix_traj(xs::Vector{Vector{Matrix{ComplexF64}}}, L::Int, a::Int)
    Nt = length(xs)
    Pa = zeros(Float64, L, Nt)
    ia = a + 1
    for k in 1:Nt
        for j in 1:L
            Pa[j,k] = real(xs[k][j][ia, ia])
        end
    end
    return Pa
end

function average_displacement(Pa_all::AbstractVector{<:Real}, j0::Int)
    s = 0.0
    for j in eachindex(Pa_all)
        s += abs(j - j0) * Pa_all[j]
    end
    return s
end

# ----------------------------
# Run: match ED parameters + sampling + error bars
# ----------------------------
L  = 13
α  = 3.0
J  = 1.0
g  = 0.5

Ntraj = 10000
seed  = 1234

Jij = build_Jij(L, α, J)
p   = PTWAParams(L, α, J, g, Jij)

j0 = (L + 1) ÷ 2
times = collect(range(0.0, 5.0, length=101))
Nt = length(times)




# accumulators for mean and error
# accumulators for mean and error
Pexc_sum  = zeros(Float64, L, Nt)
Pexc_sum2 = zeros(Float64, L, Nt)

center_sum  = zeros(Float64, Nt)
center_sum2 = zeros(Float64, Nt)

disp_sum  = zeros(Float64, Nt)
disp_sum2 = zeros(Float64, Nt)

rng = MersenneTwister(seed)

for tr in 1:Ntraj

    x0 = sample_initial_state(sampling, L, j0; cache=cache, aexc=1, rng=rng)

    xs = evolve_times_ptwa(x0, p, times)

    # populations
    P1 = measure_Pa_matrix_traj(xs, L, 1)
    P2 = measure_Pa_matrix_traj(xs, L, 2)

    Pexc = P1 .+ P2   # excitation density

    # accumulate mean and variance
    Pexc_sum  .+= Pexc
    Pexc_sum2 .+= Pexc.^2

    # center decay
    center = view(Pexc, j0, :)
    center_sum  .+= center
    center_sum2 .+= center.^2

    # displacement
    disp = zeros(Float64, Nt)
    for k in 1:Nt
        disp[k] = average_displacement(view(Pexc, :, k), j0)
    end

    disp_sum  .+= disp
    disp_sum2 .+= disp.^2

    if tr % 50 == 0
        println("pTWA traj $tr / $Ntraj")
    end
end

# ----------------------------
# Means
# ----------------------------

Pexc_avg = Pexc_sum ./ Ntraj
center_exc = center_sum ./ Ntraj
avg_disp = disp_sum ./ Ntraj

# ----------------------------
# Standard errors (SEM)
# ----------------------------

Pexc_var = (Pexc_sum2 ./ Ntraj) .- Pexc_avg.^2
Pexc_err = sqrt.(max.(Pexc_var, 0.0) ./ Ntraj)

center_var = (center_sum2 ./ Ntraj) .- center_exc.^2
center_err = sqrt.(max.(center_var, 0.0) ./ Ntraj)

disp_var = (disp_sum2 ./ Ntraj) .- avg_disp.^2
disp_err = sqrt.(max.(disp_var, 0.0) ./ Ntraj)

println("pTWA ($(sampling)) Center excitation = ",
        center_exc[end], " ± ", center_err[end])

println("pTWA ($(sampling)) Avg displacement = ",
        avg_disp[end], " ± ", disp_err[end])

# ----------------------------
# Save
# ----------------------------

outfile = "pTWA_Z3_L$(L)_alpha$(α)_g$(g)_single_excitation_$(sampling)_N$(Ntraj).jld2"

@save outfile L α J g j0 times Ntraj seed sampling Pexc_avg Pexc_err center_exc center_err avg_disp disp_err

println("Saved to: ", outfile)