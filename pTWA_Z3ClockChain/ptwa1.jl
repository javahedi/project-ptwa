using LinearAlgebra
using JLD2
using Random

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
# Gaussian Wigner sampling for product basis states |a0>
# - mean: μ^{ab} = δ_{a,a0}δ_{b,a0}
# - off-diagonals: Re/Im sampled iid with variance (μ_aa + μ_bb)/4
# - enforce Hermiticity: x^{ba} = (x^{ab})*
# ----------------------------
function sample_site_gaussian(a0::Int; rng::AbstractRNG=Random.default_rng())
    x = zeros(ComplexF64, 3, 3)

    μ = zeros(Float64, 3)
    μ[a0+1] = 1.0
    x[a0+1, a0+1] = 1.0 + 0im

    for a in 1:3
        for b in a+1:3
            σ = sqrt((μ[a] + μ[b]) / 4)

            if σ > 0
                re = σ * randn(rng)
                im = σ * randn(rng)
                x[a,b] = re + 1im*im
                x[b,a] = re - 1im*im
            end
        end
    end

    return x
end

function sample_initial_state_gaussian(L::Int, j0::Int; aexc::Int=1, rng::AbstractRNG=Random.default_rng())
    x0 = Vector{Matrix{ComplexF64}}(undef, L)
    for j in 1:L
        if j == j0
            x0[j] = sample_site_gaussian(aexc; rng=rng)  # |1>
        else
            x0[j] = sample_site_gaussian(0; rng=rng)     # |0>
        end
    end
    return x0
end

# Measurements
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

function average_displacement(Pa_all::AbstractVector{<:Real}, j0::Int)
    s = 0.0
    for j in eachindex(Pa_all)
        s += abs(j - j0) * Pa_all[j]
    end
    return s
end

# ----------------------------
# Run: match ED parameters + sampling
# ----------------------------
L  = 13
α  = 3.0
J  = 1.0
g  = 0.5

Ntraj = 1000          # <-- set to e.g. 200–500 for quick tests; 2000+ for figures
seed  = 1234         # reproducible

Jij = build_Jij(L, α, J)
p   = PTWAParams(L, α, J, g, Jij)

j0 = (L + 1) ÷ 2
times = collect(range(0.0, 5.0, length=51))
Nt = length(times)

P1_avg = zeros(Float64, L, Nt)
center_decay = zeros(Float64, Nt)
avg_disp     = zeros(Float64, Nt)

rng = MersenneTwister(seed)

for tr in 1:Ntraj
    x0 = sample_initial_state_gaussian(L, j0; aexc=1, rng=rng)
    xs = evolve_times_ptwa(x0, p, times)
    P1 = measure_P1_matrix_traj(xs, L)

    P1_avg .+= P1

    if tr % 50 == 0
        println("pTWA traj $tr / $Ntraj")
    end
end

P1_avg ./= Ntraj

for k in 1:Nt
    center_decay[k] = P1_avg[j0, k]
    avg_disp[k] = average_displacement(view(P1_avg, :, k), j0)
end

println("pTWA (Gaussian) Center decay P1(j0,t_final) = ", center_decay[end])
println("pTWA (Gaussian) Avg displacement <|j-j0|>(t_final) = ", avg_disp[end])

outfile = "pTWA_Z3_L$(L)_alpha$(α)_g$(g)_single_excitation_gaussian_N$(Ntraj).jld2"
@save outfile L α J g j0 times Ntraj seed P1_avg center_decay avg_disp

println("Saved to: ", outfile)