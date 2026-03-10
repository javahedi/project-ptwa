# ed_clockChain_Z3.jl

using LinearAlgebra
using LinearMaps
using KrylovKit
using JLD2

# ----------------------------
# Model: Z_n clock chain
# H = - ∑_{i<j} J_ij (Z_i Z_j† + h.c.)  - g ∑_i (X_i + X_i†)
# In Z-basis: interaction is diagonal; X shifts the local state a_j -> a_j±1 (mod n)
# ----------------------------

struct ClockLRParams
    n::Int
    L::Int
    alpha::Float64
    J::Float64
    g::Float64
    pairs::Vector{Tuple{Int,Int,Float64}}  # (i,j,Jij)  1-based sites
    powN::Vector{Int}                      # powN[s] = n^(s-1)
    cos2π::Matrix{Float64}                 # cos2π[a,b] = 2cos(2π(a-b)/n) with a,b in 0..n-1
end

@inline function digit_of(state::Int, site::Int, powN::Vector{Int}, n::Int)
    return (state ÷ powN[site]) % n
end

@inline function shift_plus(state::Int, site::Int, powN::Vector{Int}, n::Int)
    a  = mod(state ÷ powN[site], n)
    ap = mod(a + 1, n)
    return state + (ap - a) * powN[site]
end

@inline function shift_minus(state::Int, site::Int, powN::Vector{Int}, n::Int)
    a  = mod(state ÷ powN[site], n)
    am = mod(a - 1, n)
    return state + (am - a) * powN[site]
end

@inline function diag_energy(state::Int, p::ClockLRParams)
    E = 0.0
    n = p.n
    for (i, j, Jij) in p.pairs
        ai = digit_of(state, i, p.powN, n)
        aj = digit_of(state, j, p.powN, n)
        E += -Jij * p.cos2π[ai+1, aj+1]
    end
    return E
end

function mul_H!(y::Vector{ComplexF64}, v::Vector{ComplexF64}, p::ClockLRParams)
    fill!(y, 0.0 + 0.0im)
    dim = length(v)
    n, L = p.n, p.L

    @inbounds for idx in 1:dim
        s  = idx - 1
        vs = v[idx]

        # diagonal interaction
        y[idx] += diag_energy(s, p) * vs

        # transverse field: -g (X_i + X_i†)
        for site in 1:L
            sp = shift_plus(s, site, p.powN, n)
            sm = shift_minus(s, site, p.powN, n)
            y[sp + 1] += -p.g * vs
            y[sm + 1] += -p.g * vs
        end
    end
    return y
end

function build_pairs(L::Int, alpha::Float64, J::Float64)
    pairs = Tuple{Int,Int,Float64}[]
    for i in 1:L-1
        for j in i+1:L
            dist = abs(i - j)
            Jij = J / (dist^alpha)
            push!(pairs, (i, j, Jij))
        end
    end
    return pairs
end

function make_params(n::Int, L::Int; alpha=3.0, J=1.0, g=0.5)
    powN = [n^(s-1) for s in 1:L]
    pairs = build_pairs(L, alpha, J)

    cos2π = zeros(Float64, n, n)
    for a in 0:n-1, b in 0:n-1
        cos2π[a+1, b+1] = 2 * cos(2π * (a - b) / n)
    end

    return ClockLRParams(n, L, float(alpha), float(J), float(g), pairs, powN, cos2π)
end

# ----------------------------
# Measurements
# ----------------------------

function measure_Pa_site(psi::Vector{ComplexF64}, p::ClockLRParams, site::Int)
    n = p.n
    Pa = zeros(Float64, n)
    @inbounds for idx in 1:length(psi)
        s = idx - 1
        a = digit_of(s, site, p.powN, n)
        Pa[a+1] += abs2(psi[idx])
    end
    return Pa
end

function measure_Pa_all_sites(psi::Vector{ComplexF64}, p::ClockLRParams, a::Int)
    L = p.L
    out = zeros(Float64, L)
    for j in 1:L
        out[j] = measure_Pa_site(psi, p, j)[a+1]
    end
    return out
end

function average_displacement(Pa_all::AbstractVector{<:Real}, j0::Int)
    s = 0.0
    for j in eachindex(Pa_all)
        s += abs(j - j0) * Pa_all[j]
    end
    return s
end

# ----------------------------
# Time evolution
# ----------------------------
function evolve_times(psi0::Vector{ComplexF64}, p::ClockLRParams, times::Vector{Float64};
                      krylovdim::Int=40, tol::Float64=1e-10)
    dim = length(psi0)

    H = LinearMap{ComplexF64}(
        (y, v) -> mul_H!(y, v, p),
        dim, dim;
        ismutating=true
    )

    psit = copy(psi0)
    out = Vector{Vector{ComplexF64}}(undef, length(times))
    tprev = 0.0

    for (k, t) in enumerate(times)
        dt = t - tprev
        psit, = exponentiate(H, -1im*dt, psit; krylovdim=krylovdim, tol=tol)
        out[k] = psit
        tprev = t

        if k % 10 == 0
            println("ED step $k / $(length(times))")
        end
    end
    return out
end

# ----------------------------
# Run
# ----------------------------
n  = 3
L  = 13
α  = 1.5
J  = 1.0
g  = 0.5
p  = make_params(n, L; alpha=α, J=J, g=g)

dim = n^L
psi0 = zeros(ComplexF64, dim)

j0 = (L + 1) ÷ 2
state0 = 1 * p.powN[j0]   # a_j0 = 1, others 0
psi0[state0 + 1] = 1.0 + 0im

times = collect(range(0.0, 5.0, length=51))
psis  = evolve_times(psi0, p, times; krylovdim=50, tol=1e-11)

# ----------------------------
# Observables: P_exc = P1 + P2
# ----------------------------
Nt = length(times)

P1   = zeros(Float64, L, Nt)
P2   = zeros(Float64, L, Nt)
Pexc = zeros(Float64, L, Nt)

center_exc = zeros(Float64, Nt)
avg_disp   = zeros(Float64, Nt)

for (k, ψ) in enumerate(psis)
    P1[:, k] = measure_Pa_all_sites(ψ, p, 1)
    P2[:, k] = measure_Pa_all_sites(ψ, p, 2)
    Pexc[:, k] = P1[:, k] .+ P2[:, k]

    center_exc[k] = Pexc[j0, k]
    avg_disp[k] = average_displacement(view(Pexc, :, k), j0)
end

println("Center excitation Pexc(j0,t_final) = ", center_exc[end])
println("Avg displacement <|j-j0|>(t_final) = ", avg_disp[end])

# ----------------------------
# Save to JLD2
# ----------------------------
outfile = "ED_Z3_L$(L)_alpha$(α)_g$(g)_single_excitation.jld2"
@save outfile n L α J g j0 times P1 P2 Pexc center_exc avg_disp
println("Saved to: ", outfile)