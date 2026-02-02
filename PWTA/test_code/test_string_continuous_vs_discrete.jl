###############################################################################
# Test robustness of pTWA to string-symbol choice
# Continuous string  exp(i θ Σ n_k)
# Discrete string    ω^{Σ round(n_k)}
#
# Outputs:
#   - S(t), R2(t) for both choices
#   - Difference plots
#   - Saved JLD2 data for appendix
###############################################################################

using Random
using LinearAlgebra
using Statistics
using JLD2
using Plots


struct PTWAParams
    L::Int
    Jr::Vector{Float64}          # length L-1, Jr[r] = J/r^α
    Gr::Vector{Float64}          # length L-1, Gr[r] = G/r^α
    μ::Vector{Float64}           # length L
    ω::ComplexF64                # exp(2π i/3)
    θ::Float64                   # 2π/3
end

function PTWAParams(L::Int; J::Real=1.0, G::Real=0.0, α::Real=Inf, μ=zeros(L))
    Jr = [Float64(J) / (r^Float64(α)) for r in 1:(L-1)]
    Gr = [Float64(G) / (r^Float64(α)) for r in 1:(L-1)]
    ω  = cis(2π/3)
    θ  = 2π/3
    return PTWAParams(L, Jr, Gr, Float64.(μ), ComplexF64(ω), Float64(θ))
end



# ============================ Initial configurations ==========================

function init_single_excitation(L::Int; background::Int=0, excitation::Int=1, pos=nothing)
    (0 <= background <= 2) || error("background must be in {0,1,2}")
    (0 <= excitation <= 2) || error("excitation must be in {0,1,2}")
    if pos === nothing
        isodd(L) || error("Default center pos requires odd L; provide pos explicitly.")
        pos = (L + 1) ÷ 2
    end
    (1 <= pos <= L) || error("pos must be in 1..L")
    s = fill(background, L)
    s[pos] = excitation
    return s
end

# ========================== Discrete Wigner sampling ==========================

const ω3 = cis(2π/3)

function Zmat()
    return Diagonal(ComplexF64[ω3^0, ω3^1, ω3^2])
end

function Xmat()
    X = zeros(ComplexF64, 3, 3)
    X[2,1] = 1
    X[3,2] = 1
    X[1,3] = 1
    return X
end

const Z3 = Zmat()
const X3 = Xmat()
const inv2_mod3 = 2

function Aqp_WH(q::Int, p::Int)
    @assert 0 ≤ q ≤ 2 && 0 ≤ p ≤ 2
    A = zeros(ComplexF64, 3, 3)
    for m in 0:2
        Zm = Z3^m
        for k in 0:2
            Xk = X3^k
            phase_exp = mod(p*k - q*m + inv2_mod3*m*k, 3)
            A .+= ω3^phase_exp .* (Zm * Xk)
        end
    end
    A ./= 3
    A .= (A .+ A') ./ 2
    return A
end

function precompute_A_WH()
    Acache = Array{ComplexF64,4}(undef, 3,3, 3,3)
    for q in 0:2, p in 0:2
        Acache[:,:,q+1,p+1] = Aqp_WH(q,p)
    end
    return Acache
end


function local_wigner_probs(ρ::Matrix{ComplexF64}, Acache; tol=1e-12)
    probs = zeros(Float64, 3, 3)
    for q in 0:2, p in 0:2
        A = Acache[:,:,q+1,p+1]
        probs[q+1,p+1] = real(tr(ρ*A)) / 3
    end
    # For computational-basis states this should be nonnegative up to roundoff
    if minimum(probs) < -tol
        @warn "Wigner negativity detected: min=$(minimum(probs)) (tol=$tol)"
    end
    probs ./= sum(probs)
    return probs
end


# function local_wigner_probs(ρ::Matrix{ComplexF64}, Acache)
#     probs = zeros(Float64, 3, 3)
#     for q in 0:2, p in 0:2
#         A = Acache[:,:,q+1,p+1]
#         probs[q+1,p+1] = real(tr(ρ*A)) / 3
#     end
#     probs .= max.(probs, 0.0)
#     probs ./= sum(probs)
#     return probs
# end

function sample_initial_discrete_WH(L::Int, s::Vector{Int}; rng::AbstractRNG=Random.default_rng())
    @assert length(s) == L
    Acache = precompute_A_WH()
    x = [zeros(ComplexF64, 3, 3) for _ in 1:L]
    for j in 1:L
        ρ = zeros(ComplexF64, 3, 3)
        ρ[s[j]+1, s[j]+1] = 1.0 + 0im
        probs = local_wigner_probs(ρ, Acache)

        r = rand(rng)
        acc = 0.0
        qsel, psel = 0, 0
        for q in 0:2, p in 0:2
            acc += probs[q+1, p+1]
            if r ≤ acc
                qsel, psel = q, p
                break
            end
        end

        A = Acache[:, :, qsel+1, psel+1]
        @inbounds for a in 1:3, b in 1:3
            x[j][a,b] = A[b,a]  # x^{ab} = A_{ba}
        end
    end
    return x
end

# ====================== Gaussian parafermion initial sampling =================

function sample_initial_gaussian_parafermion(L::Int, s::Vector{Int};
                                            rng::AbstractRNG = Random.default_rng())
    @assert length(s) == L
    x = [zeros(ComplexF64, 3, 3) for _ in 1:L]

    for j in 1:L
        sj = s[j] + 1
        xj = x[j]
        xj[sj, sj] = 1.0 + 0im

        for a in 1:3, b in 1:3
            a == b && continue
            var = 0.5 * ((a == sj) + (b == sj))
            if var > 0
                σ = sqrt(var / 2)
                xj[a,b] = (randn(rng)*σ) + 1im*(randn(rng)*σ)
            end
        end
        xj .= (xj .+ xj') ./ 2
    end
    return x
end

# ============================ Local symbols (ED) ==============================

@inline function n_symbol(xj::Matrix{ComplexF64})
    return real(xj[2,2]) + 2.0 * real(xj[3,3])
end

@inline function f_symbol(xj::Matrix{ComplexF64})
    return xj[1,2] + xj[2,3]
end

@inline function fdag_symbol(xj::Matrix{ComplexF64})
    return xj[2,1] + xj[3,2]
end

@inline function f2_symbol(xj::Matrix{ComplexF64})
    return xj[1,3]
end

@inline function fdag2_symbol(xj::Matrix{ComplexF64})
    return xj[3,1]
end

@inline function Z_symbol(params::PTWAParams, xj::Matrix{ComplexF64})
    ω = params.ω
    return (1.0+0im)*xj[1,1] + ω*xj[2,2] + (ω^2)*xj[3,3]
end

@inline function populations_symbol(xj::Matrix{ComplexF64})
    return (real(xj[1,1]), real(xj[2,2]), real(xj[3,3]))
end

# ================================ Strings ====================================

@inline function ωpow_real(params::PTWAParams, x::Float64)
    return cis(params.θ * x)
end


# ====================== helper: discrete occupation ==========================

@inline function n_discrete(n::Float64)
    nd = round(Int, n)
    return clamp(nd, 0, 2)
end

# ====================== string factors =======================================

function string_factor_continuous(params, nbar, i, j; power::Int=1)
    if i == j || abs(i-j) == 1
        return 1.0 + 0im
    end
    s = 0.0
    if i < j
        @inbounds for k in (i+1):(j-1)
            s += nbar[k]
        end
        return cis(params.θ * power * s)
    else
        @inbounds for k in (j+1):(i-1)
            s += nbar[k]
        end
        return cis(-params.θ * power * s)
    end
end

function string_factor_discrete(params, nbar, i, j; power::Int=1)
    if i == j || abs(i-j) == 1
        return 1.0 + 0im
    end
    s = 0
    if i < j
        @inbounds for k in (i+1):(j-1)
            s += n_discrete(nbar[k])
        end
        return cis(params.θ * power * s)
    else
        @inbounds for k in (j+1):(i-1)
            s += n_discrete(nbar[k])
        end
        return cis(-params.θ * power * s)
    end
end

# ====================== SWITCHABLE gradient ==================================

function compute_gradient!(G, x, params; string_mode::Symbol=:continuous)

    L  = params.L
    Jr = params.Jr
    Gr = params.Gr
    μ  = params.μ
    θ  = params.θ

    for j in 1:L
        fill!(G[j], 0.0 + 0im)
    end

    nbar  = [real(x[j][2,2]) + 2real(x[j][3,3]) for j in 1:L]
    f     = [x[j][1,2] + x[j][2,3] for j in 1:L]
    fdag  = [x[j][2,1] + x[j][3,2] for j in 1:L]
    f2    = [x[j][1,3] for j in 1:L]
    fdag2 = [x[j][3,1] for j in 1:L]

    # onsite μ n
    @inbounds for j in 1:L
        G[j][2,2] += μ[j]
        G[j][3,3] += 2μ[j]
    end

    string = string_mode === :continuous ?
             string_factor_continuous :
             string_factor_discrete

    # ---- single-particle hopping ----
    for r in 1:(L-1)
        J = Jr[r]
        J == 0 && continue
        for i in 1:(L-r)
            j = i + r

            Sij = string(params, nbar, i, j; power=1)
            Sji = conj(Sij)

            G[i][1,2] += J * Sji * fdag[j]
            G[i][2,3] += J * Sji * fdag[j]
            G[i][2,1] += J * Sij * f[j]
            G[i][3,2] += J * Sij * f[j]

            G[j][1,2] += J * Sij * fdag[i]
            G[j][2,3] += J * Sij * fdag[i]
            G[j][2,1] += J * Sji * f[i]
            G[j][3,2] += J * Sji * f[i]

            if j > i+1
                amp_f  = J * (fdag[i] * f[j])
                amp_b  = J * (fdag[j] * f[i])
                @inbounds for k in (i+1):(j-1)
                    G[k][2,2] += (1im*θ)*( Sij*amp_f - Sji*amp_b )
                    G[k][3,3] += 2(1im*θ)*( Sij*amp_f - Sji*amp_b )
                end
            end
        end
    end

    return nothing
end

# ====================== equations of motion ==================================

function rhs!(dx, x, params, G; string_mode::Symbol)
    compute_gradient!(G, x, params; string_mode=string_mode)
    @inbounds for j in 1:params.L
        h = transpose(G[j])
        dx[j] .= 1im * (x[j]*h - h*x[j])
    end
end

function step_rk4!(x, params, dt, G, k1, k2, k3, k4, xtmp; string_mode)
    rhs!(k1, x, params, G; string_mode=string_mode)
    for j in eachindex(x); xtmp[j] .= x[j] .+ 0.5dt*k1[j]; end
    rhs!(k2, xtmp, params, G; string_mode=string_mode)
    for j in eachindex(x); xtmp[j] .= x[j] .+ 0.5dt*k2[j]; end
    rhs!(k3, xtmp, params, G; string_mode=string_mode)
    for j in eachindex(x); xtmp[j] .= x[j] .+ dt*k3[j]; end
    rhs!(k4, xtmp, params, G; string_mode=string_mode)

    for j in eachindex(x)
        x[j] .+= (dt/6)*(k1[j] + 2k2[j] + 2k3[j] + k4[j])
        x[j] .= (x[j] .+ x[j]') ./ 2
    end
end

# ====================== MAIN TEST ============================================

println("Running string robustness test...")

# --- parameters (match paper) ---
L = 15
α = 0.5
J = 1.0
G = 0.0
μ = zeros(L)

params = PTWAParams(L; J=J, G=G, α=α, μ=μ)

# initial single excitation
s = fill(0, L)
j0 = (L+1) ÷ 2
s[j0] = 1

ntraj = 3000
dt    = 0.025
tmax  = 10.0
nt    = Int(floor(tmax/dt)) + 1
t     = collect(0:dt:tmax)

function run_string_mode(mode::Symbol)
    S  = zeros(nt)
    R2 = zeros(nt)

    G  = [zeros(ComplexF64,3,3) for _ in 1:L]
    k1 = deepcopy(G); k2 = deepcopy(G); k3 = deepcopy(G); k4 = deepcopy(G)
    xt = deepcopy(G)

    rng = MersenneTwister(123)

    for tr in 1:ntraj
        x = sample_initial_discrete_WH(L, s; rng=rng)
        for ti in 1:nt
            P1 = [real(x[j][2,2]) for j in 1:L]
            S[ti]  += P1[j0]
            R2[ti] += sum((j-j0)^2 * P1[j] for j in 1:L) / sum(P1)
            ti < nt && step_rk4!(x, params, dt, G, k1,k2,k3,k4,xt; string_mode=mode)
        end
    end

    return S ./ ntraj, R2 ./ ntraj
end

S_cont, R2_cont = run_string_mode(:continuous)
S_disc, R2_disc = run_string_mode(:discrete)

# ====================== SAVE DATA ============================================

outfile = "string_robustness_Z3_L$(L)_alpha$(α).jld2"
@save outfile t S_cont R2_cont S_disc R2_disc
println("Saved → $outfile")

# ====================== PLOTS ================================================

plot(t, S_cont, label="continuous", lw=2)
plot!(t, S_disc, label="discrete", ls=:dash)
savefig("S_string_compare.png")

plot(t, R2_cont, label="continuous", lw=2)
plot!(t, R2_disc, label="discrete", ls=:dash)
savefig("R2_string_compare.png")

plot(t, abs.(S_cont .- S_disc), yscale=:log10,
     ylabel="|ΔS(t)|", xlabel="t", label=false)
savefig("ΔS_string.png")

plot(t, abs.(R2_cont .- R2_disc), yscale=:log10,
     ylabel="|ΔR²(t)|", xlabel="t", label=false)
savefig("ΔR2_string.png")

println("Plots saved: S, R2, ΔS, ΔR2")
