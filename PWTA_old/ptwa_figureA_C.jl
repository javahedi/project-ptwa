
###############################################################################
# pTWA for the long-range Z₃ *parafermion* chain (ED-matched: strings + hopping)
#
# Observables for paper:
#   Fig A:  P1(j,t) heatmap (population of |1⟩ level)
#   Fig C:  S(t)=P1(j0,t) survival + R^2(t)=Σ (j-j0)^2 P1(j,t)
#
###############################################################################

using Random
using LinearAlgebra
using Statistics
using JLD2
using Plots

# =============================== Parameters ==================================

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

function string_factor(params::PTWAParams, nbar::Vector{Float64}, i::Int, j::Int; power::Int=1)
    if i == j || abs(i-j) == 1
        return 1.0 + 0im
    end
    if i < j
        s = 0.0
        @inbounds for k in (i+1):(j-1)
            s += nbar[k]
        end
        return ωpow_real(params, power*s)
    else
        s = 0.0
        @inbounds for k in (j+1):(i-1)
            s += nbar[k]
        end
        return ωpow_real(params, -power*s)
    end
end

# ===================== ∂H/∂x : endpoints + intermediates =====================

function compute_gradient!(G::Vector{Matrix{ComplexF64}},
                           x::Vector{Matrix{ComplexF64}},
                           params::PTWAParams)

    L = params.L
    Jr = params.Jr
    Gr = params.Gr
    μ  = params.μ
    θ  = params.θ

    for j in 1:L
        fill!(G[j], 0.0 + 0im)
    end

    nbar  = Vector{Float64}(undef, L)
    f     = Vector{ComplexF64}(undef, L)
    fdag  = Vector{ComplexF64}(undef, L)
    f2    = Vector{ComplexF64}(undef, L)
    fdag2 = Vector{ComplexF64}(undef, L)
    @inbounds for j in 1:L
        nbar[j]  = n_symbol(x[j])
        f[j]     = f_symbol(x[j])
        fdag[j]  = fdag_symbol(x[j])
        f2[j]    = f2_symbol(x[j])
        fdag2[j] = fdag2_symbol(x[j])
    end

    # onsite μ n
    @inbounds for j in 1:L
        G[j][2,2] += μ[j]
        G[j][3,3] += 2.0 * μ[j]
    end

    @inline function add_df!(Gj, coef::ComplexF64)
        Gj[1,2] += coef
        Gj[2,3] += coef
    end
    @inline function add_dfdag!(Gj, coef::ComplexF64)
        Gj[2,1] += coef
        Gj[3,2] += coef
    end
    @inline function add_df2!(Gj, coef::ComplexF64)
        Gj[1,3] += coef
    end
    @inline function add_dfdag2!(Gj, coef::ComplexF64)
        Gj[3,1] += coef
    end
    @inline function add_dn_intermediate!(Gj, coef::ComplexF64)
        Gj[2,2] += coef
        Gj[3,3] += 2.0 * coef
    end

    # ---- single-particle hopping with strings ----
    for r in 1:(L-1)
        J = Jr[r]
        J == 0.0 && continue
        for i in 1:(L-r)
            j = i + r

            Sij = string_factor(params, nbar, i, j; power=1)
            Sji = conj(Sij)

            # endpoints
            add_dfdag!(G[i], J * Sij * f[j])
            add_df!(G[i],    J * Sji * fdag[j])
            add_dfdag!(G[j], J * Sji * f[i])
            add_df!(G[j],    J * Sij * fdag[i])

            # intermediates: string derivatives
            if j > i+1
                amp_forward  = J * (fdag[i] * f[j])
                amp_backward = J * (fdag[j] * f[i])
                @inbounds for k in (i+1):(j-1)
                    add_dn_intermediate!(G[k],
                        (1im*θ)  * Sij * amp_forward +
                        (-1im*θ) * Sji * amp_backward
                    )
                end
            end
        end
    end

    # ---- pair hopping with squared strings ----
    for r in 1:(L-1)
        Gc = Gr[r]
        Gc == 0.0 && continue
        for i in 1:(L-r)
            j = i + r

            S2ij = string_factor(params, nbar, i, j; power=2)
            S2ji = conj(S2ij)

            add_dfdag2!(G[i], Gc * S2ij * f2[j])
            add_df2!(G[i],    Gc * S2ji * fdag2[j])

            add_dfdag2!(G[j], Gc * S2ji * f2[i])
            add_df2!(G[j],    Gc * S2ij * fdag2[i])

            if j > i+1
                amp_forward  = Gc * (fdag2[i] * f2[j])
                amp_backward = Gc * (fdag2[j] * f2[i])
                @inbounds for k in (i+1):(j-1)
                    add_dn_intermediate!(G[k],
                        (1im*2θ)  * S2ij * amp_forward +
                        (-1im*2θ) * S2ji * amp_backward
                    )
                end
            end
        end
    end

    return nothing
end

# ============================== Equations of motion ===========================

function rhs!(dx::Vector{Matrix{ComplexF64}},
              x::Vector{Matrix{ComplexF64}},
              params::PTWAParams,
              workG::Vector{Matrix{ComplexF64}})

    compute_gradient!(workG, x, params)

    @inbounds for j in 1:params.L
        xj = x[j]
        hj = transpose(workG[j])   # Hubbard bracket requires transpose
        dx[j] .= 1im .* (xj * hj - hj * xj)
    end
    return nothing
end

# ================================ Integrator =================================

function step_rk4!(x::Vector{Matrix{ComplexF64}}, params::PTWAParams, dt::Float64;
                   k1, k2, k3, k4, xtmp, Gwork)

    L = params.L

    rhs!(k1, x, params, Gwork)
    @inbounds for j in 1:L
        xtmp[j] .= x[j] .+ (dt/2) .* k1[j]
    end

    rhs!(k2, xtmp, params, Gwork)
    @inbounds for j in 1:L
        xtmp[j] .= x[j] .+ (dt/2) .* k2[j]
    end

    rhs!(k3, xtmp, params, Gwork)
    @inbounds for j in 1:L
        xtmp[j] .= x[j] .+ dt .* k3[j]
    end

    rhs!(k4, xtmp, params, Gwork)
    @inbounds for j in 1:L
        x[j] .+= (dt/6) .* (k1[j] .+ 2 .* k2[j] .+ 2 .* k3[j] .+ k4[j])
        x[j] .= (x[j] .+ x[j]') ./ 2  # keep Hermitian numerically
    end

    return nothing
end

# ============================== Fig C helpers ================================

"""
    compute_R2(P1::AbstractVector{<:Real}, j0::Int; normalize=true)

R2(t) = Σ_j (j-j0)^2 P1(j,t).
If normalize=true, divides by Σ_j P1(j,t) to remove drift in total weight.
"""
function compute_R2(P1::AbstractVector{<:Real}, j0::Int; normalize::Bool=true)
    L = length(P1)
    num = 0.0
    den = 0.0
    @inbounds for j in 1:L
        w = float(P1[j])
        den += w
        num += (j - j0)^2 * w
    end
    if normalize
        return den > 1e-14 ? (num / den) : 0.0
    else
        return num
    end
end

# ============================== Main pTWA runner ==============================

"""
    run_ptwa_parafermion_AC(params; s, sampler=:discrete, ...)

Returns:
  t, Zmean, Pmean, P1t, S, R2, meta
where
  P1t[tidx, j] = P1(j,t)
  S[tidx]      = P1(j0,t)
  R2[tidx]     = mean-square radius around j0
"""
function run_ptwa_parafermion_AC(params::PTWAParams; s::Vector{Int},
                                 sampler::Symbol=:discrete,
                                 ntraj::Int=5000, tmax::Float64=4.0, dt::Float64=0.05,
                                 seed::Int=123, save_every::Int=1,
                                 normalize_R2::Bool=true)

    L = params.L
    @assert length(s) == L
    j0 = (L + 1) ÷ 2

    nt = Int(floor(tmax/dt)) + 1
    save_idx = 1:save_every:nt
    t = [(i-1)*dt for i in save_idx]
    ns = length(t)

    Zmean = zeros(ComplexF64, ns, L)
    Pmean = zeros(Float64, ns, L, 3)

    # Fig A & C
    P1t = zeros(Float64, ns, L)
    S   = zeros(Float64, ns)
    R2  = zeros(Float64, ns)

    rng = MersenneTwister(seed)

    # prealloc buffers
    Gwork = [zeros(ComplexF64,3,3) for _ in 1:L]
    k1    = [zeros(ComplexF64,3,3) for _ in 1:L]
    k2    = [zeros(ComplexF64,3,3) for _ in 1:L]
    k3    = [zeros(ComplexF64,3,3) for _ in 1:L]
    k4    = [zeros(ComplexF64,3,3) for _ in 1:L]
    xtmp  = [zeros(ComplexF64,3,3) for _ in 1:L]

    function sample_x()
        if sampler == :discrete
            return sample_initial_discrete_WH(L, s; rng=rng)
        elseif sampler == :gaussian
            return sample_initial_gaussian_parafermion(L, s; rng=rng)
        else
            error("Unknown sampler=$sampler. Use :discrete or :gaussian.")
        end
    end

    for trj in 1:ntraj
        x = sample_x()

        save_counter = 1
        function accumulate!(idx::Int)
            @inbounds for j in 1:L
                Zmean[idx,j] += Z_symbol(params, x[j])
                p0,p1,p2 = populations_symbol(x[j])
                Pmean[idx,j,1] += p0
                Pmean[idx,j,2] += p1
                Pmean[idx,j,3] += p2

                P1t[idx,j] += p1
            end
        end

        accumulate!(save_counter)

        for step in 2:nt
            step_rk4!(x, params, dt; k1=k1,k2=k2,k3=k3,k4=k4,xtmp=xtmp,Gwork=Gwork)
            if (step-1) % save_every == 0
                save_counter += 1
                accumulate!(save_counter)
            end
        end

        if trj % max(1, ntraj ÷ 10) == 0
            println("pTWA traj $trj / $ntraj")
        end
    end

    Zmean ./= ntraj
    Pmean ./= ntraj
    P1t   ./= ntraj

    # build Fig C curves from the averaged P1
    @inbounds for ti in 1:ns
        S[ti]  = P1t[ti, j0]
        R2[ti] = compute_R2(view(P1t, ti, :), j0; normalize=normalize_R2)
    end

    meta = Dict(
        "L" => L,
        "alpha" => (length(params.Jr) > 1 ? NaN : NaN),  # optional placeholder
        "Jr" => params.Jr,
        "Gr" => params.Gr,
        "mu" => params.μ,
        "dt" => dt,
        "tmax" => tmax,
        "ntraj" => ntraj,
        "sampler" => String(sampler),
        "init" => "single excitation center",
        "j0" => j0,
        "normalize_R2" => normalize_R2,
        "model" => "Z3 parafermion long-range with strings (pTWA)"
    )

    return t, Zmean, Pmean, P1t, S, R2, meta
end

# =============================== Plot helpers =================================

# Fig A: P1 heatmap
function save_figA_P1_heatmap(t, P1t; prefix="pTWA", L::Int, α, J, G, sampler::Symbol)
    plt = heatmap(1:L, t, P1t;
        xlabel="site j", ylabel="time t",
        title="pTWA  P1(j,t)  (L=$L, α=$α, J=$J, G=$G, $(sampler))",
        aspect_ratio=:auto, colorbar_title="P1"
    )
    savefig(plt, "$(prefix)_FigA_P1_heatmap_$(sampler)_L$(L)_alpha$(α).png")
end

# Fig C: S(t) and R2(t)
function save_figC_curves(t, S, R2; prefix="pTWA", L::Int, α, J, G, sampler::Symbol)
    pltS = plot(t, S, xlabel="time t", ylabel="S(t)=P1(j0,t)",
                title="pTWA Fig C: Survival  (L=$L, α=$α, J=$J, G=$G, $(sampler))",
                label="S(t)")
    savefig(pltS, "$(prefix)_FigC_Survival_$(sampler)_L$(L)_alpha$(α).png")

    pltR = plot(t, R2, xlabel="time t", ylabel="R²(t)",
                title="pTWA Fig C: Mean-square radius  (L=$L, α=$α, J=$J, G=$G, $(sampler))",
                label="R²(t)")
    savefig(pltR, "$(prefix)_FigC_R2_$(sampler)_L$(L)_alpha$(α).png")
end

# ================================ Example run ================================

if abspath(PROGRAM_FILE) == @__FILE__

    # ---- match ED test ----
    L = 15
    α = 3.0
    J = 1.0
    G = 0.0
    μ = zeros(L)

    params = PTWAParams(L; J=J, G=G, α=α, μ=μ)

    # initial: single excitation at center |0...0 1 0...0|
    s = init_single_excitation(L; background=0, excitation=1, pos=(L+1)÷2)

    # run dynamics
    ntraj = 5000
    tmax  = 4.0
    dt    = 0.05

    sampler = :discrete   # :discrete or :gaussian

    t, Zm, Pm, P1t, S, R2, meta =
        run_ptwa_parafermion_AC(params; s=s, sampler=sampler,
                                ntraj=ntraj, tmax=tmax, dt=dt,
                                seed=123, save_every=1, normalize_R2=true)

    # save full dataset for later plotting (paper pipeline)
    outfile = "pTWA_parafermion_$(sampler)_Z3_L$(L)_alpha$(α)_single_AC.jld2"
    @save outfile t Zm Pm P1t S R2 meta
    println("Saved data → $outfile")

    # Paper figures:
    save_figA_P1_heatmap(t, P1t; prefix="pTWA", L=L, α=α, J=J, G=G, sampler=sampler)
    save_figC_curves(t, S, R2; prefix="pTWA", L=L, α=α, J=J, G=G, sampler=sampler)

    println("Saved Fig A (P1 heatmap) and Fig C (S(t), R2(t)) PDFs.")
end
