###############################################################################
# pTWA for the long-range Z₃ clock chain (Gaussian initial sampling)
#
# Hamiltonian (open boundary conditions):
#   H = -∑_{r=1}^{L-1} J_r ∑_{j=1}^{L-r} ( Z_j Z_{j+r}† + Z_j† Z_{j+r} )
#       -∑_{j=1}^{L} ( f X_j + f* X_j† )
#
# with ω = exp(2π i / 3),  Z = diag(1, ω, ω²),  X|a⟩ = |a+1⟩.
#
# pTWA variables: on each site j we evolve a 3×3 matrix x_j whose entries are
# Hubbard symbols x_j^{ab} ≈ ⟨|a⟩⟨b|⟩_W, obeying Lie–Poisson dynamics
#   ẋ_j = i [ x_j, h_j ],  where h_j = ∂H_W/∂x_j (matrix of "fields").
#
# Gaussian sampling: for product basis states |s⟩ on each site we sample
# off-diagonals x^{ab} (a≠b) as complex Gaussians with variance chosen to match
# the symmetric second moment:
#   E[ |x^{ab}|² ] = ½ ⟨ {X^{ab}, X^{ba}} ⟩ = (δ_{a,s}+δ_{b,s})/2
# which implies Var(Re) = Var(Im) = (δ_{a,s}+δ_{b,s})/4.
#
# This file is self-contained and intended as a starting point for comparisons
# against ED/Krylov (e.g. ParafermionDynamic.jl) for L=10 or 12.
###############################################################################

using Random
using LinearAlgebra
using Statistics
using JLD2
using FileIO
using Plots

# ----------------------------- Model parameters -----------------------------

"""
    PTWAParams(L; J=1.0, α=Inf, f=1.0+0im, Jr=nothing)

Container for long-range Z₃ clock-chain parameters.

- `L`: chain length (e.g. 10 or 12)
- `J_r`: interaction profile. If `Jr` not provided, uses power-law `J/r^α`.
- `f`: transverse-field amplitude (complex allowed). Hamiltonian uses `f X + f* X†`.
- Open boundary conditions are used (pairs j, j+r with j=1..L-r).
"""
struct PTWAParams
    L::Int
    Jr::Vector{Float64}      # length L-1, Jr[r] = J_r
    f::ComplexF64
    ω::ComplexF64
    ωpow::NTuple{3,ComplexF64}  # (ω^0, ω^1, ω^2)
end

function PTWAParams(L::Int; J::Real=1.0, α::Real=Inf, f=1.0+0im, Jr=nothing)
    Jr_vec = if Jr === nothing
        [Float64(J) / (r^Float64(α)) for r in 1:(L-1)]
    else
        length(Jr) == L-1 || error("Jr must have length L-1")
        Float64.(Jr)
    end
    ω = cis(2π/3)  # exp(2π i / 3)
    ωpow = (1.0 + 0im, ω, ω^2)
    return PTWAParams(L, Jr_vec, ComplexF64(f), ω, ωpow)
end

# --------------------------- Initial-state patterns --------------------------

"""
    init_config(L, pattern::Symbol; left=0, right=1)

Return a length-L vector `s` with local computational-basis labels s[j] ∈ {0,1,2}.

Supported patterns:
- `:product0`   -> all |0⟩
- `:product1`   -> all |1⟩
- `:product2`   -> all |2⟩
- `:neel01`     -> |0,1,0,1,...⟩ (qutrit "Néel-like" in 0/1 subspace)
- `:neel012`    -> |0,1,2,0,1,2,...⟩
- `:domainwall` -> first half `left`, second half `right` (default 0|1)
"""
function init_config(L::Int, pattern::Symbol; left::Int=0, right::Int=1)
    s = Vector{Int}(undef, L)
    if pattern == :product0
        fill!(s, 0)
    elseif pattern == :product1
        fill!(s, 1)
    elseif pattern == :product2
        fill!(s, 2)
    elseif pattern == :neel01
        for j in 1:L
            s[j] = isodd(j) ? 0 : 1
        end
    elseif pattern == :neel012
        for j in 1:L
            s[j] = (j - 1) % 3
        end
    elseif pattern == :domainwall
        mid = L ÷ 2
        for j in 1:L
            s[j] = (j <= mid) ? left : right
        end
    else
        error("Unknown pattern: $pattern")
    end
    all(0 .<= s .<= 2) || error("Config entries must be in {0,1,2}")
    return s
end



"""
    init_single_excitation(L; background=0, excitation=1, pos=nothing)

Return a product basis configuration of length L with all sites in `background`
except a single site `pos` set to `excitation`.

- `background`, `excitation` must be in {0,1,2}
- if `pos` is not given, uses the center site (L+1)/2 for odd L
"""
function init_single_excitation(L::Int; background::Int=0, excitation::Int=1, pos=nothing)
    (0 <= background <= 2) || error("background must be in {0,1,2}")
    (0 <= excitation <= 2) || error("excitation must be in {0,1,2}")
    if pos === nothing
        isodd(L) || error("Default center position requires odd L; provide pos explicitly.")
        pos = (L + 1) ÷ 2
    end
    (1 <= pos <= L) || error("pos must be in 1..L")

    s = fill(background, L)
    s[pos] = excitation
    return s
end

# --------------------------- Gaussian Wigner sampling ------------------------


"""
    sample_initial_gaussian_psd(params, L, s; σ=0.15, rng=Random.default_rng())

Gaussian pTWA sampler that produces *physical* 3×3 density-matrix-like symbols:
- mean is the basis projector |s⟩⟨s|
- adds Hermitian Gaussian noise with scale σ
- projects to PSD and trace=1

This is a consistent Gaussian semiclassical initialization for qutrits.
"""
function sample_initial_gaussian_psd(params::PTWAParams, L, s::Vector{Int};
                                    σ::Float64=0.15, rng=Random.default_rng())
    
    x = [zeros(ComplexF64, 3, 3) for _ in 1:L]

    for j in 1:L
        sj = s[j] + 1
        ρ = zeros(ComplexF64,3,3)
        ρ[sj,sj] = 1.0 + 0im  # mean projector

        # Hermitian Gaussian noise matrix N (zero mean)
        N = zeros(ComplexF64,3,3)
        for a in 1:3
            N[a,a] = (randn(rng) * σ) + 0im
        end
        for a in 1:3, b in (a+1):3
            re = randn(rng) * σ
            im = randn(rng) * σ
            z = ComplexF64(re, im)
            N[a,b] = z
            N[b,a] = conj(z)
        end

        ρ .= ρ .+ N

        # PSD projection + trace=1
        project_density!(ρ)

        x[j] .= ρ
    end
    return x
end







const ω3 = cis(2π/3)

# Clock and shift for n=3
function Zmat()
    return Diagonal(ComplexF64[ω3^0, ω3^1, ω3^2])
end

function Xmat()
    X = zeros(ComplexF64, 3, 3)
    # X|a> = |a+1>
    X[2,1] = 1
    X[3,2] = 1
    X[1,3] = 1
    return X
end

const Z3 = Zmat()
const X3 = Xmat()

# inverse of 2 mod 3 is 2, since 2*2 = 4 ≡ 1 mod 3
const inv2_mod3 = 2

"""
    Aqp_WH(q,p)

Phase-point operator A_{qp} for n=3 using the Gross/Weyl–Heisenberg formula:
A_{qp} = (1/3) Σ_{m,k=0}^2 ω^{p k - q m + (1/2) m k} Z^m X^k
with (1/2) taken mod 3 (inv2_mod3 = 2).
"""
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
    A .= (A .+ A') ./ 2  # enforce Hermiticity numerically
    return A
end

"""
    precompute_A_WH()

Precompute all 9 phase-point operators A_{qp}.
"""
function precompute_A_WH()
    Acache = Array{ComplexF64,4}(undef, 3,3, 3,3) # A[:,:,q+1,p+1]
    for q in 0:2, p in 0:2
        Acache[:,:,q+1,p+1] = Aqp_WH(q,p)
    end
    return Acache
end


function local_wigner_probs(ρ::Matrix{ComplexF64}, Acache)
    probs = zeros(Float64, 3, 3)
    for q in 0:2, p in 0:2
        A = Acache[:,:,q+1,p+1]
        probs[q+1,p+1] = real(tr(ρ*A)) / 3
    end
    probs .= max.(probs, 0.0)
    probs ./= sum(probs)
    return probs
end



"""
    sample_initial_discrete_WH(L, s; rng=Random.default_rng())

Discrete Wigner (Gross/Weyl–Heisenberg) sampling for Z₃ product basis states.

Inputs
------
- L  : chain length
- s  : Vector{Int} with s[j] ∈ {0,1,2} specifying |s₁ s₂ … s_L⟩
- rng: random number generator

Returns
-------
- x  : Vector{Matrix{ComplexF64}}, where each x[j] is a 3×3 matrix of
       Hubbard symbols x_j^{ab} = ⟨X_j^{ab}⟩_W for one sampled trajectory.

Notes
-----
- Sampling is exact for factorized basis states when the discrete Wigner
  function is non-negative.
- The mapping x^{ab} = Tr(A X^{ab}) = A_{ba} is used.
"""
function sample_initial_discrete_WH(L::Int, s::Vector{Int};
                                    rng::AbstractRNG = Random.default_rng())

    @assert length(s) == L
    Acache = precompute_A_WH()   # 3×3×3×3 array

    x = [zeros(ComplexF64, 3, 3) for _ in 1:L]

    for j in 1:L
        # --- local density matrix ρ = |s⟩⟨s|
        ρ = zeros(ComplexF64, 3, 3)
        ρ[s[j]+1, s[j]+1] = 1.0 + 0im

        # --- discrete Wigner probabilities W(q,p)
        probs = local_wigner_probs(ρ, Acache)

        # --- sample (q,p)
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

        # --- map A_{qp} → Hubbard symbols
        A = Acache[:, :, qsel+1, psel+1]
        for a in 1:3, b in 1:3
            x[j][a,b] = A[b,a]   # x^{ab} = A_{ba}
        end
    end

    return x
end


# ------------------------- Classical symbols Z_j and X_j ---------------------

@inline function Z_symbol(params::PTWAParams, xj::Matrix{ComplexF64})
    # Z = Σ_a ω^a x^{aa}, with a=0,1,2
    ω0, ω1, ω2 = params.ωpow
    return ω0 * xj[1,1] + ω1 * xj[2,2] + ω2 * xj[3,3]
end

@inline function X_symbol(xj::Matrix{ComplexF64})
    # X = Σ_a x^{a+1,a} with cyclic wrap (0->1->2->0)
    return xj[2,1] + xj[3,2] + xj[1,3]
end


# ------------------------- Effective fields G_j = ∂H/∂x_j --------------------
# NOTE: In the Hubbard Lie–Poisson equation, the commutator uses h = (∂H/∂x)^T.
# We compute and store G = ∂H/∂x here, and transpose in rhs!.

"""
    compute_fields!(G, x, params)

Fill `G[j]` (3×3) with the gradient matrix G_j = ∂H/∂x_j.

Hamiltonian (open chain):
  H = -∑_{j<k} J_{|j-k|} ( Z_j Z_k† + Z_j† Z_k ) - ∑_j ( f X_j + f* X_j† )

We build ∂H/∂x in a Hubbard-consistent way:
- Z_j = Σ_a ω^a x_j^{aa}
- (Z_j Z_k† + h.c.) = Σ_{a,b} C_{ab} x_j^{aa} x_k^{bb},  with C_{ab}=2Re(ω^{a-b})
- X_j = x_j^{21}+x_j^{32}+x_j^{13},  X_j† = x_j^{12}+x_j^{23}+x_j^{31}
"""
function compute_fields!(G::Vector{Matrix{ComplexF64}},
                         x::Vector{Matrix{ComplexF64}},
                         params::PTWAParams)

    L  = params.L
    Jr = params.Jr
    f  = params.f
    ω  = params.ω  # exp(2π i/3)

    # reset gradients
    for j in 1:L
        fill!(G[j], 0.0 + 0im)
    end

    # ----- Interaction term: exact Hubbard-diagonal gradient -----
    # C[a,b] = ω^(a-b) + ω^-(a-b) = 2 Re(ω^(a-b)), with a,b in {0,1,2}
    C = zeros(Float64, 3, 3)
    for a0 in 0:2, b0 in 0:2
        C[a0+1, b0+1] = 2 * real(ω^(a0 - b0))
    end

    # Add contributions for each pair (j,k=j+r)
    for r in 1:(L-1)
        J = Jr[r]
        J == 0.0 && continue
        for j in 1:(L - r)
            k = j + r

            # For site j: ∂/∂x_j^{aa} [-J Σ_b C[a,b] x_j^{aa} x_k^{bb}]
            #          = -J Σ_b C[a,b] x_k^{bb}
            xk11 = real(x[k][1,1]); xk22 = real(x[k][2,2]); xk33 = real(x[k][3,3])
            for a in 1:3
                G[j][a,a] += -J * (C[a,1]*xk11 + C[a,2]*xk22 + C[a,3]*xk33)
            end

            # Symmetric for site k from site j
            xj11 = real(x[j][1,1]); xj22 = real(x[j][2,2]); xj33 = real(x[j][3,3])
            for a in 1:3
                G[k][a,a] += -J * (C[a,1]*xj11 + C[a,2]*xj22 + C[a,3]*xj33)
            end
        end
    end

    # ----- Field term: linear in X and X† -----
    # H_X = -Σ_j ( f X_j + f* X_j† )
    # X_j  = x[2,1] + x[3,2] + x[1,3]
    # X_j† = x[1,2] + x[2,3] + x[3,1]
    #
    # Therefore:
    # ∂H/∂x[2,1] = -f,   ∂H/∂x[3,2] = -f,   ∂H/∂x[1,3] = -f
    # ∂H/∂x[1,2] = -f*,  ∂H/∂x[2,3] = -f*,  ∂H/∂x[3,1] = -f*
    for j in 1:L
        Gj = G[j]
        Gj[2,1] += -f
        Gj[3,2] += -f
        Gj[1,3] += -f

        Gj[1,2] += -conj(f)
        Gj[2,3] += -conj(f)
        Gj[3,1] += -conj(f)
    end

    return nothing
end


# ----------------------------- Equations of motion ---------------------------

"""
    rhs!(dx, x, params, work_G)

Compute time derivative dx for full chain state x.

Hubbard Lie–Poisson dynamics (matrix form):
    ẋ_j = i [ x_j , h_j ],   with   h_j = (∂H/∂x_j)^T.

`work_G` stores G_j = ∂H/∂x_j, then we transpose it.
"""
function rhs!(dx::Vector{Matrix{ComplexF64}},
              x::Vector{Matrix{ComplexF64}},
              params::PTWAParams,
              work_G::Vector{Matrix{ComplexF64}})

    compute_fields!(work_G, x, params)

    for j in 1:params.L
        xj = x[j]
        Gj = work_G[j]          # Gj = ∂H/∂x
        hj = transpose(Gj)      # hj = (∂H/∂x)^T  (required by the bracket)

        dx[j] .= 1im .* (xj * hj - hj * xj)
    end
    return nothing
end







function project_density!(xj::Matrix{ComplexF64})
    xj .= (xj .+ xj') ./ 2
    F = eigen(Hermitian(xj))
    vals = max.(F.values, 0.0)
    s = sum(vals)
    if s < 1e-14
        vals .= 1/3
    else
        vals ./= s
    end
    xj .= F.vectors * Diagonal(vals) * F.vectors'
    return nothing
end


"""
    project_populations!(xj)
Project the diagonal entries of `xj` to be nonnegative and sum to 1.
This helps prevent numerical drift of populations during time evolution.
"""

function project_populations!(xj::Matrix{ComplexF64})
    # enforce Hermiticity
    xj .= (xj .+ xj') ./ 2

    # enforce trace = 1
    trj = real(tr(xj))
    xj ./= trj

    # clip tiny negative diagonals (numerical noise)
    for a in 1:3
        xj[a,a] = complex(max(real(xj[a,a]), 0.0), 0.0)
    end

    # renormalize diagonals
    s = real(xj[1,1] + xj[2,2] + xj[3,3])
    xj[1,1] /= s
    xj[2,2] /= s
    xj[3,3] /= s

    return nothing
end


# ------------------------------- Time stepping -------------------------------

"""
    step_rk4!(x, params, dt; work=...)

One RK4 step for the chain state `x` with time step `dt`.

This uses preallocated work buffers to reduce allocations.
"""
function step_rk4!(x::Vector{Matrix{ComplexF64}}, params::PTWAParams, dt::Float64;
                   k1=nothing, k2=nothing, k3=nothing, k4=nothing,
                   xtmp=nothing, hwork=nothing)

    L = params.L
    k1 === nothing && (k1 = [zeros(ComplexF64,3,3) for _ in 1:L])
    k2 === nothing && (k2 = [zeros(ComplexF64,3,3) for _ in 1:L])
    k3 === nothing && (k3 = [zeros(ComplexF64,3,3) for _ in 1:L])
    k4 === nothing && (k4 = [zeros(ComplexF64,3,3) for _ in 1:L])
    xtmp === nothing && (xtmp = [zeros(ComplexF64,3,3) for _ in 1:L])
    hwork === nothing && (hwork = [zeros(ComplexF64,3,3) for _ in 1:L])

    rhs!(k1, x, params, hwork)

    for j in 1:L
        xtmp[j] .= x[j] .+ (dt/2) .* k1[j]
    end
    rhs!(k2, xtmp, params, hwork)

    for j in 1:L
        xtmp[j] .= x[j] .+ (dt/2) .* k2[j]
    end
    rhs!(k3, xtmp, params, hwork)

    for j in 1:L
        xtmp[j] .= x[j] .+ dt .* k3[j]
    end
    rhs!(k4, xtmp, params, hwork)

    for j in 1:L
        x[j] .+= (dt/6) .* (k1[j] .+ 2 .* k2[j] .+ 2 .* k3[j] .+ k4[j])
        # Optional mild projection: enforce Hermiticity numerically
        x[j] .= (x[j] .+ x[j]') ./ 2
        # Optional: enforce trace ~ 1
        # tr = real(trace(x[j]))
        # x[j] ./= tr
        #project_populations!(x[j])
    end

    return nothing
end






"""
    save_ptwa_data(filename, t, Zmean, Xmean; meta...)

Save pTWA results to a JLD2 file.

Stored fields:
- t        :: Vector{Float64}
- Zmean    :: Matrix{ComplexF64}  (nt × L)
- Xmean    :: Matrix{ComplexF64}
- meta     :: Dict (model and run parameters)
"""
function save_ptwa_data(filename::String,
                        t::Vector,
                        Zmean::Matrix,
                        Xmean::Matrix;
                        meta=Dict())

    @save filename t Zmean Xmean meta
    println("Saved pTWA data → $filename")
end







# ------------------------- Populations from Hubbard diagonals ----------------

"""
    populations_symbol(xj)

Return local populations P = (P0,P1,P2) from the diagonal Hubbard symbols:
P_a ≈ x^{aa}. These are the pTWA analog of ED population heatmaps.
"""
@inline function populations_symbol(xj::Matrix{ComplexF64})
    # Diagonals should be real-ish; take real part for robustness
    return (real(xj[1,1]), real(xj[2,2]), real(xj[3,3]))
end

"""
    run_ptwa_ZXP(params; s, ntraj=2000, tmax=4.0, dt=0.05, seed=123, save_every=1)

Run pTWA and record:
- Zmean[t,j] = ⟨Z_j(t)⟩
- Xmean[t,j] = ⟨X_j(t)⟩
- Pmean[t,j,a] = ⟨x_j^{aa}(t)⟩ for a=0,1,2 (ED-style populations)

Returns: t, Zmean, Xmean, Pmean
"""
function run_ptwa_ZXP(params::PTWAParams; s::Vector{Int},
                      ntraj::Int=2000, tmax::Float64=4.0, dt::Float64=0.05,
                      seed::Int=123, save_every::Int=1)

    L = params.L
    @assert length(s) == L

    nt = Int(floor(tmax/dt)) + 1
    save_idx = 1:save_every:nt
    tsave = [(i-1)*dt for i in save_idx]
    ns = length(tsave)

    Zmean = zeros(ComplexF64, ns, L)
    Xmean = zeros(ComplexF64, ns, L)
    Pmean = zeros(Float64, ns, L, 3)  # P0,P1,P2

    rng = MersenneTwister(seed)

    # Work buffers reused per trajectory
    hwork = [zeros(ComplexF64,3,3) for _ in 1:L]
    k1 = [zeros(ComplexF64,3,3) for _ in 1:L]
    k2 = [zeros(ComplexF64,3,3) for _ in 1:L]
    k3 = [zeros(ComplexF64,3,3) for _ in 1:L]
    k4 = [zeros(ComplexF64,3,3) for _ in 1:L]
    xtmp = [zeros(ComplexF64,3,3) for _ in 1:L]

    for tr in 1:ntraj
        #x = sample_initial_gaussian_psd(params, L, s; rng=rng)
        x = sample_initial_discrete_WH(params.L, s; rng=rng)


        save_counter = 1

        # accumulate at current x
        function accumulate!(idx::Int)
            for j in 1:L
                Zmean[idx, j] += Z_symbol(params, x[j])
                Xmean[idx, j] += X_symbol(x[j])
                p0,p1,p2 = populations_symbol(x[j])
                Pmean[idx, j, 1] += p0
                Pmean[idx, j, 2] += p1
                Pmean[idx, j, 3] += p2
            end
        end

        # t=0
        accumulate!(save_counter)

        # evolve
        for step in 2:nt
            step_rk4!(x, params, dt; k1=k1,k2=k2,k3=k3,k4=k4,xtmp=xtmp,hwork=hwork)
            if (step-1) % save_every == 0
                save_counter += 1
                accumulate!(save_counter)
            end
        end
    end

    Zmean ./= ntraj
    Xmean ./= ntraj
    Pmean ./= ntraj

    return tsave, Zmean, Xmean, Pmean
end


"""
    save_ptwa_heatmaps(t, Zmean, Pmean; prefix="pTWA", L, α)

Save:
- Re⟨Z⟩ heatmap
- P0, P1, P2 heatmaps
"""
function save_ptwa_heatmaps(t, Zmean, Pmean; prefix="pTWA", L::Int, α)
    # Re⟨Z⟩
    pltZ = heatmap(1:L, t, real.(Zmean);
        xlabel="site j", ylabel="time t",
        title="pTWA Re⟨Z_j(t)⟩ (L=$L, α=$α)",
        aspect_ratio=:auto, colorbar_title="Re⟨Z⟩"
    )
    savefig(pltZ, "$(prefix)_ReZ_heatmap_L$(L)_alpha$(α).pdf")

    # P0,P1,P2
    for a in 1:3
        pltP = heatmap(1:L, t, Pmean[:,:,a];
            xlabel="site j", ylabel="time t",
            title="pTWA P$(a-1)(j,t) (L=$L, α=$α)",
            aspect_ratio=:auto, colorbar_title="P"
        )
        savefig(pltP, "$(prefix)_P$(a-1)_heatmap_DisceretSampling_L$(L)_alpha$(α).pdf")
        #savefig(pltP, "$(prefix)_P$(a-1)_heatmap_GaussianSampling_L$(L)_alpha$(α).pdf")

        
    end
end




# assumes you already defined:
# - sample_initial_discrete_SU3(L, s; rng=...)
# - Z_symbol(params, xj)
# - X_symbol(xj)

"""
    check_initial_ensemble_discrete(params, s; ntraj=5000, seed=1)

Check discrete-Wigner initial sampling at t=0:
- mean populations P0,P1,P2 at each site
- mean ⟨Z_j⟩ and ⟨X_j⟩
- normalization errors: P0+P1+P2-1
"""
function check_initial_ensemble_discrete(params::PTWAParams, s::Vector{Int};
                                         ntraj::Int=5000, seed::Int=1,
                                         do_plot::Bool=true)

    L = params.L
    @assert length(s) == L
    rng = MersenneTwister(seed)

    Pmean = zeros(Float64, L, 3)
    Zmean = zeros(ComplexF64, L)
    Xmean = zeros(ComplexF64, L)
    norm_err = zeros(Float64, L)

    # collect per-trajectory norms to see if any samples are crazy
    norm_err_max = 0.0

    for tr in 1:ntraj
        #x = sample_initial_discrete_SU3(L, s; rng=rng)
        x = sample_initial_discrete_WH(L, s; rng=rng)


        for j in 1:L
            p0 = real(x[j][1,1])
            p1 = real(x[j][2,2])
            p2 = real(x[j][3,3])

            Pmean[j,1] += p0
            Pmean[j,2] += p1
            Pmean[j,3] += p2

            Zmean[j] += Z_symbol(params, x[j])
            Xmean[j] += X_symbol(x[j])

            e = (p0 + p1 + p2) - 1.0
            norm_err[j] += e
            norm_err_max = max(norm_err_max, abs(e))
        end
    end

    Pmean ./= ntraj
    Zmean ./= ntraj
    Xmean ./= ntraj
    norm_err ./= ntraj

    println("=== Discrete-Wigner initial ensemble checks ===")
    println("ntraj = $ntraj")
    println("max |(P0+P1+P2)-1| across all samples/sites ≈ $norm_err_max")

    # Check a few representative sites
    j0 = (L + 1) ÷ 2
    for j in (1, j0, L)
        println("\nsite j=$j (target state |$(s[j])⟩):")
        println("  mean P0,P1,P2 = ", Pmean[j,1], ", ", Pmean[j,2], ", ", Pmean[j,3],
                "   (sum=", sum(Pmean[j,:]), ")")
        println("  mean ⟨Z⟩ = ", Zmean[j], "   (Re=", real(Zmean[j]), ", Im=", imag(Zmean[j]), ")")
        println("  mean ⟨X⟩ = ", Xmean[j])
    end

    if do_plot
        # Plot populations at t=0 (mean across trajectories)
        pltP = plot(1:L, Pmean[:,1], label="P0", xlabel="site j", ylabel="mean at t=0",
                    title="Discrete-Wigner: initial populations (mean)")
        plot!(pltP, 1:L, Pmean[:,2], label="P1")
        plot!(pltP, 1:L, Pmean[:,3], label="P2")
        savefig(pltP, "init_discrete_populations.pdf")

        # Plot Re⟨Z⟩ and Im⟨Z⟩
        pltZ = plot(1:L, real.(Zmean), label="Re⟨Z⟩", xlabel="site j",
                    title="Discrete-Wigner: initial ⟨Z⟩ (mean)")
        plot!(pltZ, 1:L, imag.(Zmean), label="Im⟨Z⟩")
        savefig(pltZ, "init_discrete_Zmean.pdf")

        # Plot ⟨X⟩ (should be ~0 mean)
        pltX = plot(1:L, real.(Xmean), label="Re⟨X⟩", xlabel="site j",
                    title="Discrete-Wigner: initial ⟨X⟩ (mean)")
        plot!(pltX, 1:L, imag.(Xmean), label="Im⟨X⟩")
        savefig(pltX, "init_discrete_Xmean.pdf")

        println("\nSaved: init_discrete_populations.pdf, init_discrete_Zmean.pdf, init_discrete_Xmean.pdf")
    end

    return Pmean, Zmean, Xmean, norm_err
end


# ------------------------------- Example usage -------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    L = 13
    α = 3.0
    params = PTWAParams(L; J=1.0, α=α, f=0.8 + 0im)

    # single excitation at center: |0...0 1 0...0|
    s = init_single_excitation(L; background=0, excitation=1, pos=(L+1)÷2)
    Pmean, Zmean, Xmean, norm_err = check_initial_ensemble_discrete(params, s; ntraj=10000, seed=7)

    # Match ED time step and horizon
    t, Zm, Xm, Pm = run_ptwa_ZXP(params; s=s, ntraj=5000, tmax=4.0, dt=0.05, seed=123, save_every=1)

    # Save data
    meta = Dict("L"=>L, "alpha"=>α, "f"=>params.f, "ntraj"=>5000,
                "initial_state"=>"single excitation center", "dt"=>0.05, "tmax"=>4.0)
    @save "pTWA_Z3_singleexc_L$(L)_alpha$(α).jld2" t Zm Xm Pm meta

    # Save heatmaps (ED-style)
    save_ptwa_heatmaps(t, Zm, Pm; prefix="pTWA", L=L, α=α)

    println("Saved: pTWA ReZ + P0/P1/P2 heatmaps and JLD2 data.")
end
