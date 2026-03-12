# run_ED_Z3_parafermion_MBL_optimized_checkpoint.jl

using LinearMaps
using KrylovKit
using Random
using JLD2
using Base.Threads
using LinearAlgebra
using Printf

BLAS.set_num_threads(1)

# ----------------------------
# Model/constants
# ----------------------------
const n = 3
const ω = cis(2π / n)
const SQRT2 = sqrt(2.0)

const ωpow1 = (1.0 + 0im, ω, ω^2)
const ωpow2 = (1.0 + 0im, ω^2, ω)

@inline pref_lower_from_digit(a::Int) = a == 0 ? 0.0 : (a == 1 ? 1.0 : SQRT2)
@inline pref_raise_from_digit(a::Int) = a == 2 ? 0.0 : (a == 0 ? 1.0 : SQRT2)

# ----------------------------
# Parameters
# ----------------------------
struct PFDisorderParams
    L::Int
    J::Float64
    g::Float64
    W::Float64
    mu::Vector{Float64}
    powN::Vector{Int}
end

# ----------------------------
# Initial state: domain wall |1..1,0..0>
# ----------------------------
function domainwall_state(L::Int, powN::Vector{Int})
    s = 0
    half = L ÷ 2
    @inbounds for j in 1:half
        s += powN[j]
    end
    return s
end

# ----------------------------
# H*v (matrix-free)
# ----------------------------
function mul_H!(y::Vector{ComplexF64}, v::Vector{ComplexF64}, p::PFDisorderParams)
    fill!(y, 0.0 + 0.0im)
    dim = length(v)
    L   = p.L
    J   = p.J
    g   = p.g
    powN = p.powN
    mu   = p.mu

    t1 = -J * (1 - g)
    t2 = -J * g

    a = Vector{UInt8}(undef, L)

    @inbounds for idx in 1:dim
        vs = v[idx]
        vs == 0 && continue
        s = idx - 1

        tmp = s
        for j in 1:L
            aj = tmp % 3
            a[j] = UInt8(aj)
            tmp ÷= 3
        end

        E = 0.0
        for j in 1:L
            E += mu[j] * Int(a[j])
        end
        y[idx] += E * vs

        for j in 1:(L-1)
            aj  = Int(a[j])
            aj1 = Int(a[j+1])
            nj  = aj

            # f†_j f_{j+1}
            if aj1 > 0 && aj < 2
                s2 = s + powN[j] - powN[j+1]
                amp = t1 * ωpow1[nj+1] * pref_lower_from_digit(aj1) * pref_raise_from_digit(aj)
                y[s2+1] += amp * vs
            end

            # h.c.
            if aj > 0 && aj1 < 2
                s2 = s - powN[j] + powN[j+1]
                amp = t1 * conj(ωpow1[nj+1]) * pref_lower_from_digit(aj) * pref_raise_from_digit(aj1)
                y[s2+1] += amp * vs
            end

            # pair hop
            if aj == 0 && aj1 == 2
                s2 = s + 2*powN[j] - 2*powN[j+1]
                amp = t2 * ωpow2[nj+1] * (SQRT2 * SQRT2)
                y[s2+1] += amp * vs
            end

            # h.c. pair
            if aj == 2 && aj1 == 0
                s2 = s - 2*powN[j] + 2*powN[j+1]
                amp = t2 * conj(ωpow2[nj+1]) * (SQRT2 * SQRT2)
                y[s2+1] += amp * vs
            end
        end
    end
    return y
end

# ----------------------------
# Measurement
# ----------------------------
function measure_n_profile(psi::Vector{ComplexF64}, p::PFDisorderParams)
    L = p.L
    nprof = zeros(Float64, L)
    a = Vector{UInt8}(undef, L)

    @inbounds for idx in 1:length(psi)
        w = abs2(psi[idx])
        w == 0 && continue

        s = idx - 1
        tmp = s
        for j in 1:L
            aj = tmp % 3
            a[j] = UInt8(aj)
            tmp ÷= 3
        end
        for j in 1:L
            nprof[j] += Int(a[j]) * w
        end
    end
    return nprof
end

function imbalance_from_n(nprof::AbstractVector{<:Real})
    L = length(nprof)
    half = L ÷ 2
    NL = sum(@view nprof[1:half])
    NR = sum(@view nprof[half+1:L])
    return (2.0 / L) * (NL - NR)
end

# ----------------------------
# Time evolution + measurement
# ----------------------------
function evolve_and_measure!(I_sum, I_sum2, n_sum, psi0, p, times;
                             krylovdim::Int=60, tol::Float64=1e-10, max_dt::Float64=0.1)

    dim = length(psi0)
    H = LinearMap{ComplexF64}((y, v) -> mul_H!(y, v, p), dim, dim; ismutating=true)

    function exp_step_substepped(ψ, dt)
        nsub = max(1, ceil(Int, abs(dt) / max_dt))
        δ = dt / nsub
        for _ in 1:nsub
            ψ, = exponentiate(H, -1im * δ, ψ; krylovdim=krylovdim, tol=tol)
        end
        return ψ
    end

    psit = copy(psi0)
    tprev = 0.0

    for (k, t) in enumerate(times)
        psit = exp_step_substepped(psit, t - tprev)

        # safety normalization
        psit ./= norm(psit)

        nprof = measure_n_profile(psit, p)
        I = imbalance_from_n(nprof)

        I_sum[k]  += I
        I_sum2[k] += I^2
        n_sum[:, k] .+= nprof

        tprev = t
    end
    return nothing
end

# ----------------------------
# Checkpoint I/O
# ----------------------------
function save_checkpoint(outfile, L, J, gpair, W, Ndis, seed, tmax, times,
                         I_sum, I_sum2, n_sum, r_completed)

    tmpfile = outfile * ".tmp"

    @save tmpfile L J gpair W Ndis seed tmax times I_sum I_sum2 n_sum r_completed
    mv(tmpfile, outfile; force=true)
    return nothing
end

function load_checkpoint_if_exists(outfile, Nt, L)
    if isfile(outfile)
        println("Checkpoint found: $outfile")
        flush(stdout)

        data = load(outfile)

        I_sum = haskey(data, "I_sum") ? data["I_sum"] : zeros(Float64, Nt)
        I_sum2 = haskey(data, "I_sum2") ? data["I_sum2"] : zeros(Float64, Nt)
        n_sum = haskey(data, "n_sum") ? data["n_sum"] : zeros(Float64, L, Nt)
        r_completed = haskey(data, "r_completed") ? data["r_completed"] : 0

        return I_sum, I_sum2, n_sum, r_completed
    else
        return zeros(Float64, Nt), zeros(Float64, Nt), zeros(Float64, L, Nt), 0
    end
end

# ----------------------------
# Main: read args
# ----------------------------
function main()

    L      = parse(Int,     ARGS[1])
    W      = parse(Float64, ARGS[2])
    Ndis   = parse(Int,     ARGS[3])
    tmax   = parse(Float64, ARGS[4])
    seed   = parse(Int,     ARGS[5])
    OUTDIR = ARGS[6]
    L      = parse(Int,     ARGS[1])
    W      = parse(Float64, ARGS[2])
    Ndis   = parse(Int,     ARGS[3])
    tmax   = parse(Float64, ARGS[4])
    seed   = parse(Int,     ARGS[5])
    OUTDIR = ARGS[6]

    J     = 1.0
    gpair = 0.3

    Nt    = 81
    times = collect(range(0.0, tmax, length=Nt))

    krylovdim = 60
    tol       = 1e-11
    max_dt    = 0.05

    # save every this many realizations
    save_every = max(1, min(10, nthreads()))

    # ----------------------------
    # Basis size
    # ----------------------------
    powN = Vector{Int}(undef, L)
    powN[1] = 1
    for j in 2:L
        powN[j] = powN[j-1] * n
    end
    dim = powN[L] * n

    println("L=$L => dim=$dim, Nt=$Nt, Ndis=$Ndis, Threads=$(nthreads()), maxthreadid=$(Threads.maxthreadid())")
    flush(stdout)

    psi0 = zeros(ComplexF64, dim)
    s0   = domainwall_state(L, powN)
    psi0[s0 + 1] = 1.0 + 0im

    # ----------------------------
    # Output / checkpoint file
    # ----------------------------
    mkpath(OUTDIR)
    outfile = "$OUTDIR/ED_Z3_parafermion_L$(L)_g$(gpair)_W$(W)_seed$(seed).jld2"

    I_sum, I_sum2, n_sum, r_completed = load_checkpoint_if_exists(outfile, Nt, L)

    if r_completed >= Ndis
        println("All realizations already completed: r_completed=$r_completed")
        Ndone = Ndis

        I_mean = I_sum ./ Ndone
        n_mean = n_sum ./ Ndone
        I_var = (I_sum2 ./ Ndone) .- I_mean.^2
        I_err = sqrt.(max.(I_var, 0.0) ./ Ndone)

        @save outfile L J gpair W Ndis seed tmax times I_sum I_sum2 n_sum r_completed I_mean I_err n_mean
        println("Saved final averages -> $outfile")
        flush(stdout)
        exit()
    end

    println("Starting / resuming from realization $(r_completed + 1)")
    flush(stdout)

    # ----------------------------
    # Thread-local accumulators for batches
    # ----------------------------
    T = Threads.maxthreadid()

    print_lock = ReentrantLock()

    r_start = r_completed + 1

    while r_start <= Ndis
        r_end = min(r_start + save_every - 1, Ndis)
        batch_size = r_end - r_start + 1

        println("Processing batch: $r_start:$r_end")
        flush(stdout)

        I_sum_tls  = [zeros(Float64, Nt) for _ in 1:T]
        I_sum2_tls = [zeros(Float64, Nt) for _ in 1:T]
        n_sum_tls  = [zeros(Float64, L, Nt) for _ in 1:T]

        @threads :static for r in r_start:r_end
            tid = threadid()

            # one RNG per realization => reproducible even after restart
            rng = MersenneTwister(seed + 100000*r)

            mu = (2W) .* rand(rng, L) .- W
            p  = PFDisorderParams(L, J, gpair, W, mu, powN)

            evolve_and_measure!(I_sum_tls[tid], I_sum2_tls[tid], n_sum_tls[tid],
                                psi0, p, times; krylovdim=krylovdim, tol=tol, max_dt=max_dt)

            lock(print_lock) do
                println("done realization $r / $Ndis on thread $tid")
                flush(stdout)
            end
        end

        # reduce this batch into global accumulators
        for tid in 1:T
            I_sum  .+= I_sum_tls[tid]
            I_sum2 .+= I_sum2_tls[tid]
            n_sum  .+= n_sum_tls[tid]
        end

        r_completed = r_end

        save_checkpoint(outfile, L, J, gpair, W, Ndis, seed, tmax, times,
                        I_sum, I_sum2, n_sum, r_completed)

        println("Checkpoint saved at r_completed=$r_completed / $Ndis -> $outfile")
        flush(stdout)

        r_start = r_end + 1
    end

    # ----------------------------
    # Final averages
    # ----------------------------
    Ndone = r_completed

    I_mean = I_sum ./ Ndone
    n_mean = n_sum ./ Ndone

    I_var = (I_sum2 ./ Ndone) .- I_mean.^2
    I_err = sqrt.(max.(I_var, 0.0) ./ Ndone)

    println("Imbalance I(t_final) = ", I_mean[end], " ± ", I_err[end])
    flush(stdout)

    @save outfile L J gpair W Ndis seed tmax times I_sum I_sum2 n_sum r_completed I_mean I_err n_mean
    println("Final save -> $outfile")
    flush(stdout)
end

main()