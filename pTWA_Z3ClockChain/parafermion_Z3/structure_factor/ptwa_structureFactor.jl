using LinearAlgebra
using FFTW
using Random
using JLD2
using Printf
using ProgressMeter

include("sampling_Zn.jl")
using .Sampling

############################################################
# Choose sampling method
############################################################

sampling = :gaussian

############################################################
# Parameters
############################################################

struct PTWAParams
    L::Int
    n::Int
    J::Float64
    Jnn::Float64
end

############################################################
# Local operators
############################################################

function zn_ops(n::Int)

    ω = cis(2π / n)

    Z = Diagonal([ω^(a-1) for a in 1:n]) |> Matrix{ComplexF64}

    B = zeros(ComplexF64, n, n)
    for a in 2:n
        B[a-1, a] = 1.0
    end

    Bd = Matrix(adjoint(B))   # ← FIX

    proj = [zeros(ComplexF64, n, n) for _ in 1:n]
    for a in 1:n
        proj[a][a, a] = 1.0
    end

    return B, Bd, Z, proj
end

############################################################
# Local pTWA field for
# H = -J Σ_j (B_j† Z_j B_{j+1} + h.c.)
############################################################

function build_h!(h::Vector{Matrix{ComplexF64}},
                  x::Vector{Matrix{ComplexF64}},
                  p::PTWAParams,
                  B::Matrix{ComplexF64},
                  Bd::Matrix{ComplexF64},
                  Z::Matrix{ComplexF64})

    L = p.L
    J = p.Jnn

    C  = Bd * Z
    Cd = adjoint(C)

    # classical symbols
    b = zeros(ComplexF64, L)   # Tr(x_j B)
    c = zeros(ComplexF64, L)   # Tr(x_j Bd Z)

    for j in 1:L
        xj = x[j]
        b[j] = tr(xj * B)
        c[j] = tr(xj * C)
    end

    for j in 1:L

        hj = h[j]
        fill!(hj, 0.0 + 0.0im)

        # bond (j, j+1)
        if j < L
            hj .+= -J * ( b[j+1] * C +
                          conj(b[j+1]) * Cd )
        end

        # bond (j-1, j)
        if j > 1
            hj .+= -J * ( c[j-1] * B +
                          conj(c[j-1]) * Bd )
        end
    end

end


function rhs!(dx::Vector{Matrix{ComplexF64}},
              x::Vector{Matrix{ComplexF64}},
              h::Vector{Matrix{ComplexF64}},
              p::PTWAParams,
              B::Matrix{ComplexF64},
              Bd::Matrix{ComplexF64},
              Z::Matrix{ComplexF64})

    build_h!(h, x, p, B, Bd, Z)

    for j in 1:p.L
        dx[j] .= 1im .* (h[j] * x[j] - x[j] * h[j])
    end

end
############################################################
# RK4 time evolution
############################################################

function evolve_times_ptwa(x0,p,times,B,Bd,Z)

    L = p.L
    n = p.n

    x = [copy(x0[j]) for j in 1:L]

    h  = [zeros(ComplexF64, n, n) for _ in 1:L]
    k1 = [zeros(ComplexF64, n, n) for _ in 1:L]
    k2 = [zeros(ComplexF64, n, n) for _ in 1:L]
    k3 = [zeros(ComplexF64, n, n) for _ in 1:L]
    k4 = [zeros(ComplexF64, n, n) for _ in 1:L]
    xt = [zeros(ComplexF64, n, n) for _ in 1:L]

    out = Vector{Vector{Matrix{ComplexF64}}}(undef, length(times))
    out[1] = [copy(x[j]) for j in 1:L]

   for k in 2:length(times)

        dt = times[k] - times[k-1]

        rhs!(k1, x, h, p, B, Bd, Z)

        for j in 1:L
            xt[j] .= x[j] .+ (dt/2) .* k1[j]
        end
        rhs!(k2, xt, h, p, B, Bd, Z)

        for j in 1:L
            xt[j] .= x[j] .+ (dt/2) .* k2[j]
        end
        rhs!(k3, xt, h, p, B, Bd, Z)

        for j in 1:L
            xt[j] .= x[j] .+ dt .* k3[j]
        end
        rhs!(k4, xt, h, p, B, Bd, Z)

        for j in 1:L
            x[j] .+= (dt/6) .* (k1[j] .+ 2k2[j] .+ 2k3[j] .+ k4[j])
        end

        out[k] = [copy(x[j]) for j in 1:L]

    end

    return out
end

############################################################
# Measurements
############################################################

function measure_Z_profile(xs::Vector{Vector{Matrix{ComplexF64}}}, n::Int)
    Nt = length(xs)
    L  = length(xs[1])

    _, _, Z, _ = zn_ops(n)

    Zt = zeros(ComplexF64, L, Nt)
    for k in 1:Nt
        for j in 1:L
            Zt[j, k] = tr(xs[k][j] * Z)
        end
    end
    return Zt
end

function measure_Pa_profile(xs::Vector{Vector{Matrix{ComplexF64}}}, a::Int)
    Nt = length(xs)
    L  = length(xs[1])

    ia = a + 1
    Pa = zeros(Float64, L, Nt)

    for k in 1:Nt
        for j in 1:L
            Pa[j, k] = real(xs[k][j][ia, ia])
        end
    end
    return Pa
end




############################################################
# Structure factors
############################################################

function structure_factor_from_profile(vals::AbstractVector)
    L = length(vals)
    F = fft(vals)
    return real.(abs2.(F) ./ L)
end


############################################################
# Domain wall imbalance
############################################################

function imbalance_from_profile(vals::AbstractVector)

    L = length(vals)
    half = div(L,2)

    left  = sum(@view vals[1:half])
    right = sum(@view vals[(half+1):L])

    return (2.0/L) * (left - right)

end

############################################################
# Main run
############################################################
for n in [6,7,8]#[3, 4, 5]
    L     = 24
    J     = 1.0
    Jnn   = 1.0
    dt    = 0.05
    tmax  = 10.0
    times = collect(0.0:dt:tmax)
    Nt    = length(times)

    Ntraj = 2000
    seed  = 1234

    # domain wall matching TEBD:
    # left half in |1>, right half in |n>
    # Gaussian sampler uses 0-based labels:
    # |1> -> 0, |n> -> n-1
    a_left  = 0
    a_right = n - 1

    p = PTWAParams(L, n, J, Jnn)

    # precompute local powers
    B, Bd, Z, proj = zn_ops(n)

   

    qs = 2π .* (0:L-1) ./ L

    # accumulators
    Sa_sum  = zeros(Float64, L, Nt)
    Sa_sum2 = zeros(Float64, L, Nt)

    I_sum   = zeros(Float64, Nt)
    I_sum2  = zeros(Float64, Nt)

    rng = MersenneTwister(seed)

    prog = Progress(Ntraj)

    for tr in 1:Ntraj

        x0 = sample_initial_state_domainwall(
            sampling, n, L;
            a_left=a_left,
            a_right=a_right,
            rng=rng
        )

        xs = evolve_times_ptwa(x0, p, times, B, Bd, Z)

        # projector profile
        P0 = measure_Pa_profile(xs,0)

        Sa = zeros(Float64,L,Nt)
        I  = zeros(Float64,Nt)

        for k in 1:Nt

            profile = view(P0,:,k)

            Sa[:,k] .= structure_factor_from_profile(profile)

            I[k] = imbalance_from_profile(profile)

        end

        Sa_sum  .+= Sa
        Sa_sum2 .+= Sa.^2

        I_sum  .+= I
        I_sum2 .+= I.^2

        next!(prog)

    end

    ############################################################
    # Averages and SEM
    ############################################################

    Sa_avg = Sa_sum ./ Ntraj
    Sa_var = (Sa_sum2 ./ Ntraj) .- Sa_avg.^2
    Sa_err = sqrt.(max.(Sa_var,0.0) ./ Ntraj)

    I_avg = I_sum ./ Ntraj
    I_var = (I_sum2 ./ Ntraj) .- I_avg.^2
    I_err = sqrt.(max.(I_var,0.0) ./ Ntraj)

    println()
    println("pTWA ($(sampling)) n=$n, L=$L")
    println("Imbalance at tmax: ", @sprintf("%.4f ± %.4f", I_avg[end], I_err[end]))
   

    ############################################################
    # Save
    ############################################################

    outfile = @sprintf(
    "pTWA_Zn_domainwall_singlehop_n%d_L%d_%s_N%d.jld2",
    n,L,String(sampling),Ntraj)

    @save outfile n L J dt tmax times qs Ntraj seed sampling a_left a_right Sa_avg Sa_err I_avg I_err

    println("Saved to: ",outfile)

   
end