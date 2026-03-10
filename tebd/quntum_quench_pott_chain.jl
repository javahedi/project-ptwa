using LinearAlgebra
using TensorOperations
using ProgressMeter
using DelimitedFiles
using Plots

###########################################################
# Helpers: NumPy-style matrix <-> tensor map
###########################################################

function mat_to_tensor4(M::AbstractMatrix{<:Complex})
    return permutedims(reshape(M, 3, 3, 3, 3), (2, 1, 4, 3))
end

function tensor4_to_mat(T::AbstractArray{<:Complex,4})
    return reshape(permutedims(T, (2, 1, 4, 3)), 9, 9)
end

###########################################################
# Potts operators
###########################################################

function potts_ops()
    w = exp(2im * π / 3)

    Z = ComplexF64[
        1 0 0
        0 w 0
        0 0 w^2
    ]

    X = ComplexF64[
        0 0 1
        1 0 0
        0 1 0
    ]

    I3 = Matrix{ComplexF64}(I, 3, 3)
    return Z, X, I3
end

###########################################################
# Bond Hamiltonians
###########################################################

function potts_bond_hamiltonian(J, f, L)
    Z, X, I3 = potts_ops()
    Zh = adjoint(Z)
    Xh = adjoint(X)

    H_bond = Vector{Array{ComplexF64,4}}(undef, L - 1)

    for i in 1:(L - 1)
        Hmat = zeros(ComplexF64, 9, 9)

        Hmat .+= -J .* (kron(Zh, Z) + kron(Z, Zh))

        wl = (i == 1)     ? 1.0 : 0.5
        wr = (i == L - 1) ? 1.0 : 0.5

        Hmat .+= -wl * f .* (kron(X, I3) + kron(Xh, I3))
        Hmat .+= -wr * f .* (kron(I3, X) + kron(I3, Xh))

        H_bond[i] = mat_to_tensor4(Hmat)
    end

    return H_bond
end

function potts_expH(J, f, L, dt; imag=false)
    H = potts_bond_hamiltonian(J, f, L)
    gates = Vector{Array{ComplexF64,4}}(undef, length(H))

    for i in eachindex(H)
        hmat = tensor4_to_mat(H[i])
        Umat = imag ? exp(-dt * hmat) : exp(-1im * dt * hmat)
        gates[i] = mat_to_tensor4(Umat)
    end

    return gates
end

###########################################################
# Initial MPS
###########################################################

function initialMPS(L, d)
    G = [zeros(ComplexF64, d, 1, 1) for _ in 1:L]
    l = [ones(Float64, 1) for _ in 1:(L + 1)]

    for i in 1:L
        G[i][1, 1, 1] = 1.0 + 0im
    end

    return l, G
end

function safe_inv(x; eps=1e-14)
    y = zeros(ComplexF64, length(x))
    for i in eachindex(x)
        if abs(x[i]) > eps
            y[i] = 1.0 / x[i]
        end
    end
    return y
end

###########################################################
# Helpers for scaling MPS tensors by Schmidt values
###########################################################

function left_scale_gamma(lvec, G)
    d, χL, χR = size(G)
    out = similar(G)
    for a in 1:χL
        out[:, a, :] .= lvec[a] .* G[:, a, :]
    end
    return out
end

function right_scale_gamma(G, lvec)
    d, χL, χR = size(G)
    out = similar(G)
    for b in 1:χR
        out[:, :, b] .= G[:, :, b] .* lvec[b]
    end
    return out
end

###########################################################
# Two-site TEBD
###########################################################

function tebd_step!(l, G, gates; chi_max=50)
    L = length(G)

    for parity in (0, 1)
        start_idx = (parity == 0) ? 1 : 2

        for i in start_idx:2:(L - 1)
            d    = size(G[i], 1)
            χ0   = size(G[i], 2)
            χ1   = size(G[i], 3)
            χ2   = size(G[i + 1], 3)

            l0 = l[i]
            l1 = l[i + 1]
            l2 = l[i + 2]

            l0inv = safe_inv(l0)
            l2inv = safe_inv(l2)

            # theta[a,s,t,b]
            A = left_scale_gamma(l0, G[i])     # (s,a,c)
            B = right_scale_gamma(A, l1)       # (s,a,c)

            @tensor C[s,t,a,b] := B[s,a,c] * G[i+1][t,c,b]

            theta = Array{ComplexF64}(undef, χ0, d, d, χ2)
            for b in 1:χ2
                theta[:, :, :, b] .= permutedims(C[:, :, :, b], (3, 1, 2)) .* l2[b]
            end

            # apply gate
            # gates[i] has indices (s_out, t_out, s_in, t_in)
            @tensor phi[s,t,a,b] := gates[i][s,t,x,y] * theta[a,x,y,b]

            # reorder to (a,s,t,b), then reshape
            phi = permutedims(phi, (3, 1, 2, 4))
            phi_mat = reshape(phi, χ0 * d, d * χ2)

            F = svd(phi_mat)
            keep = min(length(F.S), chi_max)

            U  = F.U[:, 1:keep]
            S  = F.S[1:keep]
            Vt = F.Vt[1:keep, :]

            nrm = norm(S)
            if nrm > 0
                S ./= nrm
            end

            l[i + 1] = Float64.(S)

            # update G[i]
            Uten = reshape(U, χ0, d, keep)
            Uten = permutedims(Uten, (2, 1, 3))   # (s,a,α)
            for a in 1:χ0
                Uten[:, a, :] .*= l0inv[a]
            end
            G[i] = ComplexF64.(Uten)

            # update G[i+1]
            Vten = reshape(Vt, keep, d, χ2)       # (α,t,b)
            Vten = permutedims(Vten, (2, 1, 3))   # (t,α,b)
            for b in 1:χ2
                Vten[:, :, b] .*= l2inv[b]
            end
            G[i + 1] = ComplexF64.(Vten)
        end
    end

    return nothing
end

###########################################################
# Dense state reconstruction for overlap
# reliable for moderate L, not for huge L
###########################################################

function mps_to_state(l, G)
    psi = ComplexF64[1.0 + 0im;;]  # 1×1 matrix

    for i in 1:length(G)
        psi = psi * Diagonal(ComplexF64.(l[i]))   # (..., χ_i)
        psi = reshape(psi, :, size(psi, 2))

        χleft = size(psi, 2)
        d, χL, χR = size(G[i])
        @assert χleft == χL

        tmp = Array{ComplexF64}(undef, size(psi, 1), d, χR)
        @tensor tmp[a,s,b] := psi[a,c] * G[i][s,c,b]
        psi = reshape(tmp, :, χR)
    end

    psi = psi * Diagonal(ComplexF64.(l[length(G) + 1]))
    return vec(psi)
end

###########################################################
# Efficient overlap
###########################################################

function overlap(l1, G1, l2, G2)
    E = ComplexF64[1.0 + 0im;;]  # 1×1

    L = length(G1)

    for i in 1:L
        A = left_scale_gamma(l1[i], G1[i])   # (d, χL1, χR1)
        B = left_scale_gamma(l2[i], G2[i])   # (d, χL2, χR2)

        # E: (χL1, χL2)
        # A*: (d, χL1, χR1)
        # B : (d, χL2, χR2)

        @tensor Enew[r1,r2] := E[l1i,l2i] * conj(A[s,l1i,r1]) * B[s,l2i,r2]
        E = Enew
    end

    EL = Diagonal(ComplexF64.(l1[L + 1]))
    ER = Diagonal(ComplexF64.(l2[L + 1]))

    val = EL' * E * ER
    return val[1,1]
end

###########################################################
# Main DQPT simulation
###########################################################

function main()
    L = 50
    J = 1.0

    f0 = 10.0
    f1 = 0.0

    dt = 0.02
    steps = 200

    d = 3
    chi_max = 50

    ###########################################################
    # prepare ground state (imaginary time)
    ###########################################################

    l, G = initialMPS(L, d)

    gates0 = potts_expH(J, f0, L, 0.01; imag=true)

    println("Preparing ground state with imaginary time evolution...")
    @showprogress "Imaginary time evolution " for _ in 1:1000
        tebd_step!(l, G, gates0; chi_max=chi_max)
    end

    psi0_l = deepcopy(l)
    psi0_G = deepcopy(G)

    ###########################################################
    # quench evolution
    ###########################################################

    gates1 = potts_expH(J, f1, L, dt; imag=false)

    times = Float64[]
    rate  = Float64[]

    println("Starting quench evolution...")
    @showprogress "Quench evolution " for step in 1:steps
        tebd_step!(l, G, gates1; chi_max=chi_max)

        Gt = overlap(psi0_l, psi0_G, l, G)
        Ival = -(1 / L) * log(abs(Gt)^2)

        push!(times, step * dt * J)
        push!(rate, real(Ival))
    end

    ###########################################################
    # Save data
    ###########################################################

    data = hcat(times, rate)
    filename = "dqpt_potts_L$(L)_f0$(round(f0, digits=2))_f1$(round(f1, digits=2))_julia.dat"

    open(filename, "w") do io
        write(io, "# Jt   I(t)\n")
        writedlm(io, data)
    end

    println("Saved data to: $filename")

    ###########################################################
    # Plot rate function
    ###########################################################

    plt = plot(
        times, rate,
        marker = :circle,
        linewidth = 2,
        xlabel = "J t",
        ylabel = "I(t)",
        title = "DQPT rate function (3-state Potts chain)",
        label = false,
        size = (600, 400)
    )

    display(plt)
end

main()