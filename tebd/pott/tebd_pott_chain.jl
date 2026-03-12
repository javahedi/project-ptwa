using LinearAlgebra
using TensorOperations

###############################################
# Helpers: NumPy-style matrix <-> tensor map
###############################################

function mat_to_tensor4(M::AbstractMatrix{<:Complex})
    return permutedims(reshape(M, 3, 3, 3, 3), (2, 1, 4, 3))
end

function tensor4_to_mat(T::AbstractArray{<:Complex,4})
    return reshape(permutedims(T, (2, 1, 4, 3)), 9, 9)
end

###############################################
# Local Potts operators
###############################################

function potts_operators()
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

###############################################
# Bond Hamiltonians
###############################################

function potts_H_bond(J, f, L)
    Z, X, I3 = potts_operators()
    Zh = adjoint(Z)
    Xh = adjoint(X)

    H_bond = Vector{Array{ComplexF64,4}}(undef, L - 1)

    for i in 1:(L - 1)
        Hmat = zeros(ComplexF64, 9, 9)

        Hmat .+= -J[i] .* (kron(Zh, Z) + kron(Z, Zh))

        lw = (i == 1)     ? 1.0 : 0.5
        rw = (i == L - 1) ? 1.0 : 0.5

        Hmat .+= -lw * f[i]   .* (kron(X, I3) + kron(Xh, I3))
        Hmat .+= -rw * f[i+1] .* (kron(I3, X) + kron(I3, Xh))

        H_bond[i] = mat_to_tensor4(Hmat)
    end

    return H_bond
end

function potts_expH_bond(J, f, L, δ)
    H_bond = potts_H_bond(J, f, L)
    expH_bond = Vector{Array{ComplexF64,4}}(undef, L - 1)

    for i in 1:(L - 1)
        Hmat = tensor4_to_mat(H_bond[i])
        Umat = exp(-δ * Hmat)
        expH_bond[i] = mat_to_tensor4(Umat)
    end

    return expH_bond, H_bond
end

###############################################
# Initial MPS in Vidal form
###############################################

function initial_mat(L, d)
    G = [zeros(ComplexF64, d, 1, 1) for _ in 1:L]
    l = [ones(Float64, 1) for _ in 1:(L + 1)]

    for i in 1:L
        G[i][1, 1, 1] = 1.0 + 0im
    end

    return l, G
end

###############################################
# Utilities
###############################################

function safe_inv(v; cutoff=1e-14)
    out = zeros(Float64, length(v))
    for i in eachindex(v)
        if abs(v[i]) > cutoff
            out[i] = 1.0 / v[i]
        end
    end
    return out
end

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

###############################################
# One TEBD sweep
###############################################

function tebd_sweep_parity!(l, G, expH, parity, chiMax)
    L = length(G)
    start_idx = (parity == 0) ? 1 : 2

    for i in start_idx:2:(L - 1)
        χL = size(G[i], 2)
        χM = size(G[i], 3)
        χR = size(G[i+1], 3)
        d = size(G[i], 1)

        invL = safe_inv(l[i])
        invR = safe_inv(l[i+2])

        # Build theta[a,s,t,b] in steps
        A = left_scale_gamma(l[i], G[i])         # (s,a,c)
        B = right_scale_gamma(A, l[i+1])         # (s,a,c)

        @tensor C[s,t,a,b] := B[s,a,c] * G[i+1][t,c,b]

        theta = Array{ComplexF64}(undef, χL, d, d, χR)
        for b in 1:χR
            theta[:, :, :, b] .= permutedims(C[:, :, :, b], (3, 1, 2)) .* l[i+2][b]
        end
        # theta[a,s,t,b]

        # Apply gate
        @tensor phi[a,s,t,b] := theta[a,x,y,b] * expH[i][x,y,s,t]

        # Reshape for SVD: rows=(a,s), cols=(t,b)
        phi_mat = reshape(phi, χL * d, d * χR)

        F = svd(phi_mat)
        keep = min(length(F.S), chiMax)

        U  = F.U[:, 1:keep]
        S  = F.S[1:keep]
        Vt = F.Vt[1:keep, :]

        nrm = norm(S)
        if nrm > 0
            S ./= nrm
        end
        l[i+1] = S

        # Left Gamma update
        Uten = reshape(U, χL, d, keep)
        Uten = permutedims(Uten, (2, 1, 3))  # (s,a,α)
        for a in 1:χL
            Uten[:, a, :] .*= invL[a]
        end
        G[i] = ComplexF64.(Uten)

        # Right Gamma update
        Vten = reshape(Vt, keep, d, χR)      # (α,t,b)
        Vten = permutedims(Vten, (2, 1, 3))  # (t,α,b)
        for b in 1:χR
            Vten[:, :, b] .*= invR[b]
        end
        G[i+1] = ComplexF64.(Vten)
    end

    return nothing
end

###############################################
# Energy measurement
###############################################

function bond_energy(l, G, H_bond, i)
    A = left_scale_gamma(l[i], G[i])
    B = right_scale_gamma(A, l[i+1])

    @tensor C[s,t,a,b] := B[s,a,c] * G[i+1][t,c,b]

    χR = size(C, 4)
    χL = size(C, 3)
    d1 = size(C, 1)
    d2 = size(C, 2)

    theta = Array{ComplexF64}(undef, χL, d1, d2, χR)
    for b in 1:χR
        theta[:, :, :, b] .= permutedims(C[:, :, :, b], (3, 1, 2)) .* l[i+2][b]
    end

    @tensor rho[s,t,s2,t2] := conj(theta[a,s,t,b]) * theta[a,s2,t2,b]

    return sum(rho .* H_bond[i])
end

function energy_measurement(l, G, H_bond)
    E_total = 0.0
    for i in 1:(length(G) - 1)
        E_total += real(bond_energy(l, G, H_bond, i))
    end
    return E_total
end

###############################################
# Main
###############################################

function main()
    L = 7
    d = 3
    chiMax = 30
    delta = 0.01
    steps = 1000

    J = ones(Float64, L)
    f = ones(Float64, L)

    l, G = initial_mat(L, d)
    expH_half, H_list = potts_expH_bond(J, f, L, delta / 2)
    expH_full, _      = potts_expH_bond(J, f, L, delta)

    println("Starting TEBD for L=$L Potts chain...")

    for s in 1:steps
        tebd_sweep_parity!(l, G, expH_half, 0, chiMax)
        tebd_sweep_parity!(l, G, expH_full, 1, chiMax)
        tebd_sweep_parity!(l, G, expH_half, 0, chiMax)

        if s % 100 == 0
            E = energy_measurement(l, G, H_list)
            println("Step $s | Energy: $(round(E, digits=12))")
        end
    end

    final_E = energy_measurement(l, G, H_list)
    println("\nFinal Energy: $final_E/L = $(final_E/L)")
    println("Exact Energy /L: -2.355433747522252")
    println("Difference:   $(final_E/L - (-2.355433747522252))")
end

main()
