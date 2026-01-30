using LinearAlgebra
using ParafermionDynamic
using Plots, JLD2, Statistics


# -----------------------------
# Model parameters
# -----------------------------
L = 9
n = 3

J  = 1.0
g  = 0.0
α  = 1.0
mu = zeros(L)

# Build model in full Hilbert space
model = build_model(L; n=n,
                    hopping=long_range_hopping(L, J, α),
                    pair_hopping=long_range_hopping(L, g, α),
                    mu=mu)

println("Full basis size: ", n^L)

# -----------------------------
# Initial state |+X> product state
# -----------------------------
ψ0 = plusX_state(L, n)

# -----------------------------
# Time evolution
# -----------------------------
dt    = 0.1
steps = 80
ψt    = copy(ψ0)

# -----------------------------
# Initialize observables
# -----------------------------
Zj_time = zeros(ComplexF64, L, steps+1)
Xj_time = zeros(ComplexF64, L, steps+1)
Nj_time = zeros(Float64,   L, steps+1)

# Site-resolved quantum variance arrays
Zvar_time = zeros(Float64, L, steps+1)
Xvar_time = zeros(Float64, L, steps+1)
Nvar_time = zeros(Float64, L, steps+1)

# Initial values
Zj_time[:,1] = local_Z(ψt, model)
Xj_time[:,1] = local_X(ψt, model)
Nj_time[:,1] = local_occupation(ψt, model)

for j in 1:L
    Zvar_time[j,1] = real(Zcorr_1(ψt, model, j, j)) - abs(Zj_time[j,1])^2
    Xvar_time[j,1] = real(Xcorr_1(ψt, model, j, j)) - abs(Xj_time[j,1])^2
    Nvar_time[j,1] = Nj_time[j,1]*(1 - Nj_time[j,1])  # simple discrete variance for occupation
end

# -----------------------------
# Time loop
# -----------------------------
for t in 1:steps
    global ψt = krylov_time_evolve(ψt, dt, apply_H!, model; kry_m=20)

    # Observables
    Zj_time[:, t+1] = local_Z(ψt, model)
    Xj_time[:, t+1] = local_X(ψt, model)
    Nj_time[:, t+1] = local_occupation(ψt, model)

    for j in 1:L
        Zvar_time[j, t+1] = real(Zcorr_1(ψt, model, j, j)) - abs(Zj_time[j, t+1])^2
        Xvar_time[j, t+1] = real(Xcorr_1(ψt, model, j, j)) - abs(Xj_time[j, t+1])^2
        Nvar_time[j, t+1] = Nj_time[j, t+1]*(1 - Nj_time[j, t+1])
    end

    println("Time step $t done.")
end

Xvar_time = max.(Xvar_time, 0.0)
Zvar_time = max.(Zvar_time, 0.0)
Nvar_time = max.(Nvar_time, 0.0)


# -----------------------------
# Averages over sites
# -----------------------------
Z_ave = vec(mean(real.(Zj_time), dims=1))
Z_var = vec(mean(Zvar_time, dims=1))

X_ave = vec(mean(real.(Xj_time), dims=1))
X_var = vec(mean(Xvar_time, dims=1))

N_ave = vec(mean(Nj_time, dims=1))
N_var = vec(mean(Nvar_time, dims=1))

# -----------------------------
# Plot results
# -----------------------------
t_axis = (0:steps) .* dt

plot(t_axis, X_ave;
     ribbon = sqrt.(X_var),
     xlabel = "Time",
     ylabel = "⟨X⟩ (avg over sites)",
     title = "Z₃ Parafermion Chain Dynamics",
     label = "mean ± ΔX",
     lw = 2)
savefig("X3_quantum_variance.png")

plot(t_axis, Z_ave;
     ribbon = sqrt.(Z_var),
     xlabel = "Time",
     ylabel = "⟨Z⟩ (avg over sites)",
     title = "Z₃ Parafermion Chain Dynamics",
     label = "mean ± ΔZ",
     lw = 2)
savefig("Z3_quantum_variance.png")

plot(t_axis, N_ave;
     ribbon = sqrt.(N_var),
     xlabel = "Time",
     ylabel = "⟨N⟩ (avg over sites)",
     title = "Z₃ Parafermion Chain Dynamics",
     label = "mean ± ΔN",
     lw = 2)
savefig("N3_quantum_variance.png")


# Optionally save data
# JLD2.@save "data_exact_L9.jld2" t_axis Z_ave Z_var X_ave X_var N_ave N_var
