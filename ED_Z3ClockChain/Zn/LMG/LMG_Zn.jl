using LinearAlgebra
using SparseArrays
using KrylovKit
using JLD2

function symmetric_basis(N::Int, n::Int)
    basis = Vector{Vector{Int}}()

    function build!(state, pos, remaining)
        if pos == n
            state[pos] = remaining
            push!(basis, copy(state))
            return
        end
        for k in 0:remaining
            state[pos] = k
            build!(state, pos + 1, remaining - k)
        end
    end

    build!(zeros(Int, n), 1, N)
    return basis
end

function build_clock_LMG(N::Int, n::Int; J::Float64 = 1.0, g::Float64 = 0.5)
    basis = symmetric_basis(N, n)
    dim = length(basis)
    println("dim = ", dim)

    index = Dict{Tuple{Vararg{Int}}, Int}()
    for (i, b) in enumerate(basis)
        index[Tuple(b)] = i
    end

    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]

    ω = cis(2π / n)

    for (col, state) in enumerate(basis)

        # --- exact diagonal interaction energy
        zsum = 0.0 + 0.0im
        for a in 0:n-1
            zsum += state[a+1] * (ω^a)
        end
        E = -(J / N) * (abs2(zsum) - N)

        push!(rows, col)
        push!(cols, col)
        push!(vals, E)

        # --- drive term: -g sum_a (b†_{a+1} b_a + h.c.)
        for a in 0:n-1
            b = mod(a + 1, n)

            Na = state[a+1]
            Nb = state[b+1]

            # forward: a -> b
            if Na > 0
                new = copy(state)
                new[a+1] -= 1
                new[b+1] += 1

                row = index[Tuple(new)]
                amp = -g * sqrt(Na * (Nb + 1))

                push!(rows, row)
                push!(cols, col)
                push!(vals, amp)
            end

            # backward: b -> a
            if Nb > 0
                new = copy(state)
                new[b+1] -= 1
                new[a+1] += 1

                row = index[Tuple(new)]
                amp = -g * sqrt(Nb * (Na + 1))

                push!(rows, row)
                push!(cols, col)
                push!(vals, amp)
            end
        end
    end

    H = sparse(rows, cols, vals, dim, dim)
    return H, basis
end

function build_observables(basis, n::Int, N::Int)
    dim = length(basis)
    ω = cis(2π / n)

    mZ_diag = zeros(ComplexF64, dim)
    pop_diag = zeros(Float64, n, dim)

    for (i, state) in enumerate(basis)
        z = 0.0 + 0.0im
        for a in 0:n-1
            Na = state[a+1]
            frac = Na / N
            pop_diag[a+1, i] = frac
            z += frac * (ω^a)
        end
        mZ_diag[i] = z
    end

    return mZ_diag, pop_diag
end

function measure_observables(ψ, mZ_diag, pop_diag)
    prob = abs2.(ψ)
    mz = sum(prob .* mZ_diag)
    pa = pop_diag * prob
    return mz, pa
end

function initial_state(basis, N::Int)
    ψ0 = zeros(ComplexF64, length(basis))
    for (i, b) in enumerate(basis)
        if b[1] == N
            ψ0[i] = 1.0 + 0im
            return ψ0
        end
    end
    error("Initial state not found")
end

function run_scan(; Ns = [80],#[10, 20, 30, 40, 50, 60, 70],
                    ns = [5],
                    J = 1.0,
                    g = 0.5,
                    tmax = 5.0,
                    Nt = 101,
                    outdir = "clock_LMG_scan")

    mkpath(outdir)
    times = collect(range(0.0, tmax, length = Nt))

    for N in Ns, n in ns
        println("\nRunning N=$N  n=$n")

        H, basis = build_clock_LMG(N, n; J = J, g = g)
        ψ = initial_state(basis, N)
        mZ_diag, pop_diag = build_observables(basis, n, N)

        mz_t = zeros(ComplexF64, Nt)
        pa_t = zeros(Float64, n, Nt)
        norm_t = zeros(Float64, Nt)

        tprev = 0.0

        for (k, t) in enumerate(times)
            dt = t - tprev
            ψ, = exponentiate(H, -1im * dt, ψ)

            mz, pa = measure_observables(ψ, mZ_diag, pop_diag)
            mz_t[k] = mz
            pa_t[:, k] = pa
            norm_t[k] = norm(ψ)

            tprev = t
        end

        outfile = joinpath(outdir, "clock_LMG_N$(N)_n$(n).jld2")
        @save outfile N n J g times mz_t pa_t norm_t
        println("saved -> ", outfile)
    end
end

run_scan()