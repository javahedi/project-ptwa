using LinearAlgebra
using SparseArrays
using KrylovKit
using JLD2

const J = 1.0
const g = 0.5
const tmax = 5.0
const Nt = 101
const N_top = 30
const ns_top = collect(3:7)
const n_bottom = 5
const Ns_bottom = collect(10:10:80)
const OUTDIR = joinpath(@__DIR__, "data", "ed")

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
            build!(state, pos+1, remaining-k)
        end
    end
    build!(zeros(Int,n), 1, N)
    basis
end

function build_clock_LMG(N::Int, n::Int; J=1.0, g=0.5)
    basis = symmetric_basis(N,n)
    index = Dict(Tuple(b)=>i for (i,b) in enumerate(basis))
    rows = Int[]; cols = Int[]; vals = ComplexF64[]
    ω = cis(2π/n)

    for (col,state) in enumerate(basis)
        zsum = sum(state[a+1]*ω^a for a in 0:n-1)
        E = -(J/N)*(abs2(zsum)-N)
        push!(rows,col); push!(cols,col); push!(vals,E)

        for a in 0:n-1
            b = mod(a+1,n)
            Na, Nb = state[a+1], state[b+1]

            if Na > 0
                new = copy(state)
                new[a+1] -= 1; new[b+1] += 1
                push!(rows,index[Tuple(new)]); push!(cols,col)
                push!(vals,-g*sqrt(Na*(Nb+1)))
            end

            if Nb > 0
                new = copy(state)
                new[b+1] -= 1; new[a+1] += 1
                push!(rows,index[Tuple(new)]); push!(cols,col)
                push!(vals,-g*sqrt(Nb*(Na+1)))
            end
        end
    end
    sparse(rows,cols,vals,length(basis),length(basis)), basis
end

function mZ_diagonal(basis,n,N)
    ω = cis(2π/n)
    [sum((state[a+1]/N)*ω^a for a in 0:n-1) for state in basis]
end

function polarized_state(basis,N)
    ψ = zeros(ComplexF64,length(basis))
    i = findfirst(b -> b[1] == N, basis)
    isnothing(i) && error("Initial state not found")
    ψ[i] = 1
    ψ
end

function run_case(N,n)
    mkpath(OUTDIR)
    times = collect(range(0.0,tmax,length=Nt))
    H,basis = build_clock_LMG(N,n; J=J,g=g)
    ψ = polarized_state(basis,N)
    mdiag = mZ_diagonal(basis,n,N)
    mZ = zeros(ComplexF64,Nt)
    normψ = zeros(Float64,Nt)

    tprev = 0.0
    for (k,t) in enumerate(times)
        dt = t-tprev
        if dt != 0
            ψ, = exponentiate(H,-1im*dt,ψ)
        end
        prob = abs2.(ψ)
        mZ[k] = sum(prob .* mdiag)
        normψ[k] = norm(ψ)
        tprev = t
    end

    outfile = joinpath(OUTDIR,"ed_LMG_N$(N)_n$(n).jld2")
    @save outfile N n J g times mZ normψ
    println("saved -> $outfile")
end

cases = Set{Tuple{Int,Int}}()
foreach(n -> push!(cases,(N_top,n)), ns_top)
foreach(N -> push!(cases,(N,n_bottom)), Ns_bottom)
for (N,n) in sort!(collect(cases))
    println("ED: N=$N n=$n")
    run_case(N,n)
end
