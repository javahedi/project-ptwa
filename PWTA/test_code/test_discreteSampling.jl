using LinearAlgebra
using Printf

# ---------------- Z3 kernel (same as your code) ----------------

const ω3 = cis(2π/3)
const inv2_mod3 = 2

const Z3 = Diagonal(ComplexF64[ω3^0, ω3^1, ω3^2])
const X3 = ComplexF64[0 0 1;
                      1 0 0;
                      0 1 0]

function Aqp_WH(q::Int, p::Int)
    A = zeros(ComplexF64, 3, 3)
    for m in 0:2, k in 0:2
        phase = mod(p*k - q*m + inv2_mod3*m*k, 3)
        A .+= ω3^phase * (Z3^m * X3^k)
    end
    A ./= 3
    return (A + A') ./ 2   # enforce Hermiticity
end

function precompute_A()
    Ac = Array{ComplexF64,4}(undef, 3,3,3,3)
    for q in 0:2, p in 0:2
        Ac[:,:,q+1,p+1] = Aqp_WH(q,p)
    end
    return Ac
end

# ---------------- Wigner evaluation ----------------

function wigner_table_for_state(a::Int, Ac; tol=1e-12)
    @assert a in 0:2
    ρ = zeros(ComplexF64, 3, 3)
    ρ[a+1,a+1] = 1.0

    W = zeros(Float64, 3, 3)
    for q in 0:2, p in 0:2
        A = Ac[:,:,q+1,p+1]
        W[q+1,p+1] = real(tr(ρ*A)) / 3
    end

    return W, minimum(W)
end

# ---------------- Run test ----------------

function test_basis_state_positivity()
    Ac = precompute_A()
    println("=== Discrete Wigner positivity test (Z₃, WH kernel) ===\n")

    for a in 0:2
        W, wmin = wigner_table_for_state(a, Ac)

        println("State |$a⟩⟨$a|:")
        for q in 1:3
            @printf("  ")
            for p in 1:3
                @printf("% .6f  ", W[q,p])
            end
            println()
        end
        println("  min W = $(wmin)")
        if wmin < -1e-12
            println("  ❌ NEGATIVITY DETECTED\n")
        else
            println("  ✅ Non-negative (up to numerical noise)\n")
        end
    end
end

# ---------------- Execute ----------------

test_basis_state_positivity()
