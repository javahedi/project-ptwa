using LinearAlgebra
using Printf
using Statistics
# ================= Kernel (same as your code) =================

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
    return (A + A') ./ 2
end

function precompute_A()
    Ac = Array{ComplexF64,4}(undef, 3,3,3,3)
    for q in 0:2, p in 0:2
        Ac[:,:,q+1,p+1] = Aqp_WH(q,p)
    end
    return Ac
end

# ================= Your symbol definitions ====================

@inline f_symbol(x)    = x[1,2] + x[2,3]
@inline fdag_symbol(x) = x[2,1] + x[3,2]

# ================= Test routine ===============================

function test_parafermion_algebra()
    Ac = precompute_A()

    println("=== Testing parafermion algebra at symbol level ===\n")

    for a in 0:2
        println("State |$a⟩⟨$a|:")

        # density matrix
        ρ = zeros(ComplexF64, 3, 3)
        ρ[a+1,a+1] = 1.0

        # compute average over all discrete phase points
        f_vals    = ComplexF64[]
        fdag_vals = ComplexF64[]

        for q in 0:2, p in 0:2
            A = Ac[:,:,q+1,p+1]
            x = transpose(A)   # same convention as your sampling

            push!(f_vals,    f_symbol(x))
            push!(fdag_vals, fdag_symbol(x))
        end

        # average products over phase space
        ffdag  = mean(f_vals .* fdag_vals)
        fdagf  = mean(fdag_vals .* f_vals)

        lhs = ffdag - ω3 * fdagf
        rhs = 1 - ω3

        println("  <f^3> (symbol level) = ", mean(f_vals.^3))
        println("  <f f† - ω f† f>      = ", lhs)
        println("  expected (1 - ω)     = ", rhs)
        println("  error                = ", lhs - rhs, "\n")
    end
end

# ================= Run ===============================

test_parafermion_algebra()
