using LinearAlgebra
using Printf
using Statistics


const ω = cis(2π/3)

# Hubbard operators X^{ab} = |a><b| in 3-dim space (a,b in 0,1,2)
function Xab(a::Int, b::Int)
    X = zeros(ComplexF64, 3, 3)
    X[a+1, b+1] = 1.0 + 0im
    return X
end

# Two candidate definitions for f
function f_current_code()
    return Xab(0,1) + Xab(1,2)          # <- what your code uses
end

function f_paper()
    return Xab(0,1) + sqrt(2) * Xab(1,2) # <- what your paper states
end

function check_f(name::String, f::Matrix{ComplexF64})
    fd = f'  # dagger
    I3 = Matrix{ComplexF64}(I, 3, 3)

    nil_f  = norm(f^3)
    nil_fd = norm(fd^3)

    lhs = f*fd - ω*(fd*f)
    rhs = (1 - ω) * I3
    err = norm(lhs - rhs)

    println("=== $name ===")
    @printf("||f^3||        = %.3e\n", nil_f)
    @printf("||f†^3||       = %.3e\n", nil_fd)
    @printf("||ff† - ω f†f - (1-ω)I|| = %.3e\n", err)
    println("matrix lhs (rounded):")
    display(round.(lhs; digits=6))
    println()
end

check_f("Current code: f = X01 + X12", f_current_code())
check_f("Paper: f = X01 + √2 X12", f_paper())
