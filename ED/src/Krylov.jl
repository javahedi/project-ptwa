module Krylov

    using LinearAlgebra
    using ..ParafermionModel
    using ..Hamiltonian

    export krylov_time_evolve

    """
        krylov_time_evolve(ψ0, t, applyH!, model; m=30)
    """
    function krylov_time_evolve(ψ0::AbstractVector{T}, dt::Float64,
                applyH!, model::ParafermionModel.Model; kry_m::Int=30) where T<:Number

            n = length(ψ0)
            V = Vector{Vector{T}}(undef, kry_m)
            α = zeros(ComplexF64, kry_m)
            β = zeros(ComplexF64, kry_m-1)
            w = zeros(T, n)

            norm0 = norm(ψ0)
            if norm0 == 0
                return copy(ψ0)
            end
            V[1] = copy(ψ0) / norm0

            m_eff = kry_m
            for j in 1:kry_m
                applyH!(w, V[j], model)

                α[j] = dot(V[j], w)
                w .-= α[j] .* V[j]
                if j > 1
                    w .-= β[j-1] .* V[j-1]
                end
                if j < kry_m
                    β[j] = norm(w)
                    if abs(β[j]) < 1e-14
                        m_eff = j
                        α = α[1:m_eff]
                        β = β[1:(m_eff-1)]
                        V = V[1:m_eff]
                        break
                    end
                    V[j+1] = copy(w / β[j])
                end
            end

            # Build tridiagonal and exponentiate
            TR = Tridiagonal(β[1:(m_eff-1)], α[1:m_eff], β[1:(m_eff-1)])
            eig = eigen(Matrix(TR))

            D = eig.values
            Q = eig.vectors
            U_T = Q * Diagonal(exp.(-1im .* D .* dt)) * Q'
            e1 = zeros(ComplexF64, m_eff); e1[1] = norm0
            y = U_T * e1

            # Reconstruct complex ψt
            ψt = zeros(ComplexF64, n)
            for k in 1:m_eff
                ψt .+= y[k] .* V[k]
            end

            ψt ./= norm(ψt)
            return ψt
    end

end # module Krylov