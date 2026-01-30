module Observables

    using LinearAlgebra
    using ..BasisZn
    using ..ParafermionModel

    export local_occupation, local_X, local_Z, Xcorr_1, Zcorr_1, Xcorr_2, quantum_variance_Z, quantum_variance_X
    export two_site_corr, two_site_average, structure_factor_qt, autocorrelation

    """
        local_occupation(ψ, model)
        
         ⟨nj⟩=∑​_s |ψ_s|^2 n_j(s)
         
         where  n_j(s) is the “digit” (0, 1, 2 for Z₃) at site j in basis state |s⟩
.
    """
    function local_occupation(ψ::AbstractVector{ComplexF64}, model::ParafermionModel.Model)
        N = model.L
        occ = zeros(Float64, N)
        for (idx, state) in enumerate(model.states)
            amp2 = abs2(ψ[idx])
            for i in 0:N-1
                occ[i+1] += digit_at(state, i, model.n) * amp2
            end
        end
        return occ
    end

   
    """ local_Z(ψ, model) 
    Compute ⟨Z_j⟩ = ⟨ψ|Z_j|ψ⟩ = Σ_s |ψ_s|² ω^{n_j(s)}. 
    """ 
    function local_Z(ψ::AbstractVector{ComplexF64}, model::ParafermionModel.Model) 
        N, n = model.L, model.n 
        ω = exp(2π * im / n) 
        Z_exp = zeros(ComplexF64, N) 
        for (idx, state) in enumerate(model.states) 
            amp2 = abs2(ψ[idx]) 
            for j in 0:N-1 
                d = digit_at(state, j, n) 
                Z_exp[j+1] += amp2 * ω^d 
            end 
        end 
        return Z_exp 
    end

    """
        local_X(ψ, model)
    Compute ⟨X_j⟩ = ⟨ψ|X_j|ψ⟩ for all sites j.
    """
    function local_X(ψ::AbstractVector{ComplexF64}, model::ParafermionModel.Model)
        N, n = model.L, model.n
        X_exp = zeros(ComplexF64, N)

        for j in 0:N-1
            val = 0.0 + 0im
            for (idx, state) in enumerate(model.states)
                d = digit_at(state, j, n)
                new_state = set_digit(state, j, mod(d + 1, n), n)
                if haskey(model.idxmap, new_state)
                    new_idx = model.idxmap[new_state]
                    val += conj(ψ[new_idx]) * ψ[idx]
                end
            end
            X_exp[j+1] = val
        end
        return X_exp
    end


   """
    Zcorr_1(ψ, model, i, j)
    Compute ⟨Z_i† Z_j⟩ = ⟨ψ| Z_i† Z_j |ψ⟩ for sites i, j.
    """
    function Zcorr_1(ψ::AbstractVector{ComplexF64}, model::ParafermionModel.Model, i::Int, j::Int)
        n = model.n
        ω = exp(2π*im/n)
        val = 0.0 + 0im

        for (idx, state) in enumerate(model.states)
            di = digit_at(state, i-1, n)
            dj = digit_at(state, j-1, n)
            val += conj(ψ[idx]) * ψ[idx] * ω^di * ω^dj
        end

        return val
    end



    """
        Xcorr_1(ψ, model, i, j)
    Compute ⟨X_i† X_j⟩ = ⟨ψ| X_i† X_j |ψ⟩.
    """
    function Xcorr_1(ψ::AbstractVector{ComplexF64}, model::ParafermionModel.Model, i::Int, j::Int)
        n = model.n
        val = 0.0 + 0im

        for (idx, state) in enumerate(model.states)
            di = digit_at(state, i-1, n)
            dj = digit_at(state, j-1, n)
            # Apply X_i† (−1 shift) and X_j (+1 shift)
            new_state = set_digit(state, i-1, mod(di - 1, n), n)
            new_state = set_digit(new_state, j-1, mod(dj + 1, n), n)
            if haskey(model.idxmap, new_state)
                new_idx = model.idxmap[new_state]
                val += conj(ψ[new_idx]) * ψ[idx]
            end
        end
        return val
    end





    """
        Xcorr_2(ψ, model, i, j)
    Compute ⟨X_i†² X_j²⟩ = ⟨ψ| X_i†² X_j² |ψ⟩.
    """
    function Xcorr_2(ψ::AbstractVector{ComplexF64}, model::ParafermionModel.Model, i::Int, j::Int)
        n = model.n
        val = 0.0 + 0im

        for (idx, state) in enumerate(model.states)
            di = digit_at(state, i-1, n)
            dj = digit_at(state, j-1, n)
            # Apply X_i†² (−2 shift) and X_j² (+2 shift)
            new_state = set_digit(state, i-1, mod(di - 2, n), n)
            new_state = set_digit(new_state, j-1, mod(dj + 2, n), n)
            if haskey(model.idxmap, new_state)
                new_idx = model.idxmap[new_state]
                val += conj(ψ[new_idx]) * ψ[idx]
            end
        end
        return val
    end

    
   

    """
        two_site_corr(ψ, model, i, j)
    """
    function two_site_corr(ψ::AbstractVector{ComplexF64}, model::ParafermionModel.Model, i::Int, j::Int)
        corr = 0.0
        for (idx, state) in enumerate(model.states)
            amp2 = abs2(ψ[idx])
            corr += digit_at(state, i-1, model.n) * digit_at(state, j-1, model.n) * amp2
        end
        return corr
    end

    """
        two_site_average(ψ, model)
    """
    function two_site_average(ψ::AbstractVector{ComplexF64}, model::ParafermionModel.Model)
        N = model.L
        sum_corr = 0.0
        count = 0
        for i in 1:N-1
            for j in i+1:N
                sum_corr += two_site_corr(ψ, model, i, j)
                count += 1
            end
        end
        return sum_corr / count
    end

    """
        structure_factor_qt(ψ0, ψt, model, q)
    """
    function structure_factor_qt(ψ0::AbstractVector{ComplexF64}, ψt::AbstractVector{ComplexF64},
                                model::ParafermionModel.Model, q::Float64)
        N = model.L
        S_qt = 0.0 + 0im
        for i in 0:N-1
            n_i_t = [digit_at(state, i, model.n) * ψt[idx] for (idx, state) in enumerate(model.states)]
            for j in 0:N-1
                phase = exp(1im * q * (i-j))
                n_j_0 = [digit_at(state, j, model.n) * ψ0[idx] for (idx, state) in enumerate(model.states)]
                S_qt += phase * dot(conj(ψt), n_i_t .* n_j_0)
            end
        end
        return S_qt / N
    end

    """
        autocorrelation(ψ0, ψt, model)
    """
    function autocorrelation(ψ0::AbstractVector{ComplexF64}, ψt::AbstractVector{ComplexF64}, model::ParafermionModel.Model)
        N = model.L
        ac = 0.0
        for i in 0:N-1
            ac += two_site_corr(ψt, model, i+1, i+1)
        end
        return ac / N
    end


    function quantum_variance_Z(ψ::Vector{ComplexF64}, model::ParafermionModel.Model)
        L = model.L
        Z_exp = local_Z(ψ, model)  # ⟨Z_j⟩ for all sites

        ΔZ2 = 0.0
        for i in 1:L
            for j in 1:L
                ΔZ2 += Zcorr_1(ψ, model, i, j) - conj(Z_exp[i]) * Z_exp[j]
            end
        end
        ΔZ2 /= L^2
        Z_ave = mean(Z_exp)
        return real(Z_ave), real(ΔZ2)
    end



    function quantum_variance_X(ψ::Vector{ComplexF64}, model::ParafermionModel.Model)
        L = model.L
        X_exp = local_X(ψ, model)  # ⟨X_j⟩ for all sites

        ΔX2 = 0.0
        for i in 1:L
            for j in 1:L
                ΔX2 += Xcorr_1(ψ, model, i, j) - conj(X_exp[i]) * X_exp[j]
            end
        end
        ΔX2 /= L^2
        X_ave = mean(X_exp)
        return real(X_ave), real(ΔX2)
    end



end # module Observables