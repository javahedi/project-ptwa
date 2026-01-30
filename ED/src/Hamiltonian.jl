module Hamiltonian

    using LinearAlgebra
    using ..BasisZn
    using ..ParafermionModel

    export apply_H!

    """
        apply_H!(out, ψ, model)
    Correct implementation with proper phase factors for adjacent sites

        #H = sum_{i,j} J_{ij} f_i^dagger f_j 
        #    + sum_{i,j} G_{ij} (f_i^dagger)^2 (f_j)^2
        #    + sum_i mu_i n_i
    """
    function apply_H!(out::Vector{ComplexF64}, ψ::Vector{ComplexF64}, model::ParafermionModel.Model)
        L, n = model.L, model.n
        N = length(ψ)
        fill!(out, 0.0 + 0.0im)
        
        ω = exp(2π*im/n)

        for idx in 1:N
            amp = ψ[idx]
            if iszero(amp)
                continue
            end

            state = model.states[idx]

            # Onsite ter μ
            diag = 0.0
            for i in 0:L-1
                d = digit_at(state, i, n)
                diag += model.mu[i+1] * d
            end
            out[idx] += diag * amp


            # Single-particle hopping
            for (i,j,Jx) in model.hop_list
                di = Int(digit_at(state, i, n))
                dj = Int(digit_at(state, j, n))
                
                if dj > 0 && di < (n-1)
                    phase = 1.0 + 0.0im
                    
                    # String operator phase depends on sites between i and j
                    if i < j
                        # Rightward hopping: f_i† f_j
                        # String operator: ω^{∑_{k=i+1}^{j-1} n_k}
                        for k in (i+1):(j-1)
                            dk = digit_at(state, k, n)
                            phase *= ω^dk
                        end
                    elseif i > j
                        # Leftward hopping: f_i† f_j  
                        # For parafermions: f_i† f_j = ω^{-∑_{k=j+1}^{i-1} n_k} f_j f_i†
                        for k in (j+1):(i-1)
                            dk = digit_at(state, k, n)
                            phase *= conj(ω)^dk
                        end
                    end
                    # i == j gives diagonal term, handled by onsite
                    
                    new_state = state
                    new_state = set_digit(new_state, i, di+1, n)  # Create at i
                    new_state = set_digit(new_state, j, dj-1, n)  # Annihilate at j
                    
                    if haskey(model.idxmap, new_state)
                        new_idx = model.idxmap[new_state]
                        out[new_idx] += Jx * phase * amp
                    end
                end
            end

            # Two-particle hopping (similar fixes needed here)
            for (i,j,G) in model.pair_hop_list
                di = Int(digit_at(state, i, n))
                dj = Int(digit_at(state, j, n))
                
                if dj >= 2 && di <= (n-2)
                    phase = 1.0 + 0.0im
                    if i < j
                        for k in (i+1):(j-1)
                            dk = digit_at(state, k, n)
                            phase *= ω^(2*dk)  # Squared for pair hopping
                        end
                    elseif i > j
                        for k in (j+1):(i-1)
                            dk = digit_at(state, k, n)
                            phase *= conj(ω)^(2*dk)
                        end
                    end
                    
                    new_state = state
                    new_state = set_digit(new_state, i, di+2, n)
                    new_state = set_digit(new_state, j, dj-2, n)
                    
                    if haskey(model.idxmap, new_state)
                        new_idx = model.idxmap[new_state]
                        out[new_idx] += G * phase * amp
                    end
                end
            end
        end

        return out
    end

end # module Hamiltonian