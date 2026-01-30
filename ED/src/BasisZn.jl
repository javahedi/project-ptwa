module BasisZn

    using LinearAlgebra
    using Combinatorics

    # =============================================================
    # Z_n Basis Construction with Multiple Symmetries
    # =============================================================

    export build_full_basis, build_sector_basis, digit_at, set_digit
    export build_reflection_sector_basis, build_charge_conjugation_basis, apply_reflection
    export build_combined_symmetry_basis, apply_reflection, apply_charge_conjugation

    """
        build_full_basis(L::Int, n::Int)
    Constructs the full Hilbert space basis for N sites with local dimension n.
    """
    function build_full_basis(L::Int, n::Int)
        dim = n^L
        states = Vector{UInt64}(undef, dim)
        idxmap = Dict{UInt64, Int}()
        for i in 0:dim-1
            states[i+1] = UInt64(i)
            idxmap[UInt64(i)] = i+1
        end
        return states, idxmap
    end

    """
        build_sector_basis(L::Int, n::Int, sector::Int)
    Basis with total charge conservation: Q = ∑ dⱼ mod n
    """
    function build_sector_basis(L::Int, n::Int, sector::Int)
        states, _ = build_full_basis(L, n)
        sector_states = UInt64[]
        idxmap = Dict{UInt64,Int}()
        
        for s in states
            total = sum(digit_at(s, pos, n) for pos in 0:L-1) % n
            if total == sector
                push!(sector_states, s)
                idxmap[s] = length(sector_states)
            end
        end
        
        return sector_states, idxmap
    end

    """
        digit_at(state::UInt64, pos::Int, n::Int)
    Returns the base-n digit at position `pos` (0-based, rightmost = 0)
    """
    function digit_at(state::UInt64, pos::Int, n::Int)
        return (state ÷ n^pos) % n
    end

    """
        set_digit(state::UInt64, pos::Int, val, n::Int)
    Sets the base-n digit at position `pos` to `val` and returns new state.
    Handles both Int and UInt64 values.
    """
    function set_digit(state::UInt64, pos::Int, val, n::Int)
        # Convert val to Int to ensure type consistency
        val_int = Int(val)
        old = digit_at(state, pos, n)
        state -= old * n^pos
        state += val_int * n^pos
        return state
    end

    """
        apply_reflection(state::UInt64, L::Int, n::Int)
    Apply reflection symmetry: site i → site L-1-i
    Returns the reflected state.
    """
    function apply_reflection(state::UInt64, L::Int, n::Int)
        reflected = UInt64(0)
        for i in 0:L-1
            digit = digit_at(state, i, n)
            reflected = set_digit(reflected, L-1-i, digit, n)
        end
        return reflected
    end

    """
        apply_charge_conjugation(state::UInt64, L::Int, n::Int)
    Apply charge conjugation: digit d → (n - d) mod n
    Returns the charge-conjugated state.
    """
    function apply_charge_conjugation(state::UInt64, L::Int, n::Int)
        conjugated = UInt64(0)
        for i in 0:L-1
            digit = digit_at(state, i, n)
            new_digit = (n - digit) % n
            conjugated = set_digit(conjugated, i, new_digit, n)
        end
        return conjugated
    end

    """
        build_reflection_sector_basis(L::Int, n::Int, sector::Int, reflection_parity::Int)
    Basis with reflection symmetry: parity = +1 (even) or -1 (odd) under reflection.
    """
    function build_reflection_sector_basis(L::Int, n::Int, sector::Int, reflection_parity::Int)
        charge_states, charge_idxmap = build_sector_basis(L, n, sector)
        reflection_states = UInt64[]
        idxmap = Dict{UInt64,Int}()
        
        used = Set{UInt64}()
        
        for s in charge_states
            if s in used
                continue
            end
            
            reflected = apply_reflection(s, L, n)
            
            if reflected == s  # State is reflection-symmetric
                if reflection_parity == 1  # Keep symmetric states for +1 sector
                    push!(reflection_states, s)
                    idxmap[s] = length(reflection_states)
                end
                push!(used, s)
            else
                # State and its reflection form a pair
                if reflection_parity == 1  # +1 sector: use symmetric combination
                    # For simplicity, we'll just take the "smaller" state
                    representative = min(s, reflected)
                    push!(reflection_states, representative)
                    idxmap[representative] = length(reflection_states)
                    push!(used, s)
                    push!(used, reflected)
                else  # -1 sector: would need antisymmetric combinations
                    # More complex - need to build proper symmetry-adapted basis
                    # For now, we'll skip -1 sector implementation
                end
            end
        end
        
        return reflection_states, idxmap
    end


    """
        build_combined_symmetry_basis(L, n, sector; reflection_parity=nothing, cc_parity=nothing)
    General construction that combines reflection and charge-conjugation symmetries.
    """

    function build_combined_symmetry_basis(L::Int, n::Int, sector::Int;
                                        reflection_parity::Union{Nothing,Int}=nothing,
                                        cc_parity::Union{Nothing,Int}=nothing)

        charge_states, _ = build_sector_basis(L, n, sector)
        combined_states = UInt64[]
        idxmap = Dict{UInt64,Int}()
        used = Set{UInt64}()

       # Define symmetry operations to apply (allow heterogeneous closures)
        sym_ops = Function[identity]
        if !isnothing(reflection_parity)
            push!(sym_ops, s -> apply_reflection(s, L, n))
        end
        if !isnothing(cc_parity)
            push!(sym_ops, s -> apply_charge_conjugation(s, L, n))
        end
        if !isnothing(reflection_parity) && !isnothing(cc_parity)
            push!(sym_ops, s -> apply_charge_conjugation(apply_reflection(s, L, n), L, n))
        end


        for s in charge_states
            if s in used
                continue
            end

            # Apply all symmetry operations
            partners = UInt64[sym(s) for sym in sym_ops]
            unique_partners = unique(partners)

            # Select representative according to parities (+1: symmetric)
            # For now we only implement +1 sectors; antisymmetric would require linear combos
            if (reflection_parity in (nothing, 1)) && (cc_parity in (nothing, 1))
                representative = minimum(unique_partners)
                push!(combined_states, representative)
                idxmap[representative] = length(combined_states)
                foreach(p -> push!(used, p), unique_partners)
            end
        end

        return combined_states, idxmap
    end

   

end # module