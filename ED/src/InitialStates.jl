module InitialStates

    using ..BasisZn
    using LinearAlgebra

    export polarized_state, neel_state, random_sector_state, plusX_state

    """
        polarized_state(N::Int, n::Int, val::Int=0)
    """
    function polarized_state(N::Int, n::Int, val::Int=0)
        s::UInt64 = 0
        for i in 0:N-1
            s = set_digit(s, i, val, n)
        end
        return s
    end

    """
        neel_state(N::Int, n::Int; pattern=[0,1])
    """
    function neel_state(N::Int, n::Int; pattern=[0,1])
        s::UInt64 = 0
        plen = length(pattern)
        for i in 0:N-1
            val = pattern[(i % plen) + 1]
            s = set_digit(s, i, val, n)
        end
        return s
    end

    """
        random_sector_state(N::Int, n::Int, sector::Int)
    """
    function random_sector_state(N::Int, n::Int, sector::Int)
        states, _ = build_sector_basis(N, n, sector)
        idx = rand(1:length(states))
        return states[idx]
    end


    """
    plusX_state(N::Int, n::Int)

    Constructs the product state |+_X>^{⊗ N}, where
    |+_X> = 1/√n ∑_{k=0}^{n-1} |k>. 
    Returns the normalized statevector as a complex vector.
    """
    function plusX_state(N::Int, n::Int)
        dim = n^N
        ψ = fill(ComplexF64(1/sqrt(dim)), dim)
        ψ ./= norm(ψ)  # ensure numerical normalization
        return ψ
    end

end # module InitialStates