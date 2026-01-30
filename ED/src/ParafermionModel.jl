module ParafermionModel

    using ParafermionDynamic
    export Model, build_model, nn_hopping, long_range_hopping

    mutable struct Model
        L::Int                    # number of sites
        n::Int                    # local dimension (n>2)
        sector::Union{Nothing,Int}# sector selection (like nup)
        reflection_parity::Union{Nothing,Int}
        cc_parity::Union{Nothing,Int}
        states::Vector{UInt64}    # basis states
        idxmap::Dict{UInt64,Int}  # state → index
        hop_list::Vector{Tuple{Int,Int,Float64}}        # single-particle hopping J_ij f_
        pair_hop_list::Vector{Tuple{Int,Int,Float64}}   # two-particle hopping g_ij
        mu::Vector{Float64}                             # onsite chemical potential μ_j
        #H = \sum_{i,j} J_{ij} f_i^\dagger f_j 
        #    + \sum_{i,j} G_{ij} (f_i^\dagger)^2 (f_j)^2
        #    + \sum_i \mu_i n_i
    end

    
    function build_model(L::Int; n::Int=3,
                        sector=nothing,
                        reflection_parity=nothing,
                        cc_parity=nothing,
                        hopping=[],
                        pair_hopping=[],
                        mu=zeros(L))

        # -----------------------------
        # Select appropriate basis constructor
        # -----------------------------
        if isnothing(sector) && isnothing(reflection_parity) && isnothing(cc_parity)
            states, idxmap = build_full_basis(L, n)

        elseif !isnothing(sector) && isnothing(reflection_parity) && isnothing(cc_parity)
            states, idxmap = build_sector_basis(L, n, sector)

        elseif !isnothing(sector) && (!isnothing(reflection_parity) || !isnothing(cc_parity))
            # Combined symmetry-aware builder
            states, idxmap = build_combined_symmetry_basis(L, n, sector;
                                                        reflection_parity=reflection_parity,
                                                        cc_parity=cc_parity)
        else
            error("If you specify reflection_parity or cc_parity, please also specify a charge sector.")
        end

        # -----------------------------
        # Build and return model struct
        # -----------------------------
        return Model(L, n, sector, reflection_parity, cc_parity,
                    states, idxmap,
                    [(i,j,J) for (i,j,J) in hopping],
                    [(i,j,G) for (i,j,G) in pair_hopping],
                    Vector{Float64}(mu))
    end


    # ----------------------------
    # Nearest-neighbor hopping
    # ----------------------------
    function nn_hopping(L::Int, J::Float64)
        hop = []
        for i in 0:(L-2)
            push!(hop, (i, i+1, J))  # right
            push!(hop, (i+1, i, J))  # left
        end
        return hop
    end

    # ----------------------------
    # Long-range hopping with 1/r^α decay
    # ----------------------------
    function long_range_hopping(L::Int, J::Float64, α::Float64)
        hop = []
        for i in 0:(L-1)
            for j in 0:(L-1)
                if i != j
                    r = abs(i-j)
                    push!(hop, (i, j, J / r^α))
                end
            end
        end
        return hop
    end


end # module ParafermionModel