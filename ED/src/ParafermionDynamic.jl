module ParafermionDynamic

    # Export main functionality from submodules
    export ParafermionModel, BasisZn, Hamiltonian, InitialStates, Krylov, Observables

    # Include submodules in dependency order
    include("BasisZn.jl")
    include("ParafermionModel.jl")
    include("Hamiltonian.jl")
    include("InitialStates.jl")
    include("Observables.jl")
    include("Krylov.jl")

    # Re-export specific functions from submodules
    using .BasisZn: build_full_basis, build_sector_basis, build_combined_symmetry_basis, digit_at, set_digit, apply_reflection
    using .ParafermionModel: Model, build_model, nn_hopping, long_range_hopping
    using .Hamiltonian: apply_H!
    using .InitialStates: polarized_state, neel_state, random_sector_state, plusX_state
    using .Krylov: krylov_time_evolve
    using .Observables: local_occupation, local_X, local_Z, Xcorr_1, Zcorr_1, Xcorr_2, quantum_variance_Z, quantum_variance_X,
                         two_site_corr, two_site_average, structure_factor_qt, autocorrelation

    export build_full_basis, build_sector_basis, build_combined_symmetry_basis, digit_at, set_digit, apply_reflection
    export Model, build_model, nn_hopping, long_range_hopping
    export apply_H!
    export polarized_state, neel_state, random_sector_state, plusX_state
    export krylov_time_evolve

    export local_occupation, local_X, local_Z, Xcorr_1, Zcorr_1, Xcorr_2, quantum_variance_Z, quantum_variance_X
    export two_site_corr, two_site_average, structure_factor_qt, autocorrelation

end # module