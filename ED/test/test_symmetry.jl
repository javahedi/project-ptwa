using ParafermionDynamic

function test_enhanced_symmetries()
    L = 15
    n = 3
    sector = 1
    
    println("=== Testing Enhanced Symmetries for L=$L, n=$n, sector=$sector ===")
    
    # Test each symmetry level
    charge_states, _ = build_sector_basis(L, n, sector)
    println("Charge sector only: $(length(charge_states)) states")
    
    # Test reflection symmetry
    reflection_states, _ = build_reflection_sector_basis(L, n, sector, 1)
    println("+ Reflection symmetry: $(length(reflection_states)) states ($(round(100*length(reflection_states)/length(charge_states), digits=1))%)")
    
    # Test charge conjugation
    cc_states, _ = build_charge_conjugation_basis(L, n, sector, 1)
    println("+ Charge conjugation: $(length(cc_states)) states ($(round(100*length(cc_states)/length(charge_states), digits=1))%)")
    
    # Test combined symmetries
    combined_states, _ = build_combined_symmetry_basis(L, n, sector, 1, 1)
    println("+ All symmetries: $(length(combined_states)) states ($(round(100*length(combined_states)/length(charge_states), digits=1))%)")
    
    full_dim = n^L
    println("\nOverall reduction from full basis:")
    println("Full basis: $full_dim states")
    println("Full → Charge sector: $(round(full_dim/length(charge_states), digits=1))× reduction")
    println("Full → All symmetries: $(round(full_dim/length(combined_states), digits=1))× reduction")
    
    # Show some example states
    println("\nExample states in combined symmetry basis:")
    for (i, state) in enumerate(combined_states[1:min(5, length(combined_states))])
        digits = [BasisZn.digit_at(state, pos, n) for pos in 0:L-1]
        println("  State $i: |$(join(digits, ""))⟩")
    end
end



using ParafermionDynamic

function debug_charge_conjugation()
    L = 6
    n = 3
    sector = 1
    
    println("=== Debugging Charge Conjugation ===")
    
    charge_states, _ = build_sector_basis(L, n, sector)
    
    println("Testing charge conjugation on sector $sector states:")
    println("Total states: $(length(charge_states))")
    
    symmetric_count = 0
    asymmetric_pairs = 0
    
    for (i, state) in enumerate(charge_states[1:10])  # Test first 10
        conjugated = apply_charge_conjugation(state, L, n)
        
        digits = [digit_at(state, pos, n) for pos in 0:L-1]
        conj_digits = [digit_at(conjugated, pos, n) for pos in 0:L-1]
        
        if state == conjugated
            symmetric_count += 1
            println("State $i: |$(join(digits, ""))⟩ → SYMMETRIC")
        else
            asymmetric_pairs += 1
            println("State $i: |$(join(digits, ""))⟩ → |$(join(conj_digits, ""))⟩ (different)")
        end
    end
    
    println("\nSummary:")
    println("Symmetric states: $symmetric_count")
    println("Asymmetric pairs: $asymmetric_pairs")
    
    # Check if charge conjugation changes the sector
    println("\nChecking sector preservation:")
    test_state = charge_states[1]
    conjugated = apply_charge_conjugation(test_state, L, n)
    
    original_sector = sum(digit_at(test_state, pos, n) for pos in 0:L-1) % n
    conjugated_sector = sum(digit_at(conjugated, pos, n) for pos in 0:L-1) % n
    
    println("Original state sector: $original_sector")
    println("Conjugated state sector: $conjugated_sector")
    println("Sector preserved: $(original_sector == conjugated_sector)")
end




test_enhanced_symmetries()

debug_charge_conjugation()