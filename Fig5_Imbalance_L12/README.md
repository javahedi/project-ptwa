# Fig3_Imbalance_L12

Self-contained reproduction folder for the two imbalance panels of the disordered Z3 Fock-parafermion chain.

Parameters: L=12, J=1, g=0.3, W={0.5,1,2,3,4,5,6,7,8}, 100 disorder realizations, 100 pTWA trajectories per disorder realization, t in [0,100], and inset average over [2,20]. The initial domain wall follows the supplied source codes: left half |1>, right half |0>, so I(0)=1.

The corrected pTWA implementation uses the nilpotent Fock-parafermion lowering operator B=|0><1|+|1><2| with no bosonic sqrt(2) coefficient, the nearest-neighbor strings B† U B and (B†)^2 U^2 B^2, and the convention dx/dt=i[x,h]. Only Hermiticity roundoff is cleaned up; trajectories are not trace-renormalized during RK4.

`run_ed_benchmark.jl` is built directly from the supplied bond-gate/MPS benchmark source. In the manuscript/plot it is labeled **ED**, exactly as requested. Internally, the supplied source evolves with a TEBD/MPS bond-gate algorithm (chi=100, dt=0.1); this README keeps that computational fact explicit. Its time indexing is corrected so the initial state is stored at t=0.

Run:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. check_sampling.jl
julia --project=. -t auto run_ptwa.jl
julia --project=. run_ed_benchmark.jl
julia --project=. check.jl
julia --project=. plot.jl
```

