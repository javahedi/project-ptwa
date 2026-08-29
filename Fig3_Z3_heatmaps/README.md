# Fig_Z3_heatmaps

Self-contained reproduction of the long-range Z3 clock-chain excitation heatmaps.

Parameters: L=13, j0=7, J=1, g/J=0.5, alpha=3.0,1.5,0.5, open boundaries, t in [0,5].
The initial state has |1> at j0 and |0> elsewhere. pTWA uses discrete Wigner sampling with 10,000 trajectories.

Observable:
P_exc(j,t) = <X_j^{11}(t)+X_j^{22}(t)>.

Run:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. check_sampling.jl
julia --project=. run_ed.jl
julia --project=. -t auto run_ptwa.jl
julia --project=. check.jl
julia --project=. plot.jl
```

The pTWA code uses h_ba=dH_W/dx^ab and dx/dt=i[x,h], matching the corrected manuscript convention.
