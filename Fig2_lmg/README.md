# Fig_LMG

Self-contained reproduction of the fully connected \(Z_n\) clock-model benchmark.

Run:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. run_ed.jl
julia --project=. -t auto run_ptwa.jl
julia --project=. check.jl
julia --project=. plot.jl
```

Parameters: \(J=1\), \(g/J=0.5\), \(t_{\max}=5\).

Top panel: \(N=30\), \(n=3,\dots,7\).

Bottom panel: \(n=5\), \(N=10,20,\dots,80\).

pTWA uses 2000 Gaussian trajectories.

Important: the plotted pTWA quantity is
\[
|m(t)|=\left|\frac1N\sum_j\langle Z_j\rangle\right|,
\]
so the code averages the complex trajectory estimator first and takes the
modulus afterwards.
