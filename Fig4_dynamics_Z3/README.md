# Fig2_dynamics_Z3

Self-contained reproduction folder for the dynamical-observable figure of the
long-range \(Z_3\) clock chain.

## Model

\[
H=
-\sum_{i<j}
\frac{J}{|i-j|^\alpha}
\left(
Z_i Z_j^\dagger+
Z_i^\dagger Z_j
\right)
-g\sum_i(X_i+X_i^\dagger).
\]

Parameters:

- \(L=13\)
- open boundary conditions
- \(j_0=7\)
- initial state: one local \(|1\rangle\) excitation at \(j_0\), all other sites \(|0\rangle\)
- \(J=1\)
- \(g/J=0.5\)
- \(\alpha=3.0,1.5,0.5\)
- \(Jt\in[0,5]\)
- 101 output times
- 10,000 pTWA trajectories for each sampling method

## Observables

Central-site survival probability:

\[
P_{\rm exc}(j_0,t)
=
\langle
X_{j_0}^{11}(t)+
X_{j_0}^{22}(t)
\rangle .
\]

Mean displacement:

\[
\bar r(t)
=
\sum_j
|j-j_0|\,
P_{\rm exc}(j,t).
\]

The figure compares:

- ED: black solid curves
- Gaussian pTWA: red dashed curves
- discrete-Wigner pTWA: blue dashed curves

Error bars are standard errors of the pTWA trajectory ensemble.

## Install

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Run

Check both initial-state sampling implementations:

```bash
julia --project=. check_sampling.jl
julia --project=. run_ed.jl
julia --project=. -t auto run_ptwa.jl
julia --project=. check.jl
julia --project=. plot.jl
```

Outputs:

- `Fig2_dynamics_Z3.png`
- `Fig2_dynamics_Z3.pdf`

## pTWA convention

With

\[
(h_j)_{ba}
=
\frac{\partial H_W}{\partial x_j^{ab}},
\]

the evolution equation used here is

\[
\dot x_j
=
i[x_j,h_j].
\]


