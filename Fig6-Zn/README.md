# Fig4_Zn

Self-contained reproduction folder for the \(\mathbb Z_n\) single-hopping domain-wall benchmark.

## Important consistency note

The supplied plotting code and the available pTWA source use **\(L=24\)** (their filenames are `..._L24...`). Therefore this folder uses \(L=24\). The manuscript caption currently says \(L=48\); that should be changed to \(L=24\) unless a separate \(L=48\) production data set exists.

The supplied pTWA source uses the domain wall
\[
|\psi_0\rangle=|0,\ldots,0,n-1,\ldots,n-1\rangle,
\]
with \(P_0=|0\rangle\langle0|\) and
\[
\mathcal I(t)=\frac{2}{L}\left[\sum_{j\le L/2}P_0(j,t)-\sum_{j>L/2}P_0(j,t)\right],
\]
so \(\mathcal I(0)=1\).

The single-hopping Hamiltonian is
\[
H=-J\sum_j[B_j^\dagger U_j B_{j+1}+\mathrm{H.c.}],
\]
where \(B=\sum_{a=1}^{n-1}|a-1\rangle\langle a|\) has unit matrix elements.

## Structure-factor correction

Both TEBD and pTWA now calculate the same observable,
\[
S_0(q,t)=\frac1L\left|\sum_j e^{iqj}\langle P_0(j,t)\rangle\right|^2.
\]
The older pTWA source instead Fourier transformed each trajectory and averaged the power; that is a different quantity. This folder first averages the occupation profile and only then Fourier transforms it.

## Parameters

- \(L=24\)
- \(n=3,\ldots,7\)
- \(J=1\), OBC
- \(dt=0.05/J\), \(Jt\in[0,10]\)
- pTWA: Gaussian sampling, \(N_{\rm traj}=2000\)
- TEBD: first-order even/odd Trotter, \(\chi_{\max}=80\)

With \((h_j)_{ba}=\partial H_W/\partial x_j^{ab}\), pTWA uses \(\dot x_j=i[x_j,h_j]\).

## Run

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. check_sampling.jl
julia --project=. -t auto run_ptwa.jl
julia --project=. run_tebd.jl
julia --project=. check.jl
julia --project=. plot.jl
```
