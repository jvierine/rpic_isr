# Periodic 2D2V ISR model

The separate `rpic_isr_2d` executable retains the physical electron and O+ masses
and evolves two position and two velocity components. This is an unmagnetized,
electrostatic model, not an electromagnetic or magnetized simulation and not a
three-dimensional plasma. Both transverse spatial structure and transverse
particle motion are now present; it is not a collection of independent 1D lines.

## Equations and discretization

The square has side `L=2*pi*m/k_Bragg`, periodic in x and y. Each species has
`cells*cells*ppc` particles. `--cells` means cells **per side**, and `--ppc`
means particles **per two-dimensional cell per species**. Quadratic B-spline
(TSC, nine mesh points) deposition and the same field gather couple particles through
`div(E)=n_i-n_e`. In Fourier space,
`E(kx,ky)=-i*(kx,ky)*rho(kx,ky)/(kx^2+ky^2)`.
The DC field is zero. Each component's own Nyquist derivative is zero; the
other component is retained on that plane. Leapfrog updates both velocities
and both positions. GPU force/deposition/FFT arrays and particles are float32;
diagnostic reductions and spectrum accumulation use float64.

The CPU reference uses RustFFT row/column transforms. CUDA uses batched
two-dimensional cuFFT transforms and custom deposition/interpolation/push
kernels. Independent realizations have disjoint grids and particle arrays.

Positions are independently uniform, and vx and vy are independently Gaussian.
Each component is centered and rescaled to the target initial variance. This
is a Maxwellian start, **not** a Debye-correlated equilibrium loading. Burn-in
must be assessed from temperature, energy, early/late spectrum, and longer-burn
comparisons; the 1D burn-in results do not validate this new model.

Temperature diagnostics average the central variances of the two components:
`T_s=m_s*(var(vx)+var(vy))/(2*k_B)`. Energy includes both kinetic components
and both field components. The nine-column diagnostics retain x momentum as
the line-of-sight momentum. Saved trajectories have shape `[time,32,2]`, with
16 electrons followed by 16 ions, and component order `[x,y]` in m and m/s.
Metadata states dimensionality, particle count, initialization and conventions.
An additional `fast_x_m` record contains every step for the first 2048 steps,
so electron trajectories are not aliased by the slower diagnostic cadence.

Quadratic shapes are a standard way to reduce grid-scale artifacts, but are
not an exact energy-conserving scheme. See
[Abe et al. (1986)](https://doi.org/10.1016/0021-9991(86)90193-2) and
[Huang et al. (2016)](https://www.sciencedirect.com/science/article/pii/S001046551630145X).

## Radar observable and interpretation

The observed electron-density mode is `(m,0)`, with magnitude
`k_Bragg=4*pi*440.2 MHz/c`. It is normalized by `sqrt(Np)` and corrected for
the resolved TSC shape in x, `sinc(pi*m/N)^3`. This is the same density-mode
observable as in 1D, with a quadratic shape and a two-dimensional self-consistent field. It is not
absolute received radar voltage or calibrated radar power. The normalized
Maxwellian longitudinal kinetic spectrum is a useful comparison; agreement
does not make a finite-macroparticle 2D calculation a converged 3D plasma model.
The existing anti-alias FIR, complex downsampling, Hann Welch spectrum and ACF
conventions are unchanged. No fitted amplitude factor is applied.

## Build and run

```sh
cargo test --lib
cargo build --release --bin rpic_isr_2d
target/release/rpic_isr_2d --backend cpu --cells 32 --bragg-mode 1 \
  --ppc 4 --dt .06 --burn-ms 0 --duration-ms .002 --sample-us .005 \
  --window 16 --diagnostic-stride 1 --output results/2d_cpu.h5

cargo build --release --features cuda --bin rpic_isr_2d
bash scripts/run_2d.sh 0 4000
bash scripts/run_2d.sh 1 5000
cargo run --release --bin check_ensemble -- \
  results/2d_production/gpu0.h5 results/2d_production/gpu1.h5
cargo run --release --features plot --bin plot_results -- \
  examples/440mhz_2d results/2d_production/gpu0.h5 results/2d_production/gpu1.h5
```

The production script selects a 128x128 mesh, 8 particles/cell/species (131072
particles/species), `dt*omega_pe=.06`, 10 ms burn-in and 8 ms recording, with
eight independent realizations per card. Side length is 0.681037 m,
`dx/lambda_De=.172401`, and the physical timestep is 10.6355 ns. Launching
both scripts uses the two GPUs independently, not domain decomposition.
The complete raw density series, filtered records and trajectory diagnostics
are in HDF5. Existing filenames are not overwritten.

Execution results and limitations will be recorded below after the runs finish.
