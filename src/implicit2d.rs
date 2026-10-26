//! Experimental 2D discrete-work implicit PIC, not a literal port of the 1D
//! Chen--Chacon--Barnes mover. Compatible endpoint charge, FFT Poisson, and
//! orbit-averaged gradient of the TSC-interpolated midpoint potential give
//! delta(K)+delta(0.5*rho*phi)=0 at nonlinear convergence in exact arithmetic.
use crate::pic2d::Particles2;
use crate::*;
use rustfft::{num_complex::Complex64, Fft, FftPlanner};
use std::sync::Arc;

#[derive(clap::Parser, Debug, Clone, Serialize)]
pub struct Options {
    #[arg(long, value_enum, default_value = "cpu")]
    pub backend: Backend,
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    #[arg(long, default_value_t = 32)]
    pub cells: usize,
    #[arg(long, default_value_t = 8)]
    pub ppc: usize,
    #[arg(long, default_value_t = 1)]
    pub bragg_mode: usize,
    #[arg(long, default_value_t = 0.12)]
    pub dt: f64,
    #[arg(long, default_value_t = 2000)]
    pub steps: usize,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value_t = 2e-7)]
    pub rtol: f64,
    #[arg(long, default_value_t = 2e-9)]
    pub atol: f64,
    #[arg(long, default_value_t = 40)]
    pub max_iterations: usize,
    #[arg(long, default_value_t = 12)]
    pub particle_iterations: usize,
    #[arg(long, default_value_t = 1.)]
    pub relaxation: f64,
    #[arg(long, default_value_t = 100)]
    pub diagnostic_stride: usize,
    #[arg(long, default_value = "results/implicit2d.h5")]
    pub output: String,
}
impl Options {
    pub fn physical(&self) -> Result<(Config, Scales)> {
        anyhow::ensure!(
            self.steps > 0
                && self.steps <= i32::MAX as usize
                && self.max_iterations > 0
                && self.particle_iterations > 0
                && self.diagnostic_stride > 0
                && self.rtol.is_finite()
                && self.rtol > 0.
                && self.atol.is_finite()
                && self.atol > 0.
                && self.relaxation > 0.
                && self.relaxation <= 1.,
            "Invalid implicit controls"
        );
        anyhow::ensure!(
            self.cells <= 512
                && self
                    .cells
                    .checked_mul(self.cells)
                    .and_then(|n| n.checked_mul(self.ppc))
                    .is_some_and(|n| n <= i32::MAX as usize / 2)
                && self.particle_iterations <= 1024
                && self.max_iterations <= 1024,
            "Implicit allocation or iteration limit exceeded"
        );
        let c = Config {
            backend: self.backend,
            device: self.device,
            cells: self.cells,
            ppc: self.ppc,
            bragg_mode: self.bragg_mode,
            dt: self.dt,
            seed: self.seed,
            ..Config::parse_from(["physical"])
        };
        let s = Scales::new(&c)?;
        Ok((c, s))
    }
}
#[derive(Clone)]
pub struct State {
    pub x: Vec<[f64; 2]>,
    pub v: Vec<[f64; 2]>,
}
impl State {
    pub fn new(c: &Config, s: &Scales) -> Self {
        let p = Particles2::new(c, s, c.seed);
        Self {
            x: p.xe
                .into_iter()
                .chain(p.xi)
                .map(|p| [p[0] as f64, p[1] as f64])
                .collect(),
            v: p.ve
                .into_iter()
                .chain(p.vi)
                .map(|v| [v[0] as f64, v[1] as f64])
                .collect(),
        }
    }
}
pub fn shape(x: [f64; 2], n: usize, length: f64) -> [(usize, f64, [f64; 2]); 9] {
    let u = [x[0] / length * n as f64, x[1] / length * n as f64];
    let cell = [(u[0] + 0.5).floor() as isize, (u[1] + 0.5).floor() as isize];
    let mut w = [[0.; 3]; 2];
    let mut dw = [[0.; 3]; 2];
    for d in 0..2 {
        let a = u[d] - cell[d] as f64;
        w[d] = [
            0.5 * (0.5 - a).powi(2),
            0.75 - a * a,
            0.5 * (0.5 + a).powi(2),
        ];
        dw[d] = [
            (a - 0.5) * n as f64 / length,
            -2. * a * n as f64 / length,
            (a + 0.5) * n as f64 / length,
        ];
    }
    let mut out = [(0, 0., [0.; 2]); 9];
    for y in 0..3 {
        for x in 0..3 {
            let j = (cell[1] + y as isize - 1).rem_euclid(n as isize) as usize * n
                + (cell[0] + x as isize - 1).rem_euclid(n as isize) as usize;
            out[y * 3 + x] = (
                j,
                w[0][x] * w[1][y],
                [dw[0][x] * w[1][y], w[0][x] * dw[1][y]],
            );
        }
    }
    out
}
pub fn interpolate(phi: &[f64], x: [f64; 2], n: usize, length: f64) -> f64 {
    shape(x, n, length)
        .iter()
        .map(|(j, w, _)| phi[*j] * w)
        .sum()
}
pub fn gradient(phi: &[f64], x: [f64; 2], n: usize, length: f64) -> [f64; 2] {
    let mut e = [0.; 2];
    for (j, _, dw) in shape(x, n, length) {
        for d in 0..2 {
            e[d] -= phi[j] * dw[d];
        }
    }
    e
}
/// Two-point Gauss integration is exact on each TSC polynomial piece:
/// the gradient along a straight 2D orbit has degree at most three.
pub fn orbit_field(
    phi: &[f64],
    x: [f64; 2],
    delta: [f64; 2],
    n: usize,
    length: f64,
) -> Result<[f64; 2]> {
    let mut cuts = vec![0., 1.];
    for d in 0..2 {
        let a = x[d] * n as f64 / length;
        let b = (x[d] + delta[d]) * n as f64 / length;
        if (b - a).abs() > 0. {
            let lo = (a.min(b) - 0.5).floor() as isize;
            let hi = (a.max(b) - 0.5).ceil() as isize;
            anyhow::ensure!(
                hi - lo < 64,
                "Orbit crosses too many shape pieces; reduce timestep"
            );
            for j in lo..=hi {
                let t = (j as f64 + 0.5 - a) / (b - a);
                if t > 0. && t < 1. {
                    cuts.push(t);
                }
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-14);
    let mut out = [0.; 2];
    for w in cuts.windows(2) {
        let h = 0.5 * (w[1] - w[0]);
        let center = 0.5 * (w[1] + w[0]);
        for t in [center - h / 3f64.sqrt(), center + h / 3f64.sqrt()] {
            let e = gradient(phi, [x[0] + t * delta[0], x[1] + t * delta[1]], n, length);
            for d in 0..2 {
                out[d] += h * e[d];
            }
        }
    }
    Ok(out)
}
pub struct Potential {
    n: usize,
    length: f64,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    column: Vec<Complex64>,
    scratch: Vec<Complex64>,
    buf: Vec<Complex64>,
    pub rho: Vec<f64>,
    pub phi: Vec<f64>,
    pub observable: Complex32,
}
impl Potential {
    pub fn new(n: usize, length: f64) -> Self {
        let mut plan = FftPlanner::new();
        let forward = plan.plan_fft_forward(n);
        let inverse = plan.plan_fft_inverse(n);
        let scratch = vec![
            Complex64::default();
            forward
                .get_inplace_scratch_len()
                .max(inverse.get_inplace_scratch_len())
        ];
        Self {
            n,
            length,
            forward,
            inverse,
            column: vec![Complex64::default(); n],
            scratch,
            buf: vec![Complex64::default(); n * n],
            rho: vec![0.; n * n],
            phi: vec![0.; n * n],
            observable: Complex32::default(),
        }
    }
    fn transform(&mut self, inverse: bool) {
        let fft = if inverse {
            &self.inverse
        } else {
            &self.forward
        };
        for row in self.buf.chunks_exact_mut(self.n) {
            fft.process_with_scratch(row, &mut self.scratch);
        }
        for x in 0..self.n {
            for y in 0..self.n {
                self.column[y] = self.buf[y * self.n + x];
            }
            fft.process_with_scratch(&mut self.column, &mut self.scratch);
            for y in 0..self.n {
                self.buf[y * self.n + x] = self.column[y];
            }
        }
    }
    pub fn solve(&mut self, p: &State, mode: usize) {
        self.rho.fill(0.);
        let np = p.x.len() / 2;
        let norm = (self.n * self.n) as f64 / np as f64;
        let mut obs = Complex64::default();
        let phases: Vec<_> = (0..self.n)
            .map(|j| {
                Complex64::from_polar(
                    1.,
                    -2. * std::f64::consts::PI * mode as f64 * j as f64 / self.n as f64,
                )
            })
            .collect();
        for (i, &x) in p.x.iter().enumerate() {
            let q = if i < np { -1. } else { 1. };
            for (j, w, _) in shape(x, self.n, self.length) {
                self.rho[j] += q * norm * w;
                if i < np {
                    obs += phases[j % self.n] * w;
                }
            }
        }
        let kd = std::f64::consts::PI * mode as f64 / self.n as f64;
        obs /= (np as f64).sqrt() * (kd.sin() / kd).powi(3);
        self.observable = Complex32::new(obs.re as f32, obs.im as f32);
        for (j, &r) in self.rho.iter().enumerate() {
            self.buf[j] = Complex64::new(r, 0.);
        }
        self.transform(false);
        for y in 0..self.n {
            for x in 0..self.n {
                let k = |j: usize| {
                    2. * std::f64::consts::PI
                        * (if j <= self.n / 2 {
                            j as f64
                        } else {
                            j as f64 - self.n as f64
                        })
                        / self.length
                };
                let k2 = k(x).powi(2) + k(y).powi(2);
                self.buf[y * self.n + x] = if k2 == 0. {
                    Complex64::default()
                } else {
                    self.buf[y * self.n + x] / k2
                };
            }
        }
        self.transform(true);
        for (j, v) in self.buf.iter().enumerate() {
            self.phi[j] = v.re / (self.n * self.n) as f64;
        }
    }
    pub fn energy(&self) -> f64 {
        0.5 * self
            .rho
            .iter()
            .zip(&self.phi)
            .map(|(r, p)| r * p)
            .sum::<f64>()
            / self.rho.len() as f64
    }
}
#[derive(Default)]
pub struct ResultData {
    pub diagnostics: Vec<f64>,
    pub trace: Vec<f64>,
    pub density: Vec<Complex32>,
    pub iterations: Vec<u32>,
    pub residuals: Vec<f64>,
    pub elapsed: f64,
    pub final_rho: Vec<f64>,
    pub final_phi: Vec<f64>,
}
pub const COLUMNS: [&str; 12] = [
    "time_s",
    "total_energy_kBTe",
    "electron_kinetic_kBTe",
    "ion_kinetic_kBTe",
    "potential_energy_kBTe",
    "Te_K",
    "Ti_K",
    "momentum_x_me_vte",
    "momentum_y_me_vte",
    "outer_iterations",
    "residual_rms",
    "mean_charge_ne",
];
#[allow(clippy::too_many_arguments)] // Physics and convergence state are explicit.
pub fn diagnostics(
    out: &mut ResultData,
    p: &State,
    u: f64,
    c: &Config,
    s: &Scales,
    step: usize,
    iterations: usize,
    residual: f64,
    charge: f64,
) {
    let np = p.v.len() / 2;
    let mut k = [0.; 2];
    let mut temp = [0.; 2];
    let mut mom = [0.; 2];
    for species in 0..2 {
        let mass = if species == 0 { 1. } else { s.mass_ratio };
        let mut mean = [0.; 2];
        let mut vv = 0.;
        for v in &p.v[species * np..(species + 1) * np] {
            for d in 0..2 {
                mean[d] += v[d] / np as f64;
                vv += v[d] * v[d] / np as f64;
            }
        }
        k[species] = 0.5 * mass * vv;
        temp[species] = 0.5 * mass * c.te * (vv - mean[0] * mean[0] - mean[1] * mean[1]);
        for d in 0..2 {
            mom[d] += mass * mean[d];
        }
        for &x in p.x[species * np..species * np + 16].iter() {
            out.trace.extend(x.map(|v| v * s.lambda_de));
        }
    }
    out.diagnostics.extend([
        step as f64 * s.dt_s,
        k[0] + k[1] + u,
        k[0],
        k[1],
        u,
        temp[0],
        temp[1],
        mom[0],
        mom[1],
        iterations as f64,
        residual,
        charge,
    ]);
}
pub fn advance(
    old: &State,
    mid: &[f64],
    c: &Config,
    s: &Scales,
    particle_iterations: usize,
) -> Result<State> {
    let mut next = old.clone();
    let np = old.x.len() / 2;
    for i in 0..old.x.len() {
        let qm = if i < np { -1. } else { 1. / s.mass_ratio };
        let mut v = old.v[i];
        let mut converged = false;
        for _ in 0..particle_iterations {
            let delta = [
                0.5 * c.dt * (old.v[i][0] + v[0]),
                0.5 * c.dt * (old.v[i][1] + v[1]),
            ];
            let e = orbit_field(mid, old.x[i], delta, c.cells, s.length)?;
            let vn = [
                old.v[i][0] + c.dt * qm * e[0],
                old.v[i][1] + c.dt * qm * e[1],
            ];
            let err = (vn[0] - v[0]).abs().max((vn[1] - v[1]).abs());
            v = vn;
            if err < 1e-12 * (1. + v[0].abs().max(v[1].abs())) {
                converged = true;
                break;
            }
        }
        anyhow::ensure!(converged, "Particle midpoint iteration failed");
        next.v[i] = v;
        for (d, &velocity) in v.iter().enumerate() {
            next.x[i][d] =
                (old.x[i][d] + 0.5 * c.dt * (old.v[i][d] + velocity)).rem_euclid(s.length);
        }
    }
    Ok(next)
}
pub fn simulate(o: &Options, c: &Config, s: &Scales) -> Result<ResultData> {
    let start = std::time::Instant::now();
    let mut p = State::new(c, s);
    let mut solver = Potential::new(c.cells, s.length);
    solver.solve(&p, c.bragg_mode);
    let mut out = ResultData::default();
    diagnostics(
        &mut out,
        &p,
        solver.energy(),
        c,
        s,
        0,
        0,
        0.,
        solver.rho.iter().sum::<f64>() / solver.rho.len() as f64,
    );
    out.density.push(solver.observable);
    for step in 1..=o.steps {
        let old_phi = solver.phi.clone();
        let mut mid = old_phi.clone();
        let mut accepted = None;
        for iteration in 1..=o.max_iterations {
            let trial = advance(&p, &mid, c, s, o.particle_iterations)?;
            solver.solve(&trial, c.bragg_mode);
            let residual: Vec<_> = mid
                .iter()
                .zip(&old_phi)
                .zip(&solver.phi)
                .map(|((m, a), b)| 0.5 * (a + b) - m)
                .collect();
            let rms = (residual.iter().map(|x| x * x).sum::<f64>() / mid.len() as f64).sqrt();
            let scale = (solver.phi.iter().map(|x| x * x).sum::<f64>() / mid.len() as f64).sqrt();
            if rms <= o.atol + o.rtol * scale {
                accepted = Some((trial, iteration, rms));
                break;
            }
            for (m, r) in mid.iter_mut().zip(residual) {
                *m += o.relaxation * r;
            }
        }
        let (trial, iterations, residual) = accepted.ok_or_else(|| {
            anyhow::anyhow!("Field iteration failed at step {step}; no unconverged step accepted")
        })?;
        p = trial;
        out.iterations.push(iterations as u32);
        out.residuals.push(residual);
        if step % o.diagnostic_stride == 0 || step == o.steps {
            out.density.push(solver.observable);
            diagnostics(
                &mut out,
                &p,
                solver.energy(),
                c,
                s,
                step,
                iterations,
                residual,
                solver.rho.iter().sum::<f64>() / solver.rho.len() as f64,
            );
        }
    }
    out.elapsed = start.elapsed().as_secs_f64();
    out.final_rho = solver.rho;
    out.final_phi = solver.phi;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn orbit_work_identity_across_periodic_piece_boundaries() {
        let n = 16;
        let length = 8.;
        let phi: Vec<_> = (0..n * n).map(|j| (j as f64 * 0.731).sin()).collect();
        for (x, delta) in [
            ([7.9, 0.1], [0.5, -0.9]),
            ([0.3, 2.2], [0.001, 0.]),
            ([3., 4.], [-2.1, 1.7]),
        ] {
            let e = orbit_field(&phi, x, delta, n, length).unwrap();
            let change = interpolate(&phi, [x[0] + delta[0], x[1] + delta[1]], n, length)
                - interpolate(&phi, x, n, length);
            assert!((change + e[0] * delta[0] + e[1] * delta[1]).abs() < 2e-13);
        }
    }
    #[test]
    fn coupled_energy_conserved_without_rescaling() {
        let o = Options::parse_from([
            "test",
            "--cells",
            "16",
            "--ppc",
            "2",
            "--steps",
            "40",
            "--rtol",
            "1e-11",
            "--atol",
            "1e-13",
            "--diagnostic-stride",
            "1",
        ]);
        let (c, s) = o.physical().unwrap();
        let r = simulate(&o, &c, &s).unwrap();
        let base = r.diagnostics[1];
        let error = r
            .diagnostics
            .as_chunks::<12>()
            .0
            .iter()
            .map(|d| (d[1] / base - 1.).abs())
            .fold(0f64, f64::max);
        assert!(error < 1e-10, "{error}");
    }
    #[test]
    fn rejects_unconverged_field_step() {
        let o = Options::parse_from([
            "test",
            "--cells",
            "16",
            "--ppc",
            "2",
            "--steps",
            "1",
            "--max-iterations",
            "1",
            "--rtol",
            "1e-14",
            "--atol",
            "1e-16",
        ]);
        let (c, s) = o.physical().unwrap();
        assert!(simulate(&o, &c, &s).is_err());
    }
}
