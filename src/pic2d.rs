//! Periodic square 2D2V electrostatic PIC. The radar mode is (m,0).
//! Grid order is y*N+x; each species has N*N*ppc macroparticles.
use crate::*;
use rustfft::{Fft, FftPlanner};
use std::{sync::Arc, time::Instant};

#[derive(Clone, Default)]
pub struct Particles2 {
    pub xe: Vec<[f32; 2]>,
    pub ve: Vec<[f32; 2]>,
    pub xi: Vec<[f32; 2]>,
    pub vi: Vec<[f32; 2]>,
}
impl Particles2 {
    pub fn new(c: &Config, s: &Scales, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = c.cells * c.cells * c.ppc;
        let mut positions = || {
            (0..n)
                .map(|_| {
                    [
                        rng.random::<f32>() * s.length as f32,
                        rng.random::<f32>() * s.length as f32,
                    ]
                })
                .collect()
        };
        let xe = positions();
        let xi = positions();
        let mut velocities = |std: f64| {
            let mut v: Vec<[f32; 2]> = (0..n)
                .map(|_| {
                    [
                        StandardNormal.sample(&mut rng),
                        StandardNormal.sample(&mut rng),
                    ]
                })
                .collect();
            for d in 0..2 {
                let mean = v.iter().map(|v| v[d] as f64).sum::<f64>() / n as f64;
                let variance =
                    v.iter().map(|v| (v[d] as f64 - mean).powi(2)).sum::<f64>() / n as f64;
                for v in &mut v {
                    v[d] = ((v[d] as f64 - mean) * std / variance.sqrt()) as f32;
                }
            }
            v
        };
        let ve = velocities(1.);
        let vi = velocities((c.ti / c.te / s.mass_ratio).sqrt());
        Self { xe, ve, xi, vi }
    }
}

pub fn weights(x: [f32; 2], n: usize, length: f32) -> [(usize, f32); 9] {
    let u = (x[0] / length * n as f32).rem_euclid(n as f32);
    let v = (x[1] / length * n as f32).rem_euclid(n as f32);
    let i = (u + 0.5).floor() as isize;
    let j = (v + 0.5).floor() as isize;
    let a = u - i as f32;
    let b = v - j as f32;
    let wx = [
        0.5 * (0.5 - a).powi(2),
        0.75 - a * a,
        0.5 * (0.5 + a).powi(2),
    ];
    let wy = [
        0.5 * (0.5 - b).powi(2),
        0.75 - b * b,
        0.5 * (0.5 + b).powi(2),
    ];
    let mut out = [(0, 0.); 9];
    for y in 0..3 {
        for x in 0..3 {
            out[y * 3 + x] = (
                ((j + y as isize - 1).rem_euclid(n as isize) as usize) * n
                    + (i + x as isize - 1).rem_euclid(n as isize) as usize,
                wx[x] * wy[y],
            );
        }
    }
    out
}
pub fn deposit(x: &[[f32; 2]], grid: &mut [f32], n: usize, length: f32) {
    grid.fill(0.);
    let norm = grid.len() as f32 / x.len() as f32;
    for &x in x {
        for (j, w) in weights(x, n, length) {
            grid[j] += norm * w;
        }
    }
}
pub fn gather(e: &[[f32; 2]], x: [f32; 2], n: usize, length: f32) -> [f32; 2] {
    let mut out = [0.; 2];
    for (j, w) in weights(x, n, length) {
        for d in 0..2 {
            out[d] += w * e[j][d];
        }
    }
    out
}
pub struct Solver2 {
    n: usize,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    column: Vec<Complex32>,
    scratch: Vec<Complex32>,
    ne: Vec<f32>,
    ni: Vec<f32>,
    freq: Vec<Complex32>,
    ex: Vec<Complex32>,
    ey: Vec<Complex32>,
    pub e: Vec<[f32; 2]>,
    pub charge: f64,
}
fn fft2(
    data: &mut [Complex32],
    n: usize,
    fft: &dyn Fft<f32>,
    col: &mut [Complex32],
    scratch: &mut [Complex32],
) {
    for row in data.chunks_exact_mut(n) {
        fft.process_with_scratch(row, scratch);
    }
    for x in 0..n {
        for y in 0..n {
            col[y] = data[y * n + x];
        }
        fft.process_with_scratch(col, scratch);
        for y in 0..n {
            data[y * n + x] = col[y];
        }
    }
}
impl Solver2 {
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(n);
        let inverse = planner.plan_fft_inverse(n);
        let scratch = vec![
            Complex32::default();
            forward
                .get_inplace_scratch_len()
                .max(inverse.get_inplace_scratch_len())
        ];
        Self {
            n,
            forward,
            inverse,
            column: vec![Complex32::default(); n],
            scratch,
            ne: vec![0.; n * n],
            ni: vec![0.; n * n],
            freq: vec![Complex32::default(); n * n],
            ex: vec![Complex32::default(); n * n],
            ey: vec![Complex32::default(); n * n],
            e: vec![[0.; 2]; n * n],
            charge: 0.,
        }
    }
    pub fn poisson(&mut self, rho: &[f32], length: f32) {
        for (f, &r) in self.freq.iter_mut().zip(rho) {
            *f = Complex32::new(r, 0.);
        }
        fft2(
            &mut self.freq,
            self.n,
            &*self.forward,
            &mut self.column,
            &mut self.scratch,
        );
        let n = self.n;
        for y in 0..n {
            for x in 0..n {
                let mode = |j: usize| {
                    if j <= n / 2 {
                        j as f32
                    } else {
                        j as f32 - n as f32
                    }
                };
                let kx = 2. * std::f32::consts::PI * mode(x) / length;
                let ky = 2. * std::f32::consts::PI * mode(y) / length;
                let k2 = kx * kx + ky * ky;
                let f = self.freq[y * n + x];
                let q = Complex32::new(f.im, -f.re);
                self.ex[y * n + x] = if k2 == 0. || x == n / 2 {
                    Complex32::default()
                } else {
                    q * (kx / k2)
                };
                self.ey[y * n + x] = if k2 == 0. || y == n / 2 {
                    Complex32::default()
                } else {
                    q * (ky / k2)
                };
            }
        }
        fft2(
            &mut self.ex,
            n,
            &*self.inverse,
            &mut self.column,
            &mut self.scratch,
        );
        fft2(
            &mut self.ey,
            n,
            &*self.inverse,
            &mut self.column,
            &mut self.scratch,
        );
        for j in 0..n * n {
            self.e[j] = [
                self.ex[j].re / (n * n) as f32,
                self.ey[j].re / (n * n) as f32,
            ];
        }
    }
    pub fn fields(&mut self, p: &Particles2, c: &Config, s: &Scales) -> Complex32 {
        self.fields_positions(&p.xe, &p.xi, c, s)
    }
    pub fn fields_positions(&mut self, xe: &[[f32;2]], xi: &[[f32;2]], c: &Config, s: &Scales) -> Complex32 {
        deposit(xe, &mut self.ne, self.n, s.length as f32);
        deposit(xi, &mut self.ni, self.n, s.length as f32);
        for (f, &r) in self.freq.iter_mut().zip(&self.ne) {
            *f = Complex32::new(r, 0.);
        }
        fft2(
            &mut self.freq,
            self.n,
            &*self.forward,
            &mut self.column,
            &mut self.scratch,
        );
        let kd = std::f64::consts::PI * c.bragg_mode as f64 / self.n as f64;
        let obs = self.freq[c.bragg_mode]
            * ((xe.len() as f64).sqrt() / (self.n * self.n) as f64 / (kd.sin() / kd).powi(3))
                as f32;
        for j in 0..self.n * self.n {
            self.ni[j] -= self.ne[j];
        }
        self.charge = self.ni.iter().map(|v| *v as f64).sum::<f64>() / (self.n * self.n) as f64;
        let rho = std::mem::take(&mut self.ni);
        self.poisson(&rho, s.length as f32);
        self.ni = rho;
        obs
    }
}
/// Boris velocity update, with signed q/m in electron units and B normalized
/// so that b = e B_z / (m_e omega_pe). Positions remain in the x-y plane.
pub fn boris(v: [f32; 2], e: [f32; 2], qm: f32, dt: f32, b: f32) -> [f32; 2] {
    if b == 0. { return [v[0] + dt*qm*e[0], v[1] + dt*qm*e[1]]; }
    let h = 0.5*dt*qm;
    let t = h*b;
    let s = 2.*t/(1.+t*t);
    let u = [v[0]+h*e[0], v[1]+h*e[1]];
    let prime = [u[0]+t*u[1], u[1]-t*u[0]];
    [u[0]+s*prime[1]+h*e[0], u[1]-s*prime[0]+h*e[1]]
}

pub fn magnetic_normalization(c: &Config, s: &Scales) -> f32 {
    (crate::QE*c.bz_t/(crate::ME*s.omega_pe)) as f32
}

pub fn push(p: &mut Particles2, e: &[[f32; 2]], c: &Config, s: &Scales, kick: f32, drift: f32) {
    let b = magnetic_normalization(c, s);
    for (x, v, qm) in [
        (&mut p.xe, &mut p.ve, -1.),
        (&mut p.xi, &mut p.vi, 1. / s.mass_ratio as f32),
    ] {
        for (x, v) in x.iter_mut().zip(v) {
            let field = gather(e, *x, c.cells, s.length as f32);
            *v = boris(*v, field, qm, kick, b);
            for d in 0..2 {
                x[d] = (x[d] + drift * v[d]).rem_euclid(s.length as f32);
            }
        }
    }
}
pub fn diagnostic2(
    run: &mut Run,
    p: &Particles2,
    e: &[[f32; 2]],
    c: &Config,
    s: &Scales,
    step: usize,
    charge: f64,
) {
    let mut sums = [0.; 5];
    for (j, x, v, m, q) in [
        (0, &p.xe, &p.ve, 1., -1.),
        (1, &p.xi, &p.vi, s.mass_ratio, 1.),
    ] {
        let mut mean = [0.; 2];
        let mut vv = 0.;
        for (i, (&x, &v)) in x.iter().zip(v).enumerate() {
            let field = gather(e, x, c.cells, s.length as f32);
            let centered = boris(v, field, (q/m) as f32, 0.5*c.dt as f32, magnetic_normalization(c,s));
            for d in 0..2 {
                let vel = centered[d] as f64;
                mean[d] += vel;
                vv += vel * vel;
                if i < 16 {
                    run.trajectory_x.push((x[d] as f64 * s.lambda_de) as f32);
                    run.trajectory_v.push((vel * s.vte) as f32);
                }
            }
        }
        for v in &mut mean {
            *v /= x.len() as f64;
        }
        vv /= x.len() as f64;
        sums[j] = 0.5 * m * vv;
        // T is per degree of freedom: subtract both species mean velocities.
        sums[2 + j] = 0.5 * m * c.te * (vv - mean[0] * mean[0] - mean[1] * mean[1]);
        sums[4] += m * mean[0];
    }
    let field = 0.5
        * e.iter()
            .map(|v| v[0] as f64 * v[0] as f64 + v[1] as f64 * v[1] as f64)
            .sum::<f64>()
        / e.len() as f64;
    run.diagnostics.extend([
        step as f64 * s.dt_s,
        sums[0] + sums[1] + field,
        sums[0],
        sums[1],
        field,
        sums[2],
        sums[3],
        sums[4],
        charge,
    ]);
}
pub fn simulate(c: &Config, s: &Scales, seed: u64) -> Result<Run> {
    let timer = Instant::now();
    let mut p = Particles2::new(c, s, seed);
    let mut solver = Solver2::new(c.cells);
    let mut run = Run::default();
    for step in 0..=s.burn_steps + s.steps {
        let obs = solver.fields(&p, c, s);
        if step < 2048 {
            for x in p.xe.iter().take(16).chain(p.xi.iter().take(16)) {
                for &v in x {
                    run.fast_trajectory_x.push((v as f64 * s.lambda_de) as f32);
                }
            }
        }
        if step == 0 {
            push(&mut p, &solver.e, c, s, -0.5 * c.dt as f32, 0.);
        }
        if step >= s.burn_steps && step < s.burn_steps + s.steps {
            run.voltage.push(obs);
        }
        if step % c.diagnostic_stride == 0 || step == s.burn_steps + s.steps {
            diagnostic2(&mut run, &p, &solver.e, c, s, step, solver.charge);
        }
        if step == s.burn_steps + s.steps {
            break;
        }
        push(&mut p, &solver.e, c, s, c.dt as f32, c.dt as f32);
    }
    run.elapsed_s = timer.elapsed().as_secs_f64();
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn boris_gyro_energy_sign_and_phase() {
        for qm in [-1., 1./29166.21578_f32] {
            let dt = 0.06;
            // Resolve 200 gyro periods for either species with identical rotation per step.
            let b = 1./(3.*qm.abs());
            let mut v = [1_f32, 0.];
            let steps = 62832;
            for _ in 0..steps { v = boris(v,[0.,0.],qm,dt,b); }
            assert!((v[0]*v[0]+v[1]*v[1]-1.).abs()<0.001);
            let angle = -2.*((0.5*dt*qm*b) as f64).atan()*steps as f64;
            assert!((v[0] as f64-angle.cos()).abs()<0.001);
            assert!((v[1] as f64-angle.sin()).abs()<0.001);
        }
    }
    #[test]
    fn boris_crossed_field_drift_and_reversibility() {
        let e = [0.2, -0.1];
        let b = 0.4;
        // E cross B / B^2 is a fixed velocity for both charge signs.
        for qm in [-1., 1./29166.21578_f32] {
            let drift = [e[1]/b, -e[0]/b];
            let moved = boris(drift,e,qm,0.06,b);
            assert!((moved[0]-drift[0]).abs()<1e-6 && (moved[1]-drift[1]).abs()<1e-6);
            let v = [0.7,-0.3];
            let back = boris(boris(v,e,qm,0.06,b),e,qm,-0.06,b);
            assert!((back[0]-v[0]).abs()<1e-6 && (back[1]-v[1]).abs()<1e-6);
        }
    }
    #[test]
    fn tsc_conserves_charge_and_gathers_constant() {
        let n = 16;
        let p = vec![[0., 0.], [15.9, 15.8], [16.1, -0.2], [7.2, 3.1]];
        let mut g = vec![0.; n * n];
        deposit(&p, &mut g, n, 16.);
        assert!((g.iter().sum::<f32>() - 256.).abs() < 1e-4);
        let e = vec![[2., -3.]; n * n];
        for x in p {
            let v = gather(&e, x, n, 16.);
            assert!((v[0] - 2.).abs() < 1e-6 && (v[1] + 3.).abs() < 1e-6);
        }
    }
    #[test]
    fn oblique_poisson_mode() {
        let n = 32;
        let l = 2. * std::f32::consts::PI;
        let rho: Vec<_> = (0..n * n)
            .map(|j| {
                (2. * l * (j % n) as f32 / n as f32 + 3. * l * (j / n) as f32 / n as f32).cos()
            })
            .collect();
        let mut solver = Solver2::new(n);
        solver.poisson(&rho, l);
        for (j, e) in solver.e.iter().enumerate() {
            let sine =
                (2. * l * (j % n) as f32 / n as f32 + 3. * l * (j / n) as f32 / n as f32).sin();
            assert!((e[0] - 2. / 13. * sine).abs() < 2e-6);
            assert!((e[1] - 3. / 13. * sine).abs() < 2e-6);
        }
    }
    #[test]
    fn transverse_force_and_periodic_drift() {
        let c = Config::parse_from(["t", "--cells", "32", "--bragg-mode", "1"]);
        let s = Scales::new(&c).unwrap();
        let mut p = Particles2 {
            xe: vec![[0.2, 0.3]],
            ve: vec![[0.1, 0.2]],
            xi: vec![[0.2, 0.3]],
            vi: vec![[0., 0.]],
        };
        push(&mut p, &vec![[2., -3.]; 1024], &c, &s, 0.05, 0.05);
        assert!(p.ve[0][0].abs() < 1e-7);
        assert!((p.ve[0][1] - 0.35).abs() < 1e-7);
        assert!((p.xe[0][1] - 0.3175).abs() < 1e-7);
        assert!(p.vi[0][0] > 0. && p.vi[0][1] < 0.);
    }
    #[test]
    fn nyquist_plane_retains_other_field_component() {
        let n = 32;
        let l = 2. * std::f32::consts::PI;
        let rho:Vec<_>=(0..n*n).map(|j|if j%n%2==0 {1.}else{-1.}*(2.*l*(j/n) as f32/n as f32).cos()).collect();
        let mut solver = Solver2::new(n);
        solver.poisson(&rho, l);
        for (j, e) in solver.e.iter().enumerate() {
            let expected = if j % n % 2 == 0 { 1. } else { -1. } * 2. / 260.
                * (2. * l * (j / n) as f32 / n as f32).sin();
            assert!(e[0].abs() < 1e-6);
            assert!((e[1] - expected).abs() < 1e-6);
        }
    }
    #[test]
    fn maxwellian_has_two_velocity_components() {
        let c = Config::parse_from(["t", "--cells", "32", "--ppc", "4", "--bragg-mode", "1"]);
        let s = Scales::new(&c).unwrap();
        let p = Particles2::new(&c, &s, 17);
        assert_eq!(p.xe.len(), 4096);
        for d in 0..2 {
            let mean = p.ve.iter().map(|v| v[d] as f64).sum::<f64>() / 4096.;
            let variance = p.ve.iter().map(|v| (v[d] as f64).powi(2)).sum::<f64>() / 4096.;
            assert!(mean.abs() < 1e-7);
            assert!((variance - 1.).abs() < 1e-7);
        }
    }
}
