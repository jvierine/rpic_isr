use crate::*;
use rustfft::{Fft, FftPlanner};
use std::{sync::Arc, time::Instant};

pub fn deposit(x: &[f32], grid: &mut [f32], length: f32) {
    grid.fill(0.);
    let n = grid.len();
    let weight = n as f32 / x.len() as f32;
    for &x in x {
        let u = (x / length * n as f32).rem_euclid(n as f32);
        let j = u.floor() as usize % n;
        let f = u - u.floor();
        grid[j] += weight * (1. - f);
        grid[(j + 1) % n] += weight * f;
    }
}
pub fn gather(e: &[f32], x: f32, length: f32) -> f32 {
    let u = (x / length * e.len() as f32).rem_euclid(e.len() as f32);
    let j = u.floor() as usize % e.len();
    let f = u - u.floor();
    e[j] * (1. - f) + e[(j + 1) % e.len()] * f
}
pub struct Solver {
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    ne: Vec<f32>,
    ni: Vec<f32>,
    buf: Vec<Complex32>,
    scratch: Vec<Complex32>,
    pub e: Vec<f32>,
}
impl Solver {
    pub fn new(n: usize) -> Self {
        let mut p = FftPlanner::new();
        let forward = p.plan_fft_forward(n);
        let inverse = p.plan_fft_inverse(n);
        let scratch = vec![
            Complex32::default();
            forward
                .get_inplace_scratch_len()
                .max(inverse.get_inplace_scratch_len())
        ];
        Self {
            forward,
            inverse,
            ne: vec![0.; n],
            ni: vec![0.; n],
            buf: vec![Complex32::default(); n],
            scratch,
            e: vec![0.; n],
        }
    }
    pub fn poisson(&mut self, rho: &[f32], length: f32) {
        let n = rho.len();
        for (v, &r) in self.buf.iter_mut().zip(rho) {
            *v = Complex32::new(r, 0.);
        }
        self.forward
            .process_with_scratch(&mut self.buf, &mut self.scratch);
        for (j, v) in self.buf.iter_mut().enumerate() {
            let mode = if j <= n / 2 {
                j as f32
            } else {
                j as f32 - n as f32
            };
            if j == 0 || j == n / 2 {
                *v = Complex32::default();
            } else {
                let k = 2. * std::f32::consts::PI * mode / length;
                *v = Complex32::new(v.im / k, -v.re / k);
            }
        }
        self.inverse
            .process_with_scratch(&mut self.buf, &mut self.scratch);
        for (e, v) in self.e.iter_mut().zip(&self.buf) {
            *e = v.re / n as f32;
        }
    }
    pub fn fields(&mut self, p: &Particles, s: &Scales, c: &Config) -> Complex32 {
        deposit(&p.xe, &mut self.ne, s.length as f32);
        deposit(&p.xi, &mut self.ni, s.length as f32);
        for (v, &r) in self.buf.iter_mut().zip(&self.ne) {
            *v = Complex32::new(r, 0.);
        }
        self.forward
            .process_with_scratch(&mut self.buf, &mut self.scratch);
        let kd = std::f64::consts::PI * c.bragg_mode as f64 / c.cells as f64;
        let shape = (kd.sin() / kd).powi(2);
        let obs =
            self.buf[c.bragg_mode] * ((p.xe.len() as f64).sqrt() / c.cells as f64 / shape) as f32;
        for j in 0..self.ni.len() {
            self.ni[j] -= self.ne[j];
        }
        // Reuse the ion-density allocation as charge density to avoid per-step allocation.
        let rho = std::mem::take(&mut self.ni);
        self.poisson(&rho, s.length as f32);
        self.ni = rho;
        obs
    }
}
pub fn simulate(c: &Config, s: &Scales, seed: u64) -> Result<Run> {
    let timer = Instant::now();
    let mut p = Particles::new(c, s, seed);
    let mut solver = Solver::new(c.cells);
    solver.fields(&p, s, c);
    for (x, v, a) in [
        (&p.xe, &mut p.ve, -1.),
        (&p.xi, &mut p.vi, 1. / s.mass_ratio),
    ] {
        for (&x, v) in x.iter().zip(v) {
            *v -= 0.5 * c.dt as f32 * a as f32 * gather(&solver.e, x, s.length as f32);
        }
    }
    let mut out = Run::default();
    out.voltage.reserve(s.steps);
    for step in 0..s.burn_steps + s.steps {
        let obs = solver.fields(&p, s, c);
        if step >= s.burn_steps {
            out.voltage.push(obs);
        }
        if step % c.diagnostic_stride == 0 {
            diagnostic(&mut out, &p, &solver.e, s, c, step);
            *out.diagnostics.last_mut().unwrap() =
                solver.ni.iter().map(|x| *x as f64).sum::<f64>() / c.cells as f64;
        }
        for (x, v, a) in [
            (&mut p.xe, &mut p.ve, -1.),
            (&mut p.xi, &mut p.vi, 1. / s.mass_ratio),
        ] {
            for (x, v) in x.iter_mut().zip(v) {
                *v += c.dt as f32 * a as f32 * gather(&solver.e, *x, s.length as f32);
                *x = (*x + c.dt as f32 * *v).rem_euclid(s.length as f32);
            }
        }
    }
    solver.fields(&p, s, c);
    diagnostic(&mut out, &p, &solver.e, s, c, s.burn_steps + s.steps);
    out.elapsed_s = timer.elapsed().as_secs_f64();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn charge_partition() {
        let x = vec![0., 0.01, 0.99, 1.0, -0.1];
        let mut g = vec![0.; 32];
        deposit(&x, &mut g, 1.);
        assert!((g.iter().sum::<f32>() - 32.).abs() < 1e-5);
    }
    #[test]
    fn constant_field_gather() {
        assert!((gather(&[2.; 32], -3.15, 1.) - 2.).abs() < 1e-6);
    }
    #[test]
    fn analytic_poisson() {
        let n = 128;
        let length = 8.;
        let k = 2. * std::f32::consts::PI / length;
        let r: Vec<f32> = (0..n)
            .map(|j| (k * j as f32 * length / n as f32).cos())
            .collect();
        let mut p = Solver::new(n);
        p.poisson(&r, length);
        for (j, e) in p.e.iter().enumerate() {
            let exact = (k * j as f32 * length / n as f32).sin() / k;
            assert!((e - exact).abs() < 3e-6, "{e} {exact}");
        }
    }
    #[test]
    fn zero_mode() {
        let mut p = Solver::new(32);
        p.poisson(&[1.; 32], 2.);
        assert!(p.e.iter().all(|x| x.abs() < 1e-6));
    }
}
