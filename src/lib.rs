//! 1D1V electrostatic PIC, normalized to lambda_De, omega_pe^-1 and v_te.
//! The observable is the electron density mode, NOT free ion trajectories.
pub mod analysis;
pub mod cpu;
pub mod control2d;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod gpu2d;
#[cfg(feature = "cuda")]
pub mod dual_gpu2d;
#[cfg(feature = "cuda")]
pub mod gpu_batch;
pub mod implicit2d;
#[cfg(feature = "cuda")]
pub mod implicit_gpu;
pub mod output2d;
pub mod pic2d;
pub mod pic3v;
pub mod output3v;
#[cfg(feature = "cuda")]
pub mod gpu3v;
pub mod spectrum;

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rustfft::num_complex::Complex32;
use serde::{Deserialize, Serialize};

pub const KB: f64 = 1.380649e-23;
pub const ME: f64 = 9.1093837139e-31;
pub const QE: f64 = 1.602176634e-19;
pub const EPS0: f64 = 8.8541878188e-12;
pub const AMU: f64 = 1.66053906892e-27;
pub const C: f64 = 299792458.;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
pub enum Backend {
    Cpu,
    Cuda,
}

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
#[command(about = "Float32 periodic electron-ion PIC and a monostatic ISR spectrum")]
pub struct Config {
    #[arg(long, value_enum, default_value = "cpu")]
    pub backend: Backend,
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    #[arg(long, default_value_t = 128)]
    pub cells: usize,
    #[arg(long, default_value_t = 128)]
    pub ppc: usize,
    #[arg(long, default_value_t = 4)]
    pub bragg_mode: usize,
    #[arg(long, default_value_t = 440.2)]
    pub radar_mhz: f64,
    #[arg(long, default_value_t = 1e10)]
    pub density: f64,
    #[arg(long, default_value_t = 2000.)]
    pub te: f64,
    #[arg(long, default_value_t = 1000.)]
    pub ti: f64,
    /// Uniform out-of-plane magnetic field in tesla (explicit 2D2V only).
    #[arg(long, default_value_t = 0.)]
    #[serde(default)]
    pub bz_t: f64,
    #[arg(long, default_value_t = 16.)]
    pub ion_mass_amu: f64,
    /// Normalized timestep omega_pe * dt; do not use a reduced electron mass.
    #[arg(long, default_value_t = 0.12)]
    pub dt: f64,
    #[arg(long, default_value_t = 1.)]
    pub burn_ms: f64,
    #[arg(long, default_value_t = 8.)]
    pub duration_ms: f64,
    #[arg(long, default_value_t = 4.)]
    pub sample_us: f64,
    #[arg(long, default_value_t = 1024)]
    pub window: usize,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value_t = 1)]
    pub realizations: usize,
    /// Independent copies scheduled together on one GPU; physics is unchanged.
    #[arg(long, default_value_t = 1)]
    #[serde(default)]
    pub gpu_batch_size: usize,
    /// Save an individually usable HDF5 file for each completed realization.
    #[arg(long)]
    #[serde(default)]
    pub checkpoint_dir: Option<String>,
    #[arg(long, default_value_t = 1000)]
    pub diagnostic_stride: usize,
    /// Optional cooperative yielding between CUDA chunks on a shared host.
    #[arg(long, default_value_t = 0)]
    pub throttle_ms: u64,
    #[arg(long, default_value = "results/isr.h5")]
    pub output: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Scales {
    pub k_bragg: f64,
    pub lambda_de: f64,
    pub omega_pe: f64,
    pub vte: f64,
    pub mass_ratio: f64,
    pub length_m: f64,
    pub length: f64,
    pub dx: f64,
    pub dt_s: f64,
    pub burn_steps: usize,
    pub steps: usize,
    pub sample_stride: usize,
}
impl Scales {
    pub fn new(c: &Config) -> Result<Self> {
        if [
            c.density,
            c.te,
            c.ti,
            c.ion_mass_amu,
            c.radar_mhz,
            c.dt,
            c.duration_ms,
            c.burn_ms,
            c.sample_us,
            c.bz_t,
        ]
        .iter()
        .any(|v| !v.is_finite())
        {
            bail!("Physical and numerical parameters must be finite");
        }
        if c.cells < 16
            || !c.cells.is_power_of_two()
            || c.ppc < 1
            || c.bragg_mode == 0
            || c.bragg_mode > c.cells / 8
            || c.realizations == 0
            || c.gpu_batch_size == 0
            || c.gpu_batch_size > 32
            || c.seed.checked_add(c.realizations as u64).is_none()
            || c.diagnostic_stride == 0
            || c.density <= 0.
            || c.te <= 0.
            || c.ti <= 0.
            || c.ion_mass_amu <= 0.
            || c.radar_mhz <= 0.
            || !(0. ..=0.2).contains(&c.dt)
            || c.dt == 0.
            || c.duration_ms <= 0.
            || c.burn_ms < 0.
            || c.sample_us <= 0.
            || c.window < 16
            || !c.window.is_power_of_two()
        {
            bail!("Invalid grid, particle, timestep, spectrum, or physical parameters");
        }
        let k_bragg = 4. * std::f64::consts::PI * c.radar_mhz * 1e6 / C;
        let lambda_de = (EPS0 * KB * c.te / (c.density * QE * QE)).sqrt();
        let omega_pe = (c.density * QE * QE / (EPS0 * ME)).sqrt();
        let length_m = 2. * std::f64::consts::PI * c.bragg_mode as f64 / k_bragg;
        let length = length_m / lambda_de;
        let dx = length / c.cells as f64;
        let dt_s = c.dt / omega_pe;
        if dx > 1. || c.dt / dx > 0.7 {
            bail!(
                "Resolve Debye length and electron cell crossing: dx/lambda_De={dx}, dt/dx={}",
                c.dt / dx
            );
        }
        let steps = (c.duration_ms * 1e-3 / dt_s).round() as usize;
        let sample_stride = ((c.sample_us * 1e-6 / dt_s).round() as usize).max(1);
        if steps / sample_stride < c.window + 40 {
            bail!("Record too short for a spectrum after FIR edge trimming");
        }
        Ok(Self {
            k_bragg,
            lambda_de,
            omega_pe,
            vte: (KB * c.te / ME).sqrt(),
            mass_ratio: c.ion_mass_amu * AMU / ME,
            length_m,
            length,
            dx,
            dt_s,
            burn_steps: (c.burn_ms * 1e-3 / dt_s).round() as usize,
            steps,
            sample_stride,
        })
    }
}

#[derive(Clone)]
pub struct Particles {
    pub xe: Vec<f32>,
    pub ve: Vec<f32>,
    pub xi: Vec<f32>,
    pub vi: Vec<f32>,
}
impl Particles {
    pub fn new(c: &Config, s: &Scales, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = c.cells * c.ppc;
        let xe = (0..n)
            .map(|_| rng.random::<f32>() * s.length as f32)
            .collect();
        let xi = (0..n)
            .map(|_| rng.random::<f32>() * s.length as f32)
            .collect();
        let mut ve: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
        let mut vi: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
        // Zero species drift and exact target initial variance (fixed RNG seed).
        for (v, std) in [
            (&mut ve, 1.),
            (&mut vi, (c.ti / c.te / s.mass_ratio).sqrt()),
        ] {
            let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
            let var = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n as f64;
            for x in v {
                *x = ((*x as f64 - mean) * std / var.sqrt()) as f32;
            }
        }
        Self { xe, ve, xi, vi }
    }
}

#[derive(Default)]
pub struct Run {
    pub interaction_history: Vec<f64>,
    pub directional_temperatures: Vec<f64>,
    pub control_metadata: Option<serde_json::Value>,
    pub control_history: Vec<f64>,
    pub voltage: Vec<Complex32>,
    /// Columns: time_s, total_energy, electron_kinetic, ion_kinetic,
    /// field_energy, electron_temperature_K, ion_temperature_K, momentum,
    /// mean_charge. Leapfrog velocities are synchronized for diagnostics.
    pub diagnostics: Vec<f64>,
    pub trajectory_x: Vec<f32>,
    pub trajectory_v: Vec<f32>,
    /// 2D-only high-cadence positions: first 2048 steps, 32 particles, x/y.
    pub fast_trajectory_x: Vec<f32>,
    pub elapsed_s: f64,
}
pub fn diagnostic(run: &mut Run, p: &Particles, ef: &[f32], s: &Scales, c: &Config, step: usize) {
    let mut sums = [0f64; 5];
    for (x, v, charge, mass, j) in [
        (&p.xe, &p.ve, -1., 1., 0),
        (&p.xi, &p.vi, 1., s.mass_ratio, 1),
    ] {
        let mut vv = 0.;
        let mut mean = 0.;
        for (&x, &v) in x.iter().zip(v) {
            let a = cpu::gather(ef, x, s.length as f32) as f64 * charge / mass;
            let centered = v as f64 + 0.5 * c.dt * a;
            vv += centered * centered;
            mean += centered;
        }
        mean /= v.len() as f64;
        vv /= v.len() as f64;
        sums[j] = 0.5 * mass * vv;
        sums[2 + j] = mass * (vv - mean * mean) * c.te;
        sums[4] += mass * mean;
        for (&x, &v) in x.iter().zip(v).take(16) {
            run.trajectory_x.push((x as f64 * s.lambda_de) as f32);
            let a = cpu::gather(ef, x, s.length as f32) as f64 * charge / mass;
            run.trajectory_v
                .push(((v as f64 + 0.5 * c.dt * a) * s.vte) as f32);
        }
    }
    let field = 0.5 * ef.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / ef.len() as f64;
    run.diagnostics.extend([
        step as f64 * s.dt_s,
        sums[0] + sums[1] + field,
        sums[0],
        sums[1],
        field,
        sums[2],
        sums[3],
        sums[4],
        0.,
    ]);
}

#[cfg(test)]
mod parameter_tests {
    use super::*;
    #[test]
    fn reject_nan() {
        let mut c = Config::parse_from(["test"]);
        c.te = f64::NAN;
        assert!(Scales::new(&c).is_err());
    }
    #[test]
    fn reject_unresolved() {
        let mut c = Config::parse_from(["test"]);
        c.density = 1e14;
        assert!(Scales::new(&c).is_err());
    }
    #[test]
    fn physical_bragg_scale() {
        let c = Config::parse_from(["test"]);
        let s = Scales::new(&c).unwrap();
        assert!((s.k_bragg - 18.45181957).abs() < 1e-7);
        assert!((s.mass_ratio - 29166.21578).abs() < 0.001);
    }
    #[test]
    fn one_ppc_is_valid_but_zero_is_not() {
        let mut c=Config::parse_from(["test","--cells","512","--ppc","1","--bragg-mode","2","--density","1e11","--dt",".06"]);
        assert!(Scales::new(&c).is_ok());
        c.ppc=0;
        assert!(Scales::new(&c).is_err());
    }
}
