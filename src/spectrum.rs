//! Anti-alias FIR, two-sided Welch PSD and collisionless Maxwellian benchmark.
use crate::*;
use rustfft::{num_complex::Complex64, FftPlanner};

/// Linear FIR filtering, not unfiltered decimation. Discard both edge transients.
pub fn decimate(x: &[Complex32], stride: usize) -> Vec<Complex32> {
    let half = 16 * stride;
    let fc = 0.4 / stride as f64; // cycles/raw sample; final Nyquist=0.5/stride.
    let mut h: Vec<Complex32> = (-((half) as isize)..=half as isize)
        .map(|j| {
            let t = j as f64;
            let z = 2. * fc * t;
            let sinc = if j == 0 {
                1.
            } else {
                (std::f64::consts::PI * z).sin() / (std::f64::consts::PI * z)
            };
            let u = (t + half as f64) / (2 * half) as f64;
            let w = 0.42 - 0.5 * (2. * std::f64::consts::PI * u).cos()
                + 0.08 * (4. * std::f64::consts::PI * u).cos();
            Complex32::new((2. * fc * sinc * w) as f32, 0.)
        })
        .collect();
    let norm = h.iter().map(|x| x.re).sum::<f32>();
    for v in &mut h {
        *v /= norm;
    }
    let size = (x.len() + h.len() - 1).next_power_of_two();
    let mut a = vec![Complex32::default(); size];
    a[..x.len()].copy_from_slice(x);
    h.resize(size, Complex32::default());
    let mut p = FftPlanner::new();
    let f = p.plan_fft_forward(size);
    let inv = p.plan_fft_inverse(size);
    f.process(&mut a);
    f.process(&mut h);
    for (v, w) in a.iter_mut().zip(h) {
        *v *= w;
    }
    inv.process(&mut a);
    (half..x.len() - half)
        .step_by(stride)
        .map(|j| a[j + half] / size as f32)
        .collect()
}

pub struct Spectrum {
    pub frequency: Vec<f64>,
    pub power: Vec<f64>,
    pub theory: Vec<f64>,
    pub acf: Vec<Complex64>,
    pub segments: usize,
}
pub fn welch(
    records: &[Vec<Complex32>],
    dt: f64,
    n: usize,
    c: &Config,
    s: &Scales,
) -> Result<Spectrum> {
    if n < 2
        || !n.is_power_of_two()
        || dt <= 0.
        || !dt.is_finite()
        || records.iter().any(|x| x.len() < n)
    {
        bail!("Invalid Welch window, sample interval, or record length");
    }
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let w: Vec<f32> = (0..n)
        .map(|j| (0.5 - 0.5 * (2. * std::f64::consts::PI * j as f64 / n as f64).cos()) as f32)
        .collect();
    let norm = w.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
    let mut sums = vec![0.; n];
    let mut segments = 0;
    for x in records {
        for start in (0..=x.len() - n).step_by(n / 2) {
            // No per-segment demeaning: a zero-frequency density component is
            // physical. Uniform ensemble mean of nonzero spatial modes is zero.
            let mut a: Vec<Complex32> = x[start..start + n]
                .iter()
                .zip(&w)
                .map(|(x, w)| *x * *w)
                .collect();
            fft.process(&mut a);
            for (p, z) in sums.iter_mut().zip(a) {
                *p += z.norm_sqr() as f64 * dt / norm;
            }
            segments += 1;
        }
    }
    if segments == 0 {
        bail!("No complete Welch segments");
    }
    for p in &mut sums {
        *p /= segments as f64;
    }
    let frequency: Vec<f64> = (0..n)
        .map(|j| (j as f64 - n as f64 / 2.) / (n as f64 * dt))
        .collect();
    let power = (0..n).map(|j| sums[(j + n / 2) % n]).collect();
    let theory = frequency.iter().map(|f| theory(*f, c, s)).collect();
    let mut acf: Vec<Complex64> = sums
        .iter()
        .map(|p| Complex64::new(*p / (n as f64 * dt), 0.))
        .collect();
    // Earlier voltage times conjugated later: negative-exponent transform.
    let mut p64 = FftPlanner::new();
    p64.plan_fft_forward(n).process(&mut acf);
    Ok(Spectrum {
        frequency,
        power,
        theory,
        acf,
        segments,
    })
}
fn dawson(x: f64) -> f64 {
    if x < 0. {
        return -dawson(-x);
    }
    if x >= 6. {
        let mut sum = 1.;
        let mut term = 1.;
        for j in 1..12 {
            term *= (2 * j - 1) as f64 / (2. * x * x);
            sum += term;
        }
        return sum / (2. * x);
    }
    let n = 1024;
    let step = 1. / n as f64;
    let sum = (0..=n)
        .map(|j| {
            let u = j as f64 * step;
            let w = if j == 0 || j == n {
                1.
            } else if j % 2 == 0 {
                2.
            } else {
                4.
            };
            w * (-x * x * (1. - u * u)).exp()
        })
        .sum::<f64>();
    x * step * sum / 3.
}
fn susceptibility(f: f64, k: f64, v: f64, alpha: f64) -> Complex64 {
    let z = 2. * std::f64::consts::PI * f / (k * 2f64.sqrt() * v);
    let plasma_z = Complex64::new(
        -2. * dawson(z),
        std::f64::consts::PI.sqrt() * (-z * z).exp(),
    );
    (Complex64::new(1., 0.) + plasma_z * z) * alpha
}
pub fn theory(f: f64, c: &Config, s: &Scales) -> f64 {
    let k = s.k_bragg;
    let vi = (KB * c.ti / (c.ion_mass_amu * AMU)).sqrt();
    let alpha = 1. / (k * s.lambda_de).powi(2);
    let e = susceptibility(f, k, s.vte, alpha);
    let i = susceptibility(f, k, vi, alpha * c.te / c.ti);
    let dielectric = Complex64::new(1., 0.) + e + i;
    let electron = (Complex64::new(1., 0.) - e / dielectric).norm_sqr();
    let ion = (e / dielectric).norm_sqr();
    let gaussian = |v: f64| {
        let sf = k * v / (2. * std::f64::consts::PI);
        (-0.5 * (f / sf).powi(2)).exp() / (sf * (2. * std::f64::consts::PI).sqrt())
    };
    electron * gaussian(s.vte) + ion * gaussian(vi)
}

/// Expected finite-window spectrum: quadrature of the continuous Maxwellian
/// spectrum against the periodic-Hann spectral response. Eightfold frequency
/// oversampling resolves the kernel rather than merely smoothing FFT-bin values.
/// The sampled observable has already been anti-aliased; this calculation is
/// for its ion band, far below the FIR transition and final Nyquist frequency.
pub fn windowed_theory(c: &Config, s: &Scales, dt: f64) -> Vec<f64> {
    let oversample = 8;
    let n = c.window;
    let len = n * oversample;
    let dense: Vec<_> = (0..len)
        .map(|j| {
            let signed = if j <= len / 2 {
                j as f64
            } else {
                j as f64 - len as f64
            };
            theory(signed / (len as f64 * dt), c, s)
        })
        .collect();
    let smooth = hann_convolve(&dense, n);
    (0..n)
        .map(|j| smooth[((j + n / 2) % n) * oversample])
        .collect()
}

fn hann_convolve(density: &[f64], n: usize) -> Vec<f64> {
    let len = density.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(len);
    let inv = planner.plan_fft_inverse(len);
    let mut w = vec![Complex64::default(); len];
    for (j, z) in w.iter_mut().take(n).enumerate() {
        z.re = 0.5 - 0.5 * (2. * std::f64::consts::PI * j as f64 / n as f64).cos();
    }
    let sumw2 = w.iter().map(|z| z.norm_sqr()).sum::<f64>();
    fft.process(&mut w);
    for z in &mut w {
        *z = Complex64::new(z.norm_sqr() / (len as f64 * sumw2), 0.);
    }
    fft.process(&mut w);
    let mut data: Vec<_> = density.iter().map(|x| Complex64::new(*x, 0.)).collect();
    fft.process(&mut data);
    for (x, y) in data.iter_mut().zip(w) {
        *x *= y;
    }
    inv.process(&mut data);
    data.iter().map(|z| (z.re / len as f64).max(0.)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hann_response_preserves_constant() {
        let p = hann_convolve(&vec![2.; 4096], 512);
        assert!(p.iter().all(|x| (x - 2.).abs() < 1e-12));
    }
    #[test]
    fn dawson_values() {
        assert!((dawson(1.) - 0.5380795069).abs() < 1e-8);
        assert!((dawson(8.) - 0.0630001987).abs() < 1e-8);
    }
    #[test]
    fn welch_parseval_and_acf() {
        let c = Config::parse_from(["test"]);
        let s = Scales::new(&c).unwrap();
        let dt = 4e-6;
        let n = 1024;
        let tone: Vec<_> = (0..4096)
            .map(|j| {
                Complex32::from_polar(1., 2. * std::f32::consts::PI * 32. * j as f32 / n as f32)
            })
            .collect();
        let a = welch(&[tone], dt, n, &c, &s).unwrap();
        let area = a.power.iter().sum::<f64>() / (dt * n as f64);
        assert!((area - 1.).abs() < 1e-5);
        assert!((a.acf[0].re - 1.).abs() < 1e-5);
        let peak = a
            .power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert!((a.frequency[peak] - 32. / (n as f64 * dt)).abs() < 1e-6);
        assert!(a.acf[1].im < 0., "ACF conjugates the later sample");
    }
    #[test]
    fn anti_alias_and_phase() {
        let n = 32768;
        let stride = 16;
        let x: Vec<_> = (0..n)
            .map(|j| Complex32::from_polar(1., 2. * std::f32::consts::PI * 0.005 * j as f32))
            .collect();
        let y = decimate(&x, stride);
        assert!((y[100].norm() - 1.).abs() < 1e-3);
        let x: Vec<_> = (0..n)
            .map(|j| Complex32::from_polar(1., 2. * std::f32::consts::PI * 0.1 * j as f32))
            .collect();
        let y = decimate(&x, stride);
        assert!(y.iter().map(|z| z.norm_sqr()).sum::<f32>() / (y.len() as f32) < 1e-6);
    }
}
