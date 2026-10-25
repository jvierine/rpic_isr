//! Ensemble uncertainty is calculated across independent complete realizations,
//! not by counting overlapping Welch windows as independent observations.
use crate::*;
use anyhow::ensure;

pub struct Realization {
    pub seed: u64,
    pub psd: Vec<f64>,
    pub early_psd: Vec<f64>,
    pub late_psd: Vec<f64>,
    pub max_energy_percent: f64,
    pub max_te_percent: f64,
    pub max_ti_percent: f64,
    pub max_charge: f64,
    pub final_velocities: Vec<f64>,
}
pub fn read(file: &hdf5::File) -> Result<Vec<Realization>> {
    let mut c: Config = serde_json::from_slice(&file.dataset("config_json")?.read_raw::<u8>()?)?;
    c.gpu_batch_size = c.gpu_batch_size.max(1); // legacy pilot metadata
    let s = Scales::new(&c)?;
    let dt = file.attr("sample_interval_s")?.read_scalar::<f64>()?;
    let mut runs = Vec::new();
    for r in 0..c.realizations {
        let re = file
            .dataset(&format!("sampled/{r}/real"))?
            .read_raw::<f32>()?;
        let im = file
            .dataset(&format!("sampled/{r}/imag"))?
            .read_raw::<f32>()?;
        ensure!(
            re.len() == im.len() && re.len() >= c.window,
            "Need a complete spectral window in each record"
        );
        let z: Vec<_> = re
            .iter()
            .zip(im)
            .map(|(x, y)| Complex32::new(*x, y))
            .collect();
        ensure!(
            z.iter().all(|z| z.re.is_finite() && z.im.is_finite()),
            "Nonfinite density record"
        );
        let split = z.len() / 2;
        let psd = spectrum::welch(std::slice::from_ref(&z), dt, c.window, &c, &s)?.power;
        let (early, late) = if split >= c.window {
            (
                spectrum::welch(&[z[..split].to_vec()], dt, c.window, &c, &s)?.power,
                spectrum::welch(&[z[split..].to_vec()], dt, c.window, &c, &s)?.power,
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let d = file
            .dataset(&format!("diagnostics/{r}"))?
            .read_raw::<f64>()?;
        ensure!(
            d.len() >= 9 && d.iter().all(|v| v.is_finite()),
            "Invalid diagnostics"
        );
        let max = |j: usize, base: f64, factor: f64| {
            d.as_chunks::<9>()
                .0
                .iter()
                .map(|row| ((row[j] - base) * factor).abs())
                .fold(0f64, f64::max)
        };
        let velocities = file
            .dataset(&format!("trajectories/{r}/v_m_s"))?
            .read_raw::<f32>()?;
        let last = velocities.len() - 32;
        runs.push(Realization {
            seed: c.seed + r as u64,
            psd,
            early_psd: early,
            late_psd: late,
            max_energy_percent: max(1, d[1], 100. / d[1]),
            max_te_percent: max(5, d[5], 100. / d[5]),
            max_ti_percent: max(6, d[6], 100. / d[6]),
            max_charge: max(8, 0., 1.),
            final_velocities: velocities[last..].iter().map(|v| *v as f64).collect(),
        });
    }
    Ok(runs)
}
pub fn mean_se(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let se = if values.len() > 1 {
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n * (n - 1.))).sqrt()
    } else {
        f64::NAN
    };
    (mean, se)
}
