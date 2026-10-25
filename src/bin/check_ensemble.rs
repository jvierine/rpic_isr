//! Independent-realization uncertainty and early/late stationarity diagnostic.
use anyhow::{ensure, Result};
use rpic_isr::analysis::{self, Realization};
fn main() -> Result<()> {
    let paths: Vec<_> = std::env::args().skip(1).collect();
    ensure!(!paths.is_empty(), "Usage: check_ensemble INPUT.h5 ...");
    let mut all: Vec<Realization> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut freq = Vec::new();
    let mut theory = Vec::new();
    for path in paths {
        let f = hdf5::File::open(path)?;
        let axis = f.dataset("frequency_hz")?.read_raw::<f64>()?;
        if freq.is_empty() {
            freq = axis.clone();
            theory = f.dataset("theory_psd")?.read_raw::<f64>()?;
        }
        ensure!(
            axis.len() == freq.len() && axis.iter().zip(&freq).all(|(x, y)| (x - y).abs() < 1e-8),
            "Mismatched axes"
        );
        for run in analysis::read(&f)? {
            ensure!(seen.insert(run.seed), "Repeated seed {}", run.seed);
            all.push(run);
        }
    }
    let idx: Vec<_> = freq
        .iter()
        .enumerate()
        .filter(|(_, f)| f.abs() < 12000.)
        .map(|(i, _)| i)
        .collect();
    ensure!(
        all.iter().all(|r| r.early_psd.len() == freq.len()),
        "Each half-record needs a complete spectral window for stationarity analysis"
    );
    let norm = idx.iter().map(|&i| theory[i]).sum::<f64>();
    let band = |p: &[f64]| idx.iter().map(|&i| p[i]).sum::<f64>() / norm;
    let powers: Vec<_> = all.iter().map(|r| band(&r.psd)).collect();
    let early: Vec<_> = all.iter().map(|r| band(&r.early_psd)).collect();
    let late: Vec<_> = all.iter().map(|r| band(&r.late_psd)).collect();
    let diff: Vec<_> = early.iter().zip(&late).map(|(a, b)| b - a).collect();
    for (label, v) in [
        ("Integrated power / theory", &powers),
        ("First half / theory", &early),
        ("Second half / theory", &late),
        ("Late minus early / theory (paired)", &diff),
    ] {
        let (m, se) = analysis::mean_se(v);
        println!(
            "{label}: {m:.6} +/- {se:.6} (1 SE across {} independent seeds)",
            all.len()
        );
    }
    let max = |f: fn(&Realization) -> f64| all.iter().map(f).fold(0f64, f64::max);
    println!("Max energy drift {:.5}%; max Te change {:.5}%; max Ti change {:.5}%; max mean charge {:.3e}",max(|r|r.max_energy_percent),max(|r|r.max_te_percent),max(|r|r.max_ti_percent),max(|r|r.max_charge));
    for (species, start) in [("electrons", 0), ("ions", 16)] {
        let v: Vec<_> = all
            .iter()
            .flat_map(|r| r.final_velocities[start..start + 16].iter().copied())
            .collect();
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
        let kurt = v.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / v.len() as f64 / (var * var);
        println!("Selected final {species}: N={}, mean={mean:.3} m/s, std={:.3} m/s, kurtosis={kurt:.4} (Gaussian: 3)",v.len(),var.sqrt());
    }
    Ok(())
}
