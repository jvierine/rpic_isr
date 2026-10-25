//! Compare matched seeds at two plasma ages; do not treat repeated seeds as
//! independent samples in the production ensemble.
use anyhow::{ensure, Result};
use rpic_isr::{analysis, Config};
fn main() -> Result<()> {
    let paths: Vec<_> = std::env::args().skip(1).collect();
    ensure!(paths.len() == 2, "Usage: compare_burn EARLY.h5 LATE.h5");
    let a = hdf5::File::open(&paths[0])?;
    let b = hdf5::File::open(&paths[1])?;
    let ca: Config = serde_json::from_slice(&a.dataset("config_json")?.read_raw::<u8>()?)?;
    let cb: Config = serde_json::from_slice(&b.dataset("config_json")?.read_raw::<u8>()?)?;
    ensure!(
        ca.cells == cb.cells
            && ca.ppc == cb.ppc
            && ca.dt == cb.dt
            && ca.te == cb.te
            && ca.ti == cb.ti
            && ca.density == cb.density
            && ca.ion_mass_amu == cb.ion_mass_amu
            && ca.radar_mhz == cb.radar_mhz
            && ca.bragg_mode == cb.bragg_mode
            && ca.window == cb.window
            && ca.duration_ms == cb.duration_ms,
        "Only burn-in, seed range, and scheduling may differ"
    );
    let axis = a.dataset("frequency_hz")?.read_raw::<f64>()?;
    let bx = b.dataset("frequency_hz")?.read_raw::<f64>()?;
    ensure!(axis == bx, "Different frequency axes");
    let theory = a.dataset("theory_psd")?.read_raw::<f64>()?;
    let indices: Vec<_> = axis
        .iter()
        .enumerate()
        .filter(|(_, f)| f.abs() < 12000.)
        .map(|(i, _)| i)
        .collect();
    let norm = indices.iter().map(|&i| theory[i]).sum::<f64>();
    let integrate = |p: &[f64]| indices.iter().map(|&i| p[i]).sum::<f64>() / norm;
    let early = analysis::read(&a)?;
    let late = analysis::read(&b)?;
    let mut va = Vec::new();
    let mut vb = Vec::new();
    for later in &late {
        let earlier = early
            .iter()
            .find(|r| r.seed == later.seed)
            .ok_or_else(|| anyhow::anyhow!("No early match for seed {}", later.seed))?;
        va.push(integrate(&earlier.psd));
        vb.push(integrate(&later.psd));
    }
    let differences: Vec<_> = va.iter().zip(&vb).map(|(a, b)| b - a).collect();
    for (label, values) in [
        (format!("{} ms burn-in / theory", ca.burn_ms), &va),
        (format!("{} ms burn-in / theory", cb.burn_ms), &vb),
        ("Late minus early / theory (paired)".into(), &differences),
    ] {
        let (mean, se) = analysis::mean_se(values);
        println!(
            "{label}: {mean:.6} +/- {se:.6} (1 SE; {} matched seeds)",
            values.len()
        );
    }
    Ok(())
}
