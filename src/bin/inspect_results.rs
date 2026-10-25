//! Read-only validation summary of executed HDF5 records.
use anyhow::{ensure, Result};
use rpic_isr::Config;

fn main() -> Result<()> {
    let paths: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        !paths.is_empty(),
        "Usage: inspect_results INPUT.h5 [INPUT.h5 ...]"
    );
    for path in paths {
        let f = hdf5::File::open(&path)?;
        let c: Config = serde_json::from_slice(&f.dataset("config_json")?.read_raw::<u8>()?)?;
        println!(
            "{path}: {:?}, seed={}, realizations={}",
            c.backend, c.seed, c.realizations
        );
        for name in f.group("diagnostics")?.member_names()? {
            let d = f
                .dataset(&format!("diagnostics/{name}"))?
                .read_raw::<f64>()?;
            ensure!(d.iter().all(|v| v.is_finite()), "Nonfinite diagnostics");
            let maximum = |column: usize, baseline: f64, scale: f64| {
                d.as_chunks::<9>()
                    .0
                    .iter()
                    .map(|r| ((r[column] - baseline) * scale).abs())
                    .fold(0f64, f64::max)
            };
            println!("  run {name}: max energy drift={:.5}%; max |Te/Te0-1|={:.5}%; max |Ti/Ti0-1|={:.5}%; mean charge={:.3e}",maximum(1,d[1],100./d[1]),maximum(5,d[5],100./d[5]),maximum(6,d[6],100./d[6]),maximum(8,0.,1.));
        }
        let x = f.dataset("frequency_hz")?.read_raw::<f64>()?;
        let p = f.dataset("psd")?.read_raw::<f64>()?;
        let t = f.dataset("theory_psd")?.read_raw::<f64>()?;
        ensure!(p.iter().all(|v| v.is_finite() && *v >= 0.), "Invalid PSD");
        let mut a = 0.;
        let mut b = 0.;
        let mut l1 = 0.;
        for j in 0..x.len() {
            if x[j].abs() < 12000. {
                a += p[j];
                b += t[j];
                l1 += (p[j] - t[j]).abs();
            }
        }
        println!(
            "  |f|<12kHz: PIC/theory area={:.6}; relative L1 shape error={:.5}; Welch windows={}",
            a / b,
            l1 / b,
            f.attr("welch_segments")?.read_scalar::<u64>()?
        );
    }
    Ok(())
}
