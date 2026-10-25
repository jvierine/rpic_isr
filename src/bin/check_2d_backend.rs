//! Deterministic, short-trajectory CPU/CUDA regression (not a late-time test).
use anyhow::{ensure, Result};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(args.len() == 2, "Usage: check_2d_backend CPU.h5 GPU.h5");
    let a = hdf5::File::open(&args[0])?;
    let b = hdf5::File::open(&args[1])?;
    for key in [
        "raw/0/real",
        "raw/0/imag",
        "trajectories/0/x_m",
        "trajectories/0/v_m_s",
    ] {
        let da = a.dataset(key)?;
        let db = b.dataset(key)?;
        ensure!(da.shape() == db.shape(), "Shape differs: {key}");
        let x = da.read_raw::<f32>()?;
        let y = db.read_raw::<f32>()?;
        let error = (x
            .iter()
            .zip(&y)
            .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
            .sum::<f64>()
            / x.iter().map(|x| (*x as f64).powi(2)).sum::<f64>())
        .sqrt();
        println!("{key}: relative L2 error {error:.3e}");
        ensure!(error < 1e-4, "CPU/GPU short-run mismatch: {key}");
    }
    for file in [&a, &b] {
        let d = file.dataset("diagnostics/0")?.read_raw::<f64>()?;
        let drift = d
            .chunks_exact(9)
            .map(|r| (r[1] / d[1] - 1.).abs())
            .fold(0f64, f64::max);
        println!("Maximum energy excursion: {:.6}%", 100. * drift);
        ensure!(drift < 0.001, "Short-run energy regression");
    }
    Ok(())
}
