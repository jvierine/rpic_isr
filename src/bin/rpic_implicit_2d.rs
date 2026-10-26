use anyhow::{ensure, Result};
use clap::Parser;
use rpic_isr::{implicit2d::*, *};
fn main() -> Result<()> {
    let o = Options::parse();
    let (c, s) = o.physical()?;
    ensure!(
        !std::path::Path::new(&o.output).exists(),
        "Refusing to overwrite {}",
        o.output
    );
    let result = match o.backend {
        Backend::Cpu => implicit2d::simulate(&o, &c, &s)?,
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                implicit_gpu::simulate(&o, &c, &s)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("Rebuild with --features cuda");
            }
        }
    };
    if let Some(p) = std::path::Path::new(&o.output).parent() {
        std::fs::create_dir_all(p)?;
    }
    let f = hdf5::File::create(format!("{}.tmp", o.output))?;
    let metadata = serde_json::json!({"source_revision":env!("RPIC_BUILD_REV"),"model":"experimental periodic 2D2V discrete-work implicit PIC","method":"midpoint velocities; piecewise Gauss orbit-average of gradient of TSC midpoint potential; nonlinear particle/Poisson convergence","not_literal_CCB_2011":"2D potential-based extension, not the paper's 1D Vlasov-Ampere current mover or JFNK implementation","energy":"K + 0.5 mean(rho phi); full symmetric spectral Poisson including Nyquist potential modes","precision":if matches!(o.backend,Backend::Cpu){"float64 reference"}else{"float32 particles, deposition and cuFFT; float64 residual/energy reductions"},"diagnostic_columns":COLUMNS,"thermostat":false,"rescaling":false,"density_cadence":"diagnostic times only; not a production ISR spectrum"});
    for (name, data) in [
        ("metadata_json", serde_json::to_vec(&metadata)?),
        ("config_json", serde_json::to_vec(&c)?),
        ("scales_json", serde_json::to_vec(&s)?),
        ("implicit_options_json", serde_json::to_vec(&o)?),
    ] {
        f.new_dataset_builder().with_data(&data).create(name)?;
    }
    f.new_dataset::<f64>()
        .shape([result.diagnostics.len() / 12, 12])
        .create("diagnostics")?
        .write_raw(&result.diagnostics)?;
    f.new_dataset::<f64>()
        .shape([result.trace.len() / 64, 32, 2])
        .create("positions_m")?
        .write_raw(&result.trace)?;
    for (name, data) in [
        (
            "density_real",
            result.density.iter().map(|z| z.re).collect::<Vec<_>>(),
        ),
        (
            "density_imag",
            result.density.iter().map(|z| z.im).collect(),
        ),
    ] {
        f.new_dataset_builder().with_data(&data).create(name)?;
    }
    f.new_dataset_builder()
        .with_data(&result.iterations)
        .create("iterations")?;
    f.new_dataset_builder()
        .with_data(&result.residuals)
        .create("residual_rms")?;
    f.new_attr::<f64>()
        .create("elapsed_s")?
        .write_scalar(&result.elapsed)?;
    f.flush()?;
    drop(f);
    std::fs::rename(format!("{}.tmp", o.output), &o.output)?;
    let initial = result.diagnostics[1];
    let drift = result
        .diagnostics
        .as_chunks::<12>()
        .0
        .iter()
        .map(|d| (d[1] / initial - 1.).abs())
        .fold(0f64, f64::max);
    println!(
        "Saved {}: {:.2}s, max energy excursion {:.8}%, mean nonlinear iterations {:.2}",
        o.output,
        result.elapsed,
        drift * 100.,
        result.iterations.iter().map(|&v| v as f64).sum::<f64>() / o.steps as f64
    );
    Ok(())
}
