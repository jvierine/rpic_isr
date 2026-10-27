use anyhow::{bail, Result};
use clap::Parser;
use rpic_isr::*;
use rustfft::num_complex::Complex32;
use std::path::Path;

fn main() -> Result<()> {
    let c = Config::parse();
    anyhow::ensure!(c.bz_t == 0., "Use rpic_isr_2d for a magnetic field");
    let s = Scales::new(&c)?;
    if Path::new(&c.output).exists() {
        bail!("Output already exists: {}; choose a new filename", c.output);
    }
    if let Some(p) = Path::new(&c.output).parent() {
        std::fs::create_dir_all(p)?;
    }
    println!("{}", serde_json::to_string_pretty(&s)?);
    let mut records = Vec::new();
    let mut runs = Vec::new();
    #[cfg(feature = "cuda")]
    let mut pending = std::collections::VecDeque::new();
    for r in 0..c.realizations {
        let seed = c.seed + r as u64;
        let run = match c.backend {
            Backend::Cpu => cpu::simulate(&c, &s, seed)?,
            Backend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    if c.gpu_batch_size <= 1 {
                        gpu::simulate(&c, &s, seed)?
                    } else {
                        if pending.is_empty() {
                            let end = (r + c.gpu_batch_size).min(c.realizations);
                            let seeds: Vec<_> = (r..end).map(|j| c.seed + j as u64).collect();
                            pending = gpu_batch::simulate(&c, &s, &seeds)?.into();
                        }
                        pending.pop_front().unwrap()
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    bail!("Rebuild with --features cuda");
                }
            }
        };
        let filtered = spectrum::decimate(&run.voltage, s.sample_stride);
        let energy: Vec<f64> = run
            .diagnostics
            .as_chunks::<9>()
            .0
            .iter()
            .map(|r| r[1])
            .collect();
        let drift = energy
            .iter()
            .map(|x| (x / energy[0] - 1.).abs())
            .fold(0f64, f64::max);
        println!(
            "realization {r} seed={seed}: {:.2}s; {} samples; max energy drift {:.3}%",
            run.elapsed_s,
            filtered.len(),
            drift * 100.
        );
        if let Some(dir) = &c.checkpoint_dir {
            let mut single = c.clone();
            single.seed = seed;
            single.realizations = 1;
            single.checkpoint_dir = None;
            single.output = Path::new(dir)
                .join(format!("seed{seed}.h5"))
                .to_string_lossy()
                .into_owned();
            save(
                &single,
                &s,
                std::slice::from_ref(&filtered),
                std::slice::from_ref(&run),
            )?;
        }
        records.push(filtered);
        runs.push(run);
    }
    save(&c, &s, &records, &runs)
}

fn save(c: &Config, s: &Scales, records: &[Vec<Complex32>], runs: &[Run]) -> Result<()> {
    if Path::new(&c.output).exists() {
        bail!("Refusing to overwrite {}", c.output);
    }
    if let Some(p) = Path::new(&c.output).parent() {
        std::fs::create_dir_all(p)?;
    }
    let dt = s.dt_s * s.sample_stride as f64;
    let spec = spectrum::welch(records, dt, c.window, c, s)?;
    let temporary = format!("{}.tmp", c.output);
    let f = hdf5::File::create(&temporary)?;
    let metadata = serde_json::json!({"source_revision":env!("RPIC_BUILD_REV"),"package_version":env!("CARGO_PKG_VERSION"),"model":"periodic 1D1V electrostatic PIC","observable":"CIC-corrected electron density mode / sqrt(Np)","psd_unit":"Hz^-1","diagnostic_columns":["time_s","total_energy_kBTe","electron_kinetic_kBTe","ion_kinetic_kBTe","field_energy_kBTe","Te_K","Ti_K","momentum_me_vte","mean_charge_ne"]});
    f.new_dataset_builder()
        .with_data(serde_json::to_vec(&metadata)?.as_slice())
        .create("metadata_json")?;
    f.new_dataset_builder()
        .with_data(serde_json::to_string_pretty(&c)?.as_bytes())
        .create("config_json")?;
    f.new_dataset_builder()
        .with_data(serde_json::to_string_pretty(&s)?.as_bytes())
        .create("scales_json")?;
    f.new_attr::<f64>()
        .create("sample_interval_s")?
        .write_scalar(&dt)?;
    f.new_attr::<u64>()
        .create("welch_segments")?
        .write_scalar(&(spec.segments as u64))?;
    for (name, values) in [
        ("frequency_hz", spec.frequency),
        ("psd", spec.power),
        ("theory_psd", spec.theory),
        ("acf_real", spec.acf.iter().map(|z| z.re).collect()),
        ("acf_imag", spec.acf.iter().map(|z| z.im).collect()),
    ] {
        f.new_dataset_builder().with_data(&values).create(name)?;
    }
    let raw = f.create_group("raw")?;
    let sampled = f.create_group("sampled")?;
    let diag = f.create_group("diagnostics")?;
    let trajectories = f.create_group("trajectories")?;
    for r in 0..runs.len() {
        let save_complex = |g: &hdf5::Group, z: &[Complex32]| -> Result<()> {
            g.new_dataset_builder()
                .with_data(&z.iter().map(|z| z.re).collect::<Vec<_>>())
                .create("real")?;
            g.new_dataset_builder()
                .with_data(&z.iter().map(|z| z.im).collect::<Vec<_>>())
                .create("imag")?;
            Ok(())
        };
        save_complex(&raw.create_group(&r.to_string())?, &runs[r].voltage)?;
        save_complex(&sampled.create_group(&r.to_string())?, &records[r])?;
        let nr = runs[r].diagnostics.len() / 9;
        diag.new_dataset::<f64>()
            .shape([nr, 9])
            .create(r.to_string().as_str())?
            .write_raw(&runs[r].diagnostics)?;
        let g = trajectories.create_group(&r.to_string())?;
        g.new_dataset::<f32>()
            .shape([nr, 32])
            .create("x_m")?
            .write_raw(&runs[r].trajectory_x)?;
        g.new_dataset::<f32>()
            .shape([nr, 32])
            .create("v_m_s")?
            .write_raw(&runs[r].trajectory_v)?;
    }
    drop((raw, sampled, diag, trajectories));
    f.flush()?;
    drop(f);
    std::fs::rename(&temporary, &c.output)?;
    println!(
        "Saved {} ({} Welch segments; overlapping segments are not independent)",
        c.output, spec.segments
    );
    Ok(())
}
