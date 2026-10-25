use crate::*;
use std::path::Path;
pub fn save(
    c: &Config,
    s: &Scales,
    records: &[Vec<Complex32>],
    runs: &[Run],
    dimensions: usize,
) -> Result<()> {
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
    let metadata = serde_json::json!({"source_revision":env!("RPIC_BUILD_REV"),"package_version":env!("CARGO_PKG_VERSION"),"model":format!("periodic {dimensions}D{dimensions}V electrostatic PIC"),"dimensions":dimensions,"grid_shape":[c.cells,c.cells],"particles_per_species":c.cells*c.cells*c.ppc,"bragg_mode_xy":[c.bragg_mode,0],"trajectory_axes":["diagnostic_time","particle (16 electrons then 16 ions)","component (x,y)"],"temperature_definition":"mean of the x and y central velocity variances times mass/kB","initialization":"independent uniform positions and Gaussian velocities per component, zero species drift, exact target component variance; not Debye-correlated equilibrium","particle_shape":"quadratic B-spline (TSC)","fast_trajectory_interval":"dt_s","observable":"TSC-corrected electron density mode / sqrt(Np)","psd_unit":"Hz^-1","diagnostic_columns":["time_s","total_energy_kBTe","electron_kinetic_kBTe","ion_kinetic_kBTe","field_energy_kBTe","Te_K","Ti_K","x_momentum_me_vte","mean_charge_ne"]});
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
            .shape([nr, 32, dimensions])
            .create("x_m")?
            .write_raw(&runs[r].trajectory_x)?;
        g.new_dataset::<f32>()
            .shape([nr, 32, dimensions])
            .create("v_m_s")?
            .write_raw(&runs[r].trajectory_v)?;
        g.new_dataset::<f32>()
            .shape([runs[r].fast_trajectory_x.len() / 64, 32, 2])
            .create("fast_x_m")?
            .write_raw(&runs[r].fast_trajectory_x)?;
        g.new_attr::<f64>()
            .create("fast_interval_s")?
            .write_scalar(&s.dt_s)?;
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
