use crate::*;
use std::path::Path;
pub fn save(
    c: &crate::pic3v::Setup,
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
    f.new_attr::<f64>().create("magnetic_field_z_t")?.write_scalar(&c.bz_t)?;
    let metadata = serde_json::json!({"source_revision":env!("RPIC_BUILD_REV"),"package_version":env!("CARGO_PKG_VERSION"),"model":"periodic 2D3V electrostatic PIC","dimensions":dimensions,"velocity_dimensions":3,"magnetic_field_xyz_t":[c.bx_t,c.by_t,c.bz_t],"hot_fraction":c.hot_fraction,"hot_energy_ev":c.hot_energy_ev,"grid_shape":[c.cells,c.cells],"particles_per_species":c.cells*c.cells*c.ppc,"bragg_mode_xy":[c.bragg_mode,0],"trajectory_axes":["diagnostic_time","particle (16 electrons then 16 ions)","component (x,y)"],"temperature_definition":"mean of three central velocity variances times mass/kB; electron Te is population-weighted core and tail","initialization":"independent uniform positions and Gaussian velocities per component, zero species drift, exact target component variance; not Debye-correlated equilibrium","particle_shape":"quadratic B-spline (TSC)","fast_trajectory_interval":"dt_s","observable":"TSC-corrected electron density mode / sqrt(Np)","psd_unit":"Hz^-1","diagnostic_columns":["time_s","total_energy_kBTe","electron_kinetic_kBTe","ion_kinetic_kBTe","field_energy_kBTe","Te_K","Ti_K","x_momentum_me_vte","mean_charge_ne"]});
    let mut metadata = metadata;
    metadata["population_energy_columns"] = serde_json::json!(["time_s", "core_kinetic", "tail_kinetic", "ion_kinetic", "core_T_K", "tail_T_K", "ion_T_K", "core_J_dot_E", "tail_J_dot_E", "ion_J_dot_E", "cumulative_thermostat_energy_added"]);
    metadata["energy_unit"] = serde_json::json!("N_e k_B T_core_initial");
    metadata["population_power_unit"] = serde_json::json!("N_e k_B T_core_initial omega_pe; instantaneous sampled power");
    metadata["directional_temperature_columns"] = serde_json::json!(["core_parallel", "core_perpendicular", "tail_parallel", "tail_perpendicular", "ion_parallel", "ion_perpendicular"]);
    metadata["velocity_trajectory_components"] = serde_json::json!(["x", "y", "z"]);
    metadata["initialization"] = serde_json::json!("Independent uniform positions; independent Gaussian velocities with zero drift and exact component variance separately for core, tail and ions. Hot energy is mean 3D kinetic energy. Not Debye-correlated equilibrium.");
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
        // The existing analytic model is unmagnetized; never mislabel it as Bernstein theory.
        if name == "theory_psd" { continue; }
        f.new_dataset_builder().with_data(&values).create(name)?;
    }
    let raw = f.create_group("raw")?;
    let sampled = f.create_group("sampled")?;
    let diag = f.create_group("diagnostics")?;
    let trajectories = f.create_group("trajectories")?;
    // A separate 20 kHz ion-band output; keep the original high-rate spectral
    // record unchanged. Anti-alias filtering precedes the 50 us decimation.
    let snapshots = f.create_group("sampled_50us")?;
    let cadence = (50e-6 / s.dt_s).round().max(1.) as usize;
    snapshots
        .new_attr::<f64>()
        .create("sample_interval_s")?
        .write_scalar(&(cadence as f64 * s.dt_s))?;
    snapshots
        .new_attr::<f64>()
        .create("first_sample_after_burn_s")?
        .write_scalar(&(16. * cadence as f64 * s.dt_s))?;
    snapshots
        .new_attr::<f64>()
        .create("fir_cutoff_hz")?
        .write_scalar(&(0.4 / (cadence as f64 * s.dt_s)))?;
    for r in 0..runs.len() {
        if let Some(ref control)=runs[r].control_metadata {
            let g=f.create_group(&format!("control_{r}"))?;
            g.new_dataset_builder().with_data(&serde_json::to_vec(control)?).create("metadata_json")?;
            g.new_dataset::<f64>().shape([runs[r].control_history.len()/16,16]).create("history")?.write_raw(&runs[r].control_history)?;
        }
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
        if runs[r].voltage.len() > 32 * cadence {
            let coarse = spectrum::decimate(&runs[r].voltage, cadence);
            save_complex(&snapshots.create_group(&r.to_string())?, &coarse)?;
        }
        let nr = runs[r].diagnostics.len() / 9;
        diag.new_dataset::<f64>()
            .shape([nr, 9])
            .create(r.to_string().as_str())?
            .write_raw(&runs[r].diagnostics)?;
        f.new_dataset::<f64>().shape([runs[r].interaction_history.len()/11,11]).create(format!("population_energy_{r}").as_str())?.write_raw(&runs[r].interaction_history)?;
        f.new_dataset::<f64>().shape([runs[r].directional_temperatures.len()/6,6]).create(format!("parallel_perpendicular_temperatures_K_{r}").as_str())?.write_raw(&runs[r].directional_temperatures)?;
        let g = trajectories.create_group(&r.to_string())?;
        g.new_dataset::<f32>()
            .shape([nr, 32, dimensions])
            .create("x_m")?
            .write_raw(&runs[r].trajectory_x)?;
        g.new_dataset::<f32>()
            .shape([nr, 32, 3])
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
    drop((raw, sampled, diag, trajectories, snapshots));
    f.flush()?;
    drop(f);
    std::fs::rename(&temporary, &c.output)?;
    println!(
        "Saved {} ({} Welch segments; overlapping segments are not independent)",
        c.output, spec.segments
    );
    Ok(())
}

