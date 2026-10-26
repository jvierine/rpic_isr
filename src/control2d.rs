//! Explicit temperature control and an auditable energy ledger, not energy-conserving PIC.
use crate::*;
use clap::{Args, ValueEnum};
use std::path::Path;

#[derive(Clone, Copy, Debug, Serialize, ValueEnum, PartialEq)]
pub enum Thermostat {
    Off,
    AfterEquilibrium,
}
#[derive(Clone, Debug, Args, Serialize)]
pub struct Controls {
    #[arg(long, value_delimiter = ',', default_value = "0,1")]
    pub devices: Vec<usize>,
    #[arg(long, value_enum, default_value = "after-equilibrium")]
    pub thermostat: Thermostat,
    /// Operational stationarity gate; not proof of full kinetic equilibrium.
    #[arg(long, default_value_t = 0.5)]
    pub equilibrium_window_ms: f64,
    #[arg(long, default_value_t = 0.005)]
    pub equilibrium_tolerance: f64,
    #[arg(long, default_value_t = 2.)]
    pub thermostat_tau_ms: f64,
    #[arg(long, default_value_t = 4.)]
    pub thermostat_settle_ms: f64,
    /// Guard the energy balance INCLUDING cumulative thermostat energy removal.
    #[arg(long, default_value_t = 1.)]
    pub max_energy_drift_percent: f64,
    #[arg(long)]
    pub live_diagnostics: Option<String>,
    /// Graceful diagnostic-checkpoint deadline; also use an outer process timeout.
    #[arg(long)]
    pub max_wall_seconds: Option<f64>,
}
#[derive(Clone, Debug)]
pub struct State {
    pub row: [f64; 9],
    pub mean: [[f64; 2]; 2],
    pub anisotropy: [f64; 2],
    pub radial_kurtosis: [f64; 2],
}
pub fn from_moments(m: &[f64], n: usize, cells: usize, c: &Config, s: &Scales, time: f64) -> State {
    let mut mean = [[0.; 2]; 2];
    let mut ke = [0.; 2];
    let mut temp = [0.; 2];
    let mut aniso = [0.; 2];
    let mut kurt = [0.; 2];
    for species in 0..2 {
        let j = species * 6;
        let mass = if species == 0 { 1. } else { s.mass_ratio };
        mean[species] = [m[j] / n as f64, m[j + 1] / n as f64];
        let vx = m[j + 2] / n as f64 - mean[species][0].powi(2);
        let vy = m[j + 3] / n as f64 - mean[species][1].powi(2);
        ke[species] = 0.5 * mass * (m[j + 2] + m[j + 3]) / n as f64;
        temp[species] = 0.5 * mass * c.te * (vx + vy);
        aniso[species] = (vx - vy) / (vx + vy);
        // Valid as a Maxwellian diagnostic only when mean drift is negligible.
        kurt[species] = (m[j + 5] / n as f64) / ((m[j + 2] + m[j + 3]) / n as f64).powi(2);
    }
    let field = 0.5 * m[12] / (cells * cells) as f64;
    State {
        row: [
            time,
            ke[0] + ke[1] + field,
            ke[0],
            ke[1],
            field,
            temp[0],
            temp[1],
            mean[0][0] + s.mass_ratio * mean[1][0],
            m[13] / (cells * cells) as f64,
        ],
        mean,
        anisotropy: aniso,
        radial_kurtosis: kurt,
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    Burn = 0,
    Settle = 1,
    Record = 2,
}
pub struct Controller {
    pub options: Controls,
    pub phase: Phase,
    pub activation_time: Option<f64>,
    pub record_step: Option<usize>,
    pub history: Vec<f64>,
    pub removed: f64,
    initial_energy: Option<f64>,
    last_time: Option<f64>,
    samples: Vec<State>,
}
impl Controller {
    pub fn new(options: Controls) -> Result<Self> {
        anyhow::ensure!(
            options
                .max_wall_seconds
                .is_none_or(|x| x.is_finite() && x > 0.),
            "Invalid wall-time budget"
        );
        anyhow::ensure!(
            !options.devices.is_empty() && options.devices.len() <= 2,
            "Select one or two GPUs"
        );
        anyhow::ensure!(
            options.devices.len() == 1 || options.devices[0] != options.devices[1],
            "GPU IDs must differ"
        );
        for x in [
            options.equilibrium_window_ms,
            options.equilibrium_tolerance,
            options.thermostat_tau_ms,
            options.max_energy_drift_percent,
        ] {
            anyhow::ensure!(
                x.is_finite() && x > 0.,
                "Control parameters must be positive and finite"
            );
        }
        anyhow::ensure!(
            options.thermostat_settle_ms.is_finite() && options.thermostat_settle_ms >= 0.,
            "Invalid settle interval"
        );
        Ok(Self {
            options,
            phase: Phase::Burn,
            activation_time: None,
            record_step: None,
            history: vec![],
            removed: 0.,
            initial_energy: None,
            last_time: None,
            samples: vec![],
        })
    }
    pub fn stable(&self) -> bool {
        let Some(last) = self.samples.last() else {
            return false;
        };
        let t = last.row[0];
        let width = self.options.equilibrium_window_ms * 1e-3;
        let a: Vec<_> = self
            .samples
            .iter()
            .filter(|x| x.row[0] >= t - width && x.row[0] < t - width / 2.)
            .collect();
        let b: Vec<_> = self
            .samples
            .iter()
            .filter(|x| x.row[0] >= t - width / 2.)
            .collect();
        if a.len() < 4 || b.len() < 4 {
            return false;
        }
        for species in 0..2 {
            let j = 5 + species;
            let ma = a.iter().map(|x| x.row[j]).sum::<f64>() / a.len() as f64;
            let mb = b.iter().map(|x| x.row[j]).sum::<f64>() / b.len() as f64;
            if b.iter().any(|x| {
                !x.anisotropy[species].is_finite()
                    || !x.radial_kurtosis[species].is_finite()
                    || x.mean[species].iter().any(|v| !v.is_finite())
            }) {
                return false;
            }
            if (mb / ma - 1.).abs() > self.options.equilibrium_tolerance {
                return false;
            }
            if b.iter().map(|x| x.anisotropy[species].abs()).sum::<f64>() / b.len() as f64 > 0.05 {
                return false;
            }
            if b.iter()
                .map(|x| (x.radial_kurtosis[species] - 2.).abs())
                .sum::<f64>()
                / b.len() as f64
                > 0.1
            {
                return false;
            }
        }
        true
    }
    pub fn decide(&mut self, pre: &State, step: usize, c: &Config) -> Result<[f64; 2]> {
        anyhow::ensure!(
            pre.row.iter().all(|x| x.is_finite()) && pre.row[5] > 0. && pre.row[6] > 0.,
            "Non-finite/non-positive plasma diagnostic"
        );
        self.initial_energy.get_or_insert(pre.row[1]);
        self.samples.push(pre.clone());
        let floor = pre.row[0] - self.options.equilibrium_window_ms * 1e-3 * 1.1;
        self.samples.retain(|x| x.row[0] >= floor);
        if self.options.thermostat == Thermostat::Off {
            if pre.row[0] >= c.burn_ms * 1e-3 && self.record_step.is_none() {
                self.record_step = Some(step);
                self.phase = Phase::Record;
            }
        } else {
            if self.phase == Phase::Burn && pre.row[0] >= c.burn_ms * 1e-3 && self.stable() {
                self.phase = Phase::Settle;
                self.activation_time = Some(pre.row[0]);
            }
            if self.phase == Phase::Settle
                && pre.row[0] - self.activation_time.unwrap()
                    >= self.options.thermostat_settle_ms * 1e-3
                && self.stable()
                && (pre.row[5] / c.te - 1.).abs() < 0.01
                && (pre.row[6] / c.ti - 1.).abs() < 0.01
            {
                self.phase = Phase::Record;
                self.record_step = Some(step);
            }
        }
        let interval = pre.row[0] - self.last_time.unwrap_or(pre.row[0]);
        self.last_time = Some(pre.row[0]);
        if self.options.thermostat == Thermostat::Off || self.phase == Phase::Burn {
            return Ok([1., 1.]);
        }
        let fraction = (interval / (self.options.thermostat_tau_ms * 1e-3)).min(1.);
        Ok([
            (1. + fraction * (c.te / pre.row[5] - 1.)).sqrt(),
            (1. + fraction * (c.ti / pre.row[6] - 1.)).sqrt(),
        ])
    }
    pub fn record(&mut self, pre: &State, post: &State, alpha: [f64; 2]) -> Result<()> {
        let work = pre.row[1] - post.row[1];
        self.removed += work;
        let balance = post.row[1] + self.removed;
        let drift = 100. * (balance / self.initial_energy.unwrap() - 1.);
        self.history.extend([
            pre.row[0],
            pre.row[1],
            post.row[1],
            post.row[5],
            post.row[6],
            work,
            self.removed,
            balance,
            drift,
            self.phase as u8 as f64,
            alpha[0],
            alpha[1],
            post.anisotropy[0],
            post.anisotropy[1],
            post.radial_kurtosis[0],
            post.radial_kurtosis[1],
        ]);
        anyhow::ensure!(
            post.row.iter().all(|x| x.is_finite())
                && drift.abs() <= self.options.max_energy_drift_percent,
            "Energy balance guard: drift={drift:.6}% (thermostat removal included)"
        );
        Ok(())
    }
    pub fn metadata(&self) -> serde_json::Value {
        serde_json::json!({"options":self.options,"thermostatted":self.activation_time.is_some(),"energy_conserving":false,"gate":"minimum burn plus two-half temperature stationarity, anisotropy and zero-drift radial kurtosis; not proof of equilibrium","activation_time_s":self.activation_time,"record_step":self.record_step,"removed_energy_kBTe":self.removed,"columns":["time_s","energy_pre","energy_post","Te_K","Ti_K","thermostat_work_removed","cumulative_work_removed","energy_plus_removed_work","balance_drift_percent","phase_0burn_1settle_2record","alpha_e","alpha_i","anisotropy_e","anisotropy_i","radial_kurtosis_e","radial_kurtosis_i"]})
    }
    pub fn checkpoint(
        &self,
        path: &str,
        c: &Config,
        s: &Scales,
        status: &str,
        diagnostics: &[f64],
    ) -> Result<()> {
        if let Some(p) = Path::new(path).parent() {
            std::fs::create_dir_all(p)?;
        }
        let tmp = format!("{path}.tmp");
        let f = hdf5::File::create(&tmp)?;
        for (name, data) in [
            ("config_json", serde_json::to_vec(c)?),
            ("scales_json", serde_json::to_vec(s)?),
            ("control_json", serde_json::to_vec(&self.metadata())?),
            ("status", status.as_bytes().to_vec()),
        ] {
            f.new_dataset_builder().with_data(&data).create(name)?;
        }
        f.new_dataset::<f64>()
            .shape([self.history.len() / 16, 16])
            .create("control_history")?
            .write_raw(&self.history)?;
        f.new_dataset::<f64>()
            .shape([diagnostics.len() / 9, 9])
            .create("diagnostics")?
            .write_raw(diagnostics)?;
        f.flush()?;
        drop(f);
        std::fs::rename(tmp, path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options() -> Controls {
        Controls {
            devices: vec![0, 1],
            thermostat: Thermostat::AfterEquilibrium,
            equilibrium_window_ms: 0.5,
            equilibrium_tolerance: 0.005,
            thermostat_tau_ms: 2.,
            thermostat_settle_ms: 4.,
            max_energy_drift_percent: 1.,
            live_diagnostics: None,
            max_wall_seconds: None,
        }
    }
    fn state(time: f64, temp: f64) -> State {
        State {
            row: [time, 1.5, 1., 0.5, 0., temp, 1000., 0., 0.],
            mean: [[0.; 2]; 2],
            anisotropy: [0.; 2],
            radial_kurtosis: [2.; 2],
        }
    }
    #[test]
    fn gate_requires_stationarity_and_minimum_burn() {
        let c = Config::parse_from(["test", "--burn-ms", "1"]);
        let mut x = Controller::new(options()).unwrap();
        for j in 0..20 {
            let a = x.decide(&state(j as f64 * 0.00005, 2000.), j, &c).unwrap();
            assert_eq!(a, [1., 1.]);
        }
        x.decide(&state(0.001, 2000.), 20, &c).unwrap();
        assert_eq!(x.phase, Phase::Settle);
    }
    #[test]
    fn thermostat_does_not_hide_created_energy() {
        let c = Config::parse_from(["test"]);
        let mut x = Controller::new(options()).unwrap();
        let a = state(0., 2000.);
        x.decide(&a, 0, &c).unwrap();
        x.record(&a, &a, [1., 1.]).unwrap();
        let mut pre = a.clone();
        pre.row[1] = 1.53;
        let mut post = pre.clone();
        post.row[1] = 1.5;
        assert!(x.record(&pre, &post, [0.99, 0.99]).is_err());
        assert!((x.removed - 0.03).abs() < 1e-12);
    }
    #[test]
    fn species_have_distinct_targets() {
        let c = Config::parse_from(["test"]);
        let mut x = Controller::new(options()).unwrap();
        x.phase = Phase::Settle;
        x.activation_time = Some(0.);
        x.decide(&state(0., 2000.), 0, &c).unwrap();
        let mut warm = state(0.0001, 2100.);
        warm.row[6] = 900.;
        let a = x.decide(&warm, 1, &c).unwrap();
        assert!(a[0] < 1. && a[1] > 1.);
        assert_eq!(c.te, 2000.);
        assert_eq!(c.ti, 1000.);
    }
    #[test]
    fn nonfinite_shape_is_not_stationary() {
        let mut x = Controller::new(options()).unwrap();
        for j in 0..20 {
            let mut a = state(j as f64 * 0.00005, 2000.);
            a.radial_kurtosis[0] = f64::NAN;
            x.samples.push(a);
        }
        assert!(!x.stable());
    }
}
