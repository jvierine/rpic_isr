//! Native Rustplotlib rendering from HDF5; no Matplotlib or fitted rescaling.
use anyhow::{ensure, Result};
use rustplotlib::*;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() >= 2,
        "Usage: plot_results OUTPUT_DIR INPUT.h5 [INPUT.h5 ...]"
    );
    std::fs::create_dir_all(&args[0])?;
    let files: Vec<_> = args[1..]
        .iter()
        .map(hdf5::File::open)
        .collect::<std::result::Result<_, _>>()?;
    let first = &files[0];
    let f = first.dataset("frequency_hz")?.read_raw::<f64>()?;
    let theory = first.dataset("theory_psd")?.read_raw::<f64>()?;
    let mut physical: rpic_isr::Config =
        serde_json::from_slice(&first.dataset("config_json")?.read_raw::<u8>()?)?;
    physical.gpu_batch_size = physical.gpu_batch_size.max(1);
    let scales = rpic_isr::Scales::new(&physical)?;
    let matched_theory = rpic_isr::spectrum::windowed_theory(
        &physical,
        &scales,
        first.attr("sample_interval_s")?.read_scalar::<f64>()?,
    );
    let mut p = vec![0.; f.len()];
    let mut count = 0.;
    let mut seeds = std::collections::HashSet::new();
    let mut individual = Vec::new();
    let mut windows_per_seed = None;
    for file in &files {
        ensure!(
            file.dataset("trajectories/0/x_m")?.ndim()
                == first.dataset("trajectories/0/x_m")?.ndim(),
            "Cannot mix 1D and 2D ensembles"
        );
        let config: rpic_isr::Config =
            serde_json::from_slice(&file.dataset("config_json")?.read_raw::<u8>()?)?;
        for r in 0..config.realizations {
            ensure!(
                seeds.insert(config.seed + r as u64),
                "Duplicate realization seed; refusing to double count"
            );
        }
        ensure!(
            close(&file.dataset("frequency_hz")?.read_raw::<f64>()?, &f),
            "Frequency axes differ"
        );
        ensure!(
            close(&file.dataset("theory_psd")?.read_raw::<f64>()?, &theory),
            "Physical parameters differ"
        );
        let n = file.attr("welch_segments")?.read_scalar::<u64>()? as f64;
        let per_seed = n / config.realizations as f64;
        ensure!(
            windows_per_seed.is_none_or(|n: f64| (n - per_seed).abs() < 1e-10),
            "Unequal segment counts per seed; cannot use equal-seed confidence intervals"
        );
        windows_per_seed = Some(per_seed);
        individual.extend(rpic_isr::analysis::read(file)?);
        for (a, b) in p.iter_mut().zip(file.dataset("psd")?.read_raw::<f64>()?) {
            *a += n * b;
        }
        count += n;
    }
    for a in &mut p {
        *a /= count;
    }
    let standard_error: Vec<_> = (0..p.len())
        .map(|j| {
            rpic_isr::analysis::mean_se(&individual.iter().map(|r| r.psd[j]).collect::<Vec<_>>()).1
        })
        .collect();
    let points = |y: &[f64]| {
        f.iter()
            .zip(y)
            .filter(|(x, _)| x.abs() <= 12000.)
            .map(|(x, y)| [*x * 1e-3, *y * 1e6])
            .collect()
    };
    let ymax = p
        .iter()
        .enumerate()
        .filter(|(j, _)| f[*j].abs() < 12000.)
        .map(|(j, y)| {
            (*y + if standard_error[j].is_finite() {
                1.96 * standard_error[j]
            } else {
                0.
            })
            .max(matched_theory[j])
        })
        .fold(0f64, f64::max)
        * 1.1e6;
    let mut fig = Figure::new(IntendedUse::Custom {
        width_mm: 170.,
        font_pt: 11.,
    });
    fig.height_mm = 103.;
    let mut a = Panel::new(
        Rect::new(25., 22., 136., 61.),
        Axes::new(
            [-12., 12.],
            [0., ymax],
            r"Doppler frequency [kHz]",
            r"$S(k,f)$ [$10^{-6}\,\mathrm{Hz}^{-1}$]",
        ),
    );
    if individual.len() > 1 {
        let mut band: Vec<_> = f
            .iter()
            .enumerate()
            .filter(|(_, x)| x.abs() <= 12000.)
            .map(|(j, x)| [*x * 1e-3, (p[j] + 1.96 * standard_error[j]) * 1e6])
            .collect();
        band.extend(
            f.iter()
                .enumerate()
                .rev()
                .filter(|(_, x)| x.abs() <= 12000.)
                .map(|(j, x)| [*x * 1e-3, (p[j] - 1.96 * standard_error[j]).max(0.) * 1e6]),
        );
        let mut fill = Line::new(band, "#b9d6e8");
        fill.width_pt = 0.;
        fill.fill = Some("#b9d6e8".into());
        fill.fill_opacity = 0.65;
        a.push(Layer::Line(fill));
    }
    a.push(Layer::Line(Line::new(points(&p), "#186e9c")));
    let mut l = Line::new(points(&matched_theory), "#c45128");
    l.dash = Some("4 2".into());
    a.push(Layer::Line(l));
    fig.panels.push(a);
    fig.labels.push(Label::new(
        r"$440.2\,\mathrm{MHz}:\ \mathrm{O}^{+}\ \mathrm{ion\ line}$",
        85.,
        7.,
    ));
    let mut l = Label::new(
        if individual.len() > 1 {
            r"PIC mean ($95\%$ interval)"
        } else {
            "PIC realization"
        },
        59.,
        16.,
    );
    l.color = "#186e9c".into();
    fig.labels.push(l);
    let mut l = Label::new("Maxwellian theory", 120., 16.);
    l.color = "#c45128".into();
    fig.labels.push(l);
    fig.save(Path::new(&args[0]).join("ion_line.png"))?;
    fig.save(Path::new(&args[0]).join("ion_line.svg"))?;
    let mut diag = Figure::new(IntendedUse::Custom {
        width_mm: 170.,
        font_pt: 11.,
    });
    diag.height_mm = 155.;
    let mut temp = Panel::new(
        Rect::new(30., 17., 131., 49.),
        Axes::new(
            [0., 10.],
            [-0.1, 1.],
            "Time [ms]",
            "Temperature change [percent]",
        ),
    );
    temp.axes.as_mut().unwrap().yticks = vec![0., 0.5, 1.];
    let mut energy = Panel::new(
        Rect::new(30., 96., 131., 39.),
        Axes::new([0., 10.], [-1., 2.], "Time [ms]", "Energy change [percent]"),
    );
    let mut tmax = 0f64;
    let mut emin = 0f64;
    let mut emax = 0f64;
    let mut tempmin = 0f64;
    let mut tempmax = 0f64;
    for file in &files {
        for name in file.group("diagnostics")?.member_names()? {
            let d = file
                .dataset(&format!("diagnostics/{name}"))?
                .read_raw::<f64>()?;
            let e0 = d[1];
            tmax = tmax.max(d[d.len() - 9] * 1e3);
            for (j, color) in [(5, "#186e9c"), (6, "#c45128")] {
                for row in d.as_chunks::<9>().0 {
                    let change = 100. * (row[j] / d[j] - 1.);
                    tempmin = tempmin.min(change);
                    tempmax = tempmax.max(change);
                }
                let mut l = Line::new(
                    d.as_chunks::<9>()
                        .0
                        .iter()
                        .map(|r| [r[0] * 1e3, 100. * (r[j] / d[j] - 1.)])
                        .collect(),
                    color,
                );
                l.opacity = 0.65;
                temp.push(Layer::Line(l));
            }
            let pts: Vec<_> = d
                .as_chunks::<9>()
                .0
                .iter()
                .map(|r| [r[0] * 1e3, 100. * (r[1] / e0 - 1.)])
                .collect();
            for q in &pts {
                emin = emin.min(q[1]);
                emax = emax.max(q[1]);
            }
            let mut l = Line::new(pts, "#245774");
            l.opacity = 0.65;
            energy.push(Layer::Line(l));
        }
    }
    temp.axes.as_mut().unwrap().xlim = [0., tmax];
    temp.axes.as_mut().unwrap().ylim = [tempmin - 0.1, tempmax + 0.1];
    temp.axes.as_mut().unwrap().yticks = vec![0., tempmax / 2., tempmax];
    temp.axes.as_mut().unwrap().yticklabels = Some(vec![
        "0".into(),
        format!("{:.2}", tempmax / 2.),
        format!("{tempmax:.2}"),
    ]);
    temp.axes.as_mut().unwrap().xticks = (0..=4).map(|i| tmax * i as f64 / 4.).collect();
    temp.axes.as_mut().unwrap().xticklabels = Some(
        (0..=4)
            .map(|i| format!("{:.2}", tmax * i as f64 / 4.))
            .collect(),
    );
    energy.axes.as_mut().unwrap().xlim = [0., tmax];
    energy.axes.as_mut().unwrap().xticks = temp.axes.as_ref().unwrap().xticks.clone();
    energy.axes.as_mut().unwrap().xticklabels = temp.axes.as_ref().unwrap().xticklabels.clone();
    let padding = 0.1 * (emax - emin).max(1e-4);
    energy.axes.as_mut().unwrap().ylim = [emin - padding, emax + padding];
    energy.axes.as_mut().unwrap().yticks = vec![emin, (emin + emax) / 2., emax];
    energy.axes.as_mut().unwrap().yticklabels = Some(vec![
        format!("{emin:.2}"),
        format!("{:.2}", (emin + emax) / 2.),
        format!("{emax:.2}"),
    ]);
    diag.panels.push(temp);
    diag.panels.push(energy);
    diag.labels.push(Label::new(
        "Electron (blue) and ion (orange) temperatures",
        87.,
        8.,
    ));
    diag.labels
        .push(Label::new("Total particle and field energy", 87., 85.));
    diag.save(Path::new(&args[0]).join("diagnostics.png"))?;
    diag.save(Path::new(&args[0]).join("diagnostics.svg"))?;
    let scales: serde_json::Value =
        serde_json::from_slice(&first.dataset("scales_json")?.read_raw::<u8>()?)?;
    let length = scales["length_m"].as_f64().unwrap();
    let d = first.dataset("diagnostics/0")?.read_raw::<f64>()?;
    let x = first.dataset("trajectories/0/x_m")?.read_raw::<f32>()?;
    let dimensions = if first.dataset("trajectories/0/x_m")?.ndim() == 3 {
        2
    } else {
        1
    };
    let mut trajectories = Figure::new(IntendedUse::Custom {
        width_mm: 170.,
        font_pt: 11.,
    });
    trajectories.height_mm = 100.;
    let mut panel = Panel::new(
        Rect::new(26., 16., 135., 64.),
        Axes::new([0., 1.], [0., length], "Time [ms]", "Periodic position [m]"),
    );
    for (j, color) in [
        (16, "#186e9c"),
        (17, "#c45128"),
        (18, "#437647"),
        (19, "#855c97"),
    ] {
        let mut points = Vec::new();
        let mut previous = None;
        for (r, row) in d.as_chunks::<9>().0.iter().enumerate() {
            if row[0] > 0.001 {
                break;
            }
            let value = x[dimensions * (32 * r + j)] as f64;
            if previous.is_some_and(|p: f64| (value - p).abs() > length / 2.) {
                points.push([f64::NAN, f64::NAN]);
            }
            points.push([row[0] * 1e3, value]);
            previous = Some(value);
        }
        panel.push(Layer::Line(Line::new(points, color)));
    }
    trajectories.panels.push(panel);
    trajectories
        .labels
        .push(Label::new(r"Four $\mathrm{O}^{+}$ trajectories", 87., 7.));
    trajectories.save(Path::new(&args[0]).join("trajectories.png"))?;
    trajectories.save(Path::new(&args[0]).join("trajectories.svg"))?;
    if dimensions == 2 {
        let mut plane = Figure::new(IntendedUse::Custom {
            width_mm: 170.,
            font_pt: 11.,
        });
        plane.height_mm = 108.;
        for (offset, particle, title) in [
            (0., 16, r"$\mathrm{O}^{+}$ trajectories"),
            (83., 0, "Electron trajectories"),
        ] {
            let mut panel = Panel::new(
                Rect::new(20. + offset, 19., 61., 61.),
                Axes::new([0., length], [0., length], "$x$ [m]", "$y$ [m]"),
            );
            let fast = particle == 0 && first.link_exists("trajectories/0/fast_x_m");
            let fast_x = if fast {
                first
                    .dataset("trajectories/0/fast_x_m")?
                    .read_raw::<f32>()?
            } else {
                Vec::new()
            };
            let interval = if fast {
                first
                    .group("trajectories/0")?
                    .attr("fast_interval_s")?
                    .read_scalar::<f64>()?
            } else {
                0.
            };
            for (j, color) in [
                (particle, "#186e9c"),
                (particle + 1, "#c45128"),
                (particle + 2, "#437647"),
            ] {
                let mut pts = Vec::new();
                let mut previous: Option<[f64; 2]> = None;
                let positions = if fast { &fast_x } else { &x };
                let frames = positions.len() / 64;
                for r in 0..frames {
                    let time = if fast { r as f64 * interval } else { d[9 * r] };
                    if time > if particle == 0 { 0.5e-6 } else { 0.001 } {
                        break;
                    }
                    let p = [
                        positions[2 * (32 * r + j)] as f64,
                        positions[2 * (32 * r + j) + 1] as f64,
                    ];
                    if previous.is_some_and(|q| {
                        (p[0] - q[0]).abs() > length / 2. || (p[1] - q[1]).abs() > length / 2.
                    }) {
                        pts.push([f64::NAN, f64::NAN]);
                    }
                    pts.push(p);
                    previous = Some(p);
                }
                panel.push(Layer::Line(Line::new(pts, color)));
            }
            let axes = panel.axes.as_mut().unwrap();
            axes.xticks = vec![0., length / 2., length];
            axes.yticks = axes.xticks.clone();
            axes.xticklabels = Some(axes.xticks.iter().map(|v| format!("{v:.2}")).collect());
            axes.yticklabels = axes.xticklabels.clone();
            plane.panels.push(panel);
            plane.labels.push(Label::new(title, 50. + offset, 8.));
            plane.labels.push(Label::new(
                if particle == 0 {
                    r"First $0.5\,\mu\mathrm{s}$"
                } else {
                    "First 1 ms"
                },
                50. + offset,
                101.,
            ));
        }
        plane.save(Path::new(&args[0]).join("trajectories_2d.png"))?;
        plane.save(Path::new(&args[0]).join("trajectories_2d.svg"))?;
    }
    // Derived scientific data and complete input configurations, not a CSV.
    let combined = hdf5::File::create(Path::new(&args[0]).join("ensemble.h5"))?;
    for (name, v) in [
        ("frequency_hz", &f),
        ("psd", &p),
        ("theory_psd", &theory),
        ("windowed_theory_psd", &matched_theory),
        ("psd_standard_error", &standard_error),
    ] {
        combined.new_dataset_builder().with_data(v).create(name)?;
    }
    combined
        .new_dataset_builder()
        .with_data(&individual.iter().map(|r| r.seed).collect::<Vec<_>>())
        .create("seeds")?;
    let per_seed: Vec<_> = individual
        .iter()
        .flat_map(|r| r.psd.iter().copied())
        .collect();
    combined
        .new_dataset::<f64>()
        .shape([individual.len(), p.len()])
        .create("psd_per_realization")?
        .write_raw(&per_seed)?;
    if individual.iter().all(|r| !r.early_psd.is_empty()) {
        for (name, values) in [
            (
                "early_psd_per_realization",
                individual
                    .iter()
                    .flat_map(|r| r.early_psd.iter().copied())
                    .collect::<Vec<_>>(),
            ),
            (
                "late_psd_per_realization",
                individual
                    .iter()
                    .flat_map(|r| r.late_psd.iter().copied())
                    .collect::<Vec<_>>(),
            ),
        ] {
            combined
                .new_dataset::<f64>()
                .shape([individual.len(), p.len()])
                .create(name)?
                .write_raw(&values)?;
        }
    }
    combined
        .new_attr::<u64>()
        .create("welch_segments")?
        .write_scalar(&(count as u64))?;
    combined
        .new_attr::<u64>()
        .create("independent_realizations")?
        .write_scalar(&(seeds.len() as u64))?;
    let provenance = combined.create_group("inputs")?;
    for (i, file) in files.iter().enumerate() {
        let g = provenance.create_group(&i.to_string())?;
        for name in ["config_json", "scales_json"] {
            g.new_dataset_builder()
                .with_data(&file.dataset(name)?.read_raw::<u8>()?)
                .create(name)?;
        }
        if file.link_exists("metadata_json") {
            g.new_dataset_builder()
                .with_data(&file.dataset("metadata_json")?.read_raw::<u8>()?)
                .create("metadata_json")?;
        }
    }
    let band: Vec<_> = f
        .iter()
        .enumerate()
        .filter(|(_, x)| x.abs() < 12000.)
        .map(|(i, _)| i)
        .collect();
    println!(
        "Ion-band integrated power PIC/theory = {:.4}",
        band.iter().map(|&i| p[i]).sum::<f64>() / band.iter().map(|&i| theory[i]).sum::<f64>()
    );
    println!("Averaged {count} overlapping Welch segments without fitting amplitude");
    Ok(())
}

fn close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.is_finite()
                && y.is_finite()
                && (x - y).abs() <= 1e-10 * (x.abs().max(y.abs()).max(1e-20))
        })
}
