//! Native Rustplotlib comparison of actual matched explicit/implicit records.
use anyhow::{ensure, Result};
use rustplotlib::*;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "Usage: plot_implicit OUTPUT_DIR EXPLICIT.h5 IMPLICIT.h5"
    );
    std::fs::create_dir_all(&args[0])?;
    let explicit = hdf5::File::open(&args[1])?;
    let implicit = hdf5::File::open(&args[2])?;
    let e = explicit.dataset("diagnostics/0")?.read_raw::<f64>()?;
    let i = implicit.dataset("diagnostics")?.read_raw::<f64>()?;
    let epoints: Vec<_> = e
        .as_chunks::<9>()
        .0
        .iter()
        .map(|r| [r[0] * 1e3, 100. * (r[1] / e[1] - 1.)])
        .collect();
    let ipoints: Vec<_> = i
        .as_chunks::<12>()
        .0
        .iter()
        .map(|r| [r[0] * 1e3, 100. * (r[1] / i[1] - 1.)])
        .collect();
    let ppm: Vec<_> = ipoints.iter().map(|p| [p[0], p[1] * 1e4]).collect();
    let tmax = epoints.last().unwrap()[0].max(ipoints.last().unwrap()[0]);
    let emin = epoints.iter().map(|p| p[1]).fold(0f64, f64::min);
    let emax = epoints.iter().map(|p| p[1]).fold(0f64, f64::max);
    let imin = ppm.iter().map(|p| p[1]).fold(0f64, f64::min);
    let imax = ppm.iter().map(|p| p[1]).fold(0f64, f64::max);
    let span = (imax - imin).max(0.01);
    let mut f = Figure::new(IntendedUse::Custom {
        width_mm: 170.,
        font_pt: 11.,
    });
    f.height_mm = 161.;
    let mut top = Panel::new(
        Rect::new(28., 24., 133., 43.),
        Axes::new(
            [0., tmax],
            [emin - 0.01 * (emax - emin), emax * 1.1],
            "Time [ms]",
            "Energy change [percent]",
        ),
    );
    top.push(Layer::Line(Line::new(epoints, "#186e9c")));
    top.push(Layer::Line(Line::new(ipoints, "#c45128")));
    let mut bottom = Panel::new(
        Rect::new(28., 103., 133., 37.),
        Axes::new(
            [0., tmax],
            [imin - 0.1 * span, imax + 0.1 * span],
            "Time [ms]",
            "Energy change [ppm]",
        ),
    );
    bottom.push(Layer::Line(Line::new(ppm, "#c45128")));
    for panel in [&mut top, &mut bottom] {
        let a = panel.axes.as_mut().unwrap();
        a.xticks = (0..=4).map(|j| tmax * j as f64 / 4.).collect();
        a.xticklabels = Some(a.xticks.iter().map(|v| format!("{v:.2}")).collect());
    }
    bottom.axes.as_mut().unwrap().yticks = vec![imin, (imin + imax) * 0.5, imax];
    bottom.axes.as_mut().unwrap().yticklabels = Some(vec![
        format!("{imin:.2}"),
        format!("{:.2}", (imin + imax) * 0.5),
        format!("{imax:.2}"),
    ]);
    f.panels.push(top);
    f.panels.push(bottom);
    f.labels.push(Label::new(
        "Matched 2D particle-in-cell energy test",
        85.,
        7.,
    ));
    let mut label = Label::new("Explicit", 63., 17.);
    label.color = "#186e9c".into();
    f.labels.push(label);
    let mut label = Label::new("Implicit", 111., 17.);
    label.color = "#c45128".into();
    f.labels.push(label);
    f.labels.push(Label::new(
        "Implicit result, expanded vertical scale",
        85.,
        94.,
    ));
    for ext in ["png", "svg"] {
        f.save(std::path::Path::new(&args[0]).join(format!("energy_comparison.{ext}")))?;
    }
    Ok(())
}
