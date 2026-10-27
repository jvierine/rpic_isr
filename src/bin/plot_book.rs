//! Native Rustplotlib figure from uncompressed HDF5 PIC data; no synthetic paths.
use anyhow::Result;
use rustplotlib::*;
fn line(p: &mut Panel, pts: Vec<[f64; 2]>, color: &str, width: f64, alpha: f64, dashed: bool) {
    let mut l = Line::new(pts, color);
    l.width_pt = width;
    l.opacity = alpha;
    if dashed {
        l.dash = Some("0.7 0.5".into());
    }
    p.push(Layer::Line(l));
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let h = hdf5::File::open(&args[1])?;
    let attr = |s: &str| h.attr(s).unwrap().read_scalar::<f64>().unwrap();
    let tr = h.dataset("tracks")?;
    let shape = tr.shape();
    let nt = shape[0];
    let np = shape[1];
    let tracks = tr.read_raw::<f64>()?;
    let v = h.dataset("velocity_m_s")?.read_raw::<f64>()?;
    let species = h.dataset("species")?.read_raw::<i64>()?;
    let phi = h.dataset("potential_mV")?;
    let g = phi.shape()[0];
    let field = phi.read_raw::<f64>()?;
    let pos = |t: usize, j: usize| [tracks[(t * np + j) * 2], tracks[(t * np + j) * 2 + 1]];
    let mut f = Figure::new(IntendedUse::Custom {
        width_mm: 150.,
        font_pt: 11.,
    });
    f.height_mm = 166.;
    let cmap = ColorMap::new(vec![[0.22, 0.29, 0.70], [1., 1., 0.86], [0.75, 0.17, 0.15]])?;
    let scale = f.scale(Scale::new([-20., 20.], Normalization::Linear, cmap)?);
    let mut p = Panel::new(
        Rect::new(20., 16., 108., 108.),
        Axes::new(
            [-5., 5.],
            [-5., 5.],
            r"$x/\lambda_{\mathrm{De}}$",
            r"$y/\lambda_{\mathrm{De}}$",
        ),
    );
    let dx = attr("grid_spacing");
    let start = attr("crop_start") * dx - attr("length_lambda_de") / 2.;
    let grid: Vec<f64> = (0..=g).map(|i| start + (i as f64 - 0.5) * dx).collect();
    p.push(Layer::Pcolormesh(Field::new(
        grid.clone(),
        grid,
        field,
        scale,
    )?));
    let colors = ["#005d91", "#934108"];
    let ve = (1.380649e-23 * attr("Te_K") / 9.1093837e-31).sqrt();
    let ion_factor = (attr("Te_K") / attr("Ti_K") * attr("mass_ratio")).sqrt();
    for j in 0..np {
        let s = species[j] as usize;
        let col = colors[s];
        for t in (0..nt - 1).step_by(4) {
            let end = (t + 4).min(nt - 1);
            let a = 0.08 + 0.47 * (t as f64 / (nt - 1) as f64).powi(2);
            line(
                &mut p,
                (t..=end).map(|k| pos(k, j)).collect(),
                col,
                0.85,
                a,
                true,
            );
        }
        let a = pos(nt - 1, j);
        let mul = if s == 0 { 1. } else { ion_factor };
        let d = [v[2 * j] / ve * mul * 0.42, v[2 * j + 1] / ve * mul * 0.42];
        let b = [a[0] + d[0], a[1] + d[1]];
        line(&mut p, vec![a, b], col, 1., 1., false);
        let len = d[0].hypot(d[1]);
        if len > 0. {
            let u = [d[0] / len, d[1] / len];
            line(
                &mut p,
                vec![
                    [
                        b[0] - 0.12 * u[0] + 0.055 * u[1],
                        b[1] - 0.12 * u[1] - 0.055 * u[0],
                    ],
                    b,
                    [
                        b[0] - 0.12 * u[0] - 0.055 * u[1],
                        b[1] - 0.12 * u[1] + 0.055 * u[0],
                    ],
                ],
                col,
                1.,
                1.,
                false,
            );
        }
        let circle: Vec<_> = (0..=24)
            .map(|i| {
                let th = i as f64 / 24. * std::f64::consts::TAU;
                [a[0] + 0.047 * th.cos(), a[1] + 0.047 * th.sin()]
            })
            .collect();
        let mut mark = Line::new(circle, col);
        mark.fill = Some(col.into());
        mark.width_pt = 0.5;
        p.push(Layer::Line(mark));
    }
    f.panels.push(p);
    f.colorbars.push(Colorbar::new(
        Rect::new(129., 16., 3., 108.),
        scale,
        r"$\phi\ [\mathrm{mV}]$",
        vec![-20., -10., 0., 10., 20.],
    ));
    let mut e = Label::new(r"Electrons: $\mathbf{v}_e$", 42., 145.);
    e.color = colors[0].into();
    f.labels.push(e);
    let mut i = Label::new(r"O$^+$ ions: $242\,\mathbf{v}_i$", 103., 145.);
    i.color = colors[1].into();
    f.labels.push(i);
    f.labels.push(Label::new(
        r"Dashed trails: preceding $2.15\,\mu\mathrm{s}$",
        75.,
        155.,
    ));
    for ext in ["pdf", "png", "svg"] {
        f.save(format!("{}.{}", args[2], ext))?;
    }
    Ok(())
}
