//! Native Rustplotlib plot of matched post-30-us Welch estimates from HDF5.
use anyhow::Result;
use rustplotlib::*;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let input = hdf5::File::open(&args[1])?;
    let mut fig = Figure::new(IntendedUse::Custom { width_mm: 200., font_pt: 11. });
    fig.height_mm = 160.;
    for (row, limits) in [[-5.,5.],[2.7,3.4]].iter().enumerate() {
        let mut panel = Panel::new(Rect::new(28.,25.+row as f64*68.,164.,45.),
            Axes::new(*limits,[-130.,-40.],"Frequency [MHz]",r"PSD [dB re $1\,\mathrm{Hz}^{-1}$]"));
        for (name,color) in [("control","#416e86"),("ring","#be5324")] {
            let f = input.dataset(&format!("{name}/frequency_hz"))?.read_raw::<f64>()?;
            let p = input.dataset(&format!("{name}/post30us_psd"))?.read_raw::<f64>()?;
            let mut points:Vec<[f64;2]> = f.iter().zip(&p).map(|(x,y)| [x*1e-6,10.*y.max(1e-30).log10()]).collect();
            points.sort_by(|a,b| a[0].total_cmp(&b[0]));
            panel.push(Layer::Line(Line::new(points,color)));
        }
        fig.panels.push(panel);
    }
    for (text,x,color) in [("Thermal control",70.,"#416e86"),("1 percent ring",139.,"#be5324")] {
        let mut l=Label::new(text,x,8.); l.color=color.into(); fig.labels.push(l);
    }
    fig.labels.push(Label::new(r"Same 30--120 $\mu$s interval; no separate peak normalization",100.,17.));
    for ext in ["png","pdf"] {fig.save(format!("{}.{ext}",args[2]))?;}
    Ok(())
}
