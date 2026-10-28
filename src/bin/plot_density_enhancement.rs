//! Ratio of matched, absolute ring/control Welch spectra. No peak normalization.
use anyhow::{ensure, Result};
use rustplotlib::*;
fn main() -> Result<()> {
    let args:Vec<_>=std::env::args().collect();
    let h=hdf5::File::open(&args[1])?;
    let mut names=h.member_names()?; names.sort();
    let mut fps=Vec::new(); let mut columns=Vec::new();
    for name in &names {
        fps.push(h.group(name)?.attr("plasma_frequency_hz")?.read_scalar::<f64>()?*1e-6);
        let f=h.dataset(&format!("{name}/ring/frequency_hz"))?.read_raw::<f64>()?;
        let fc=h.dataset(&format!("{name}/control/frequency_hz"))?.read_raw::<f64>()?;
        ensure!(f==fc,"Unmatched frequency bins");
        let r=h.dataset(&format!("{name}/ring/psd"))?.read_raw::<f64>()?;
        let c=h.dataset(&format!("{name}/control/psd"))?.read_raw::<f64>()?;
        ensure!(r.iter().chain(c.iter()).all(|v|v.is_finite()&&*v>0.));
        let vals:Vec<f64>=(0..239).map(|j| {
            let target=(-5.95+j as f64*0.05)*1e6;
            let k=f.iter().enumerate().min_by(|(_,a),(_,b)|(*a-target).abs().total_cmp(&(*b-target).abs())).unwrap().0;
            10.*(r[k]/c[k]).log10()
        }).collect();
        columns.push(vals);
    }
    let mut xe:Vec<f64>=fps.iter().map(|x|x-0.025).collect(); xe.push(fps.last().unwrap()+0.025);
    let ye:Vec<f64>=(0..240).map(|j|-5.975+j as f64*0.05).collect();
    let mut values=Vec::new(); for j in 0..239 {for col in &columns {values.push(col[j]);}}
    let out=hdf5::File::create(format!("{}.h5",args[2]))?;
    out.new_dataset_builder().with_data(&fps).create("plasma_frequency_mhz")?;
    out.new_dataset::<f64>().shape((239,fps.len())).create("enhancement_db")?.write_raw(&values)?;
    let freq:Vec<f64>=(0..239).map(|j|-5.95+j as f64*0.05).collect();
    out.new_dataset_builder().with_data(&freq).create("doppler_frequency_mhz")?;
    let mut fig=Figure::new(IntendedUse::Custom{width_mm:175.,font_pt:11.});fig.height_mm=175.;
    let cm=ColorMap::new(vec![[0.12,0.25,0.65],[1.,1.,0.88],[0.75,0.08,0.09]])?;
    let scale=fig.scale(Scale::new([-20.,20.],Normalization::Linear,cm)?);
    let mut p=Panel::new(Rect::new(25.,22.,112.,128.),Axes::new([xe[0],*xe.last().unwrap()],[-5.975,5.975],r"Plasma frequency $f_{pe}$ [MHz]","Doppler frequency [MHz]"));
    let xmin=xe[0]; let xmax=*xe.last().unwrap();
    p.push(Layer::Pcolormesh(Field::new(xe,ye,values,scale)?));
    for n in -5i32..=5 {
        if n==0 {continue;}
        let mut marker=Line::new(vec![[xmin,n as f64],[xmax,n as f64]],"#303030");
        marker.width_pt=if n.abs()==3 {1.1} else {0.6};
        marker.dash=Some("1.5 1.2".into());
        p.push(Layer::Line(marker));
        let mut label=Label::new(format!(r"${:+}f_{{ce}}$",n),130.,22.+(5.975-n as f64)/11.95*128.-1.5);
        label.background=Some("#ffffe0".into());
        fig.labels.push(label);
    }
    fig.panels.push(p);
    fig.colorbars.push(Colorbar::new(Rect::new(142.,22.,4.,128.),scale,r"$10\log_{10}(P_{\rm ring}/P_{\rm thermal})$ [dB]",vec![-20.,-10.,0.,10.,20.]));
    fig.labels.push(Label::new("1 percent suprathermal enhancement",87.5,12.));
    fig.labels.push(Label::new(r"Fixed $f_{ce}=1$ MHz; matched 30--120 $\mu$s",87.5,169.));
    for ext in ["png","pdf"] {fig.save(format!("{}.{ext}",args[2]))?;}
    Ok(())
}
