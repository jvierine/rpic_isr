//! Matched sweep spectra: nearest native Welch bins on a common 50 kHz grid.
use anyhow::Result;
use rustplotlib::*;
fn main() -> Result<()> {
    let args:Vec<_>=std::env::args().collect();
    let h=hdf5::File::open(&args[1])?;
    let mut names=h.member_names()?; names.sort();
    let fps:Vec<f64>=names.iter().map(|n|h.group(n).unwrap().attr("plasma_frequency_hz").unwrap().read_scalar::<f64>().unwrap()*1e-6).collect();
    let mut xedges:Vec<f64>=fps.iter().map(|x|x-0.025).collect();
    xedges.push(fps.last().unwrap()+0.025);
    let yedges:Vec<f64>=(0..240).map(|j|-5.975+j as f64*0.05).collect();
    let mut fig=Figure::new(IntendedUse::Custom{width_mm:250.,font_pt:11.});
    fig.height_mm=175.;
    let cm=ColorMap::new(vec![[0.03,0.06,0.20],[0.10,0.25,0.45],[0.15,0.49,0.53],[0.52,0.72,0.43],[0.98,0.83,0.32],[1.,0.97,0.85]])?;
    let scale=fig.scale(Scale::new([-130.,-50.],Normalization::Linear,cm)?);
    for (col,(case,title)) in [("control","Thermal control"),("ring","1 percent suprathermal ring")].iter().enumerate() {
        let mut columns=Vec::new();
        for name in &names {
            let f=h.dataset(&format!("{name}/{case}/frequency_hz"))?.read_raw::<f64>()?;
            let p=h.dataset(&format!("{name}/{case}/psd"))?.read_raw::<f64>()?;
            let vals:Vec<f64>=(0..239).map(|j| {
                let target=(-5.95+j as f64*0.05)*1e6;
                let k=f.iter().enumerate().min_by(|(_,a),(_,b)|(*a-target).abs().total_cmp(&(*b-target).abs())).unwrap().0;
                10.*p[k].max(1e-30).log10()
            }).collect();
            columns.push(vals);
        }
        let mut values=Vec::new();
        for j in 0..239 {for column in &columns {values.push(column[j]);}}
        let x=25.+col as f64*100.;
        let mut panel=Panel::new(Rect::new(x,23.,83.,128.),Axes::new([xedges[0],*xedges.last().unwrap()],[-5.975,5.975],r"Plasma frequency $f_{pe}$ [MHz]","Doppler frequency [MHz]"));
        panel.push(Layer::Pcolormesh(Field::new(xedges.clone(),yedges.clone(),values,scale)?));
        fig.panels.push(panel);
        fig.labels.push(Label::new(*title,x+41.5,14.));
    }
    fig.colorbars.push(Colorbar::new(Rect::new(214.,23.,4.,128.),scale,r"PSD [dB re $1\,\mathrm{Hz}^{-1}$]",vec![-130.,-110.,-90.,-70.,-50.]));
    fig.labels.push(Label::new(r"Fixed $f_{ce}=1$ MHz; 30--120 $\mu$s; identical power scale",125.,170.));
    for ext in ["png","pdf"] {fig.save(format!("{}.{ext}",args[2]))?;}
    Ok(())
}
