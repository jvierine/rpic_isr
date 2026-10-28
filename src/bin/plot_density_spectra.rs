//! Full two-sided spectra for completed density sweep rows, native Rustplotlib.
use anyhow::Result;
use rustplotlib::*;
fn main() -> Result<()> {
    let args:Vec<_> = std::env::args().collect();
    let input=hdf5::File::open(&args[1])?;
    let mut names=input.member_names()?; names.sort();
    let rows=(names.len()+2)/3;
    let mut fig=Figure::new(IntendedUse::Custom{width_mm:285.,font_pt:10.});
    fig.height_mm=25.+rows as f64*55.;
    for (s,x,c) in [("Thermal control",65.,"#416e86"),("1 percent suprathermal ring",190.,"#be5324")] {
        let mut l=Label::new(s,x,7.);l.color=c.into();fig.labels.push(l);
    }
    fig.labels.push(Label::new(r"Fixed $f_{ce}=1$ MHz; matched 30--120 $\mu$s records; 50 kHz bins",142.,15.));
    for (i,name) in names.iter().enumerate() {
        let x=22.+(i%3) as f64*94.; let y=30.+(i/3) as f64*55.;
        let fp=input.group(name)?.attr("plasma_frequency_hz")?.read_scalar::<f64>()?*1e-6;
        fig.labels.push(Label::new(format!(r"$f_{{pe}}={fp:.2}$ MHz"),x+34.,y-4.));
        let mut p=Panel::new(Rect::new(x,y,68.,34.),Axes::new([-6.,6.],[-140.,-30.],"Frequency [MHz]",r"PSD [dB/Hz]"));
        for (case,color) in [("control","#416e86"),("ring","#be5324")] {
            let f=input.dataset(&format!("{name}/{case}/frequency_hz"))?.read_raw::<f64>()?;
            let s=input.dataset(&format!("{name}/{case}/psd"))?.read_raw::<f64>()?;
            p.push(Layer::Line(Line::new(f.iter().zip(s).map(|(f,s)| [f*1e-6,10.*s.max(1e-30).log10()]).collect(),color)));
        }
        fig.panels.push(p);
    }
    for ext in ["png","pdf"] {fig.save(format!("{}.{ext}",args[2]))?;}
    Ok(())
}
