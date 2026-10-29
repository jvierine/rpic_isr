//! Ring-only, time-resolved spectra; all panels share absolute PSD scaling.
use anyhow::Result;
use rustplotlib::*;
fn main()->Result<()> {
 let a:Vec<_>=std::env::args().collect();let h=hdf5::File::open(&a[1])?;
 let fp=h.dataset("plasma_frequency_hz")?.read_raw::<f64>()?;
 let mut x:Vec<f64>=fp.iter().map(|v|v*1e-6-0.025).collect();x.push(fp.last().unwrap()*1e-6+0.025);
 let y:Vec<f64>=(0..240).map(|j|-5.975+j as f64*0.05).collect();
 let mut fig=Figure::new(IntendedUse::Custom{width_mm:250.,font_pt:11.});fig.height_mm=285.;
 let cm=ColorMap::new(vec![[0.03,0.06,0.20],[0.10,0.25,0.45],[0.15,0.49,0.53],[0.52,0.72,0.43],[0.98,0.83,0.32],[1.,0.97,0.85]])?;
 let scale=fig.scale(Scale::new([-130.,-50.],Normalization::Linear,cm)?);
 for (i,(name,title)) in [("full",r"Full post-injection record: 30--1000 $\mu$s"),("early",r"Early: 30--250 $\mu$s"),("middle",r"Middle: 250--600 $\mu$s"),("late",r"Late: 600--1000 $\mu$s")].iter().enumerate(){
  let xx=25.+(i%2) as f64*99.;let yy=25.+(i/2) as f64*124.;
  let pwr=h.dataset(&format!("{name}/psd"))?.read_raw::<f64>()?;
  let mut p=Panel::new(Rect::new(xx,yy,82.,98.),Axes::new([x[0],*x.last().unwrap()],[-5.975,5.975],r"Plasma frequency $f_{pe}$ [MHz]","Doppler frequency [MHz]"));
  p.push(Layer::Pcolormesh(Field::new(x.clone(),y.clone(),pwr.iter().map(|p|10.*p.log10()).collect(),scale)?));
  for n in [-5.,-4.,-3.,-2.,-1.,1.,2.,3.,4.,5.] {let mut l=Line::new(vec![[x[0],n],[*x.last().unwrap(),n]],"#aaaaaa");l.width_pt=0.4;l.dash=Some("1 1.5".into());p.push(Layer::Line(l));}
  fig.panels.push(p);fig.labels.push(Label::new(*title,xx+41.,yy-5.));
 }
 fig.colorbars.push(Colorbar::new(Rect::new(212.,25.,4.,222.),scale,r"PSD [dB re $1\,\mathrm{Hz}^{-1}$]",vec![-130.,-110.,-90.,-70.,-50.]));
 fig.labels.push(Label::new(r"1 percent ring; fixed $f_{ce}=1$ MHz; dashed lines: gyroharmonics",125.,276.));
 for ext in ["png","pdf"]{fig.save(format!("{}.{ext}",a[2]))?;}Ok(())
}
