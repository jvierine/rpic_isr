use anyhow::{ensure,Result};
use clap::Parser;
use rpic_isr::{pic3v::Setup,*};
fn main()->Result<()> {
 let c=Setup::parse();c.validate()?;let s=Scales::new(&c)?;
 ensure!(!std::path::Path::new(&c.output).exists(),"Output exists");
 ensure!(c.cells<=512 && c.cells*c.cells*c.ppc<=i32::MAX as usize/4,"Allocation limit");
 ensure!(c.realizations==1,"Use separate process/output for independent paired realizations");
 println!("2D3V {}",serde_json::to_string(&c)?);
 let run=match c.backend {
 Backend::Cpu=>pic3v::simulate(&c,&s,c.seed)?,
 Backend::Cuda=>{
 #[cfg(feature="cuda")] {gpu3v::simulate(&c,&s,&[c.seed])?.remove(0)}
 #[cfg(not(feature="cuda"))] {anyhow::bail!("Build with --features cuda")}
 }};
 let filtered=spectrum::decimate(&run.voltage,s.sample_stride);
 println!("Completed {:.3}s",run.elapsed_s);
 output3v::save(&c,&s,&[filtered],&[run],2)
}
