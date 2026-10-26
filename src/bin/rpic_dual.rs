use anyhow::Result;
use clap::Parser;
use rpic_isr::{control2d::Controls, *};
#[derive(Parser)]
#[command(
    about = "One float32 PIC plasma on one/two GPUs, with live energy accounting and optional delayed thermostat"
)]
struct Options {
    #[command(flatten)]
    simulation: Config,
    #[command(flatten)]
    controls: Controls,
}
fn main() -> Result<()> {
    let o = Options::parse();
    #[cfg(feature = "cuda")]
    {
        let live = o
            .controls
            .live_diagnostics
            .clone()
            .unwrap_or_else(|| format!("{}.live.h5", o.simulation.output));
        let (c, s, run) = dual_gpu2d::simulate(&o.simulation, o.controls)?;
        let record = spectrum::decimate(&run.voltage, s.sample_stride);
        output2d::save(&c, &s, &[record], &[run], 2)?;
        let temporary = format!("{live}.complete.tmp");
        std::fs::copy(&live, &temporary)?;
        {
            let f = hdf5::File::open_rw(&temporary)?;
            f.unlink("status")?;
            f.new_dataset_builder()
                .with_data(b"complete".as_slice())
                .create("status")?;
            f.flush()?;
        }
        std::fs::rename(temporary, live)?;
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = o;
        anyhow::bail!("Build with --features cuda")
    }
}
