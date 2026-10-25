//! Separate executable: 1D results and their CLI remain unchanged.
use anyhow::{ensure, Result};
use clap::Parser;
use rpic_isr::*;
use std::path::Path;
fn main() -> Result<()> {
    let c = Config::parse();
    let s = Scales::new(&c)?;
    ensure!(
        !Path::new(&c.output).exists(),
        "Output exists: {}",
        c.output
    );
    ensure!(
        c.cells <= 512
            && c.cells
                .checked_mul(c.cells)
                .and_then(|n| n.checked_mul(c.ppc))
                .and_then(|n| n.checked_mul(c.gpu_batch_size))
                .is_some_and(|n| n <= i32::MAX as usize / 4),
        "2D allocation/index limit exceeded"
    );
    println!(
        "2D2V: {} x {} cells; {} particles/species/seed; {}",
        c.cells,
        c.cells,
        c.cells * c.cells * c.ppc,
        serde_json::to_string(&s)?
    );
    let mut records = Vec::new();
    let mut runs = Vec::new();
    #[cfg(feature = "cuda")]
    let mut pending = std::collections::VecDeque::new();
    for r in 0..c.realizations {
        let seed = c.seed + r as u64;
        let run = match c.backend {
            Backend::Cpu => pic2d::simulate(&c, &s, seed)?,
            Backend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    if pending.is_empty() {
                        let seeds: Vec<_> = (r..(r + c.gpu_batch_size).min(c.realizations))
                            .map(|j| c.seed + j as u64)
                            .collect();
                        pending = gpu2d::simulate(&c, &s, &seeds)?.into();
                    }
                    pending.pop_front().unwrap()
                }
                #[cfg(not(feature = "cuda"))]
                {
                    anyhow::bail!("Rebuild with --features cuda");
                }
            }
        };
        let filtered = spectrum::decimate(&run.voltage, s.sample_stride);
        println!(
            "seed {seed}: {:.2}s, {} samples",
            run.elapsed_s,
            filtered.len()
        );
        if let Some(dir) = &c.checkpoint_dir {
            let mut single = c.clone();
            single.seed = seed;
            single.realizations = 1;
            single.checkpoint_dir = None;
            single.output = Path::new(dir)
                .join(format!("seed{seed}.h5"))
                .to_string_lossy()
                .into_owned();
            output2d::save(
                &single,
                &s,
                std::slice::from_ref(&filtered),
                std::slice::from_ref(&run),
                2,
            )?;
        }
        records.push(filtered);
        runs.push(run);
    }
    output2d::save(&c, &s, &records, &runs, 2)
}
