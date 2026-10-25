//! Disjoint periodic PIC copies share launches and batched cuFFT plans.
//! This changes scheduling only, not particle weights or physical equations.
use crate::*;
use cudarc::{
    cufft::{
        sys::{cufftType, float2},
        CudaFft,
    },
    driver::*,
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::time::{Duration, Instant};

pub fn simulate(c: &Config, s: &Scales, seeds: &[u64]) -> Result<Vec<Run>> {
    if seeds.is_empty() {
        bail!("No seeds");
    }
    let timer = Instant::now();
    let ctx = CudaContext::new(c.device)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(compile_ptx_with_opts(
        include_str!("kernels_batched.cu"),
        CompileOptions {
            arch: Some("compute_61"),
            fmad: Some(false),
            ..Default::default()
        },
    )?)?;
    let clear = module.load_function("clear_grid")?;
    let deposit = module.load_function("deposit")?;
    let poisson = module.load_function("poisson")?;
    let push = module.load_function("push")?;
    let mut initial = Particles {
        xe: Vec::new(),
        ve: Vec::new(),
        xi: Vec::new(),
        vi: Vec::new(),
    };
    for &seed in seeds {
        let p = Particles::new(c, s, seed);
        initial.xe.extend(p.xe);
        initial.ve.extend(p.ve);
        initial.xi.extend(p.xi);
        initial.vi.extend(p.vi);
    }
    let n = (c.cells * c.ppc) as i32;
    let ng = c.cells as i32;
    let nf = ng / 2 + 1;
    let batches = seeds.len() as i32;
    let length = s.length as f32;
    let dt = c.dt as f32;
    let mu = s.mass_ratio as f32;
    let mut xe = stream.clone_htod(&initial.xe)?;
    let mut ve = stream.clone_htod(&initial.ve)?;
    let mut xi = stream.clone_htod(&initial.xi)?;
    let mut vi = stream.clone_htod(&initial.vi)?;
    let mut grid = stream.alloc_zeros::<f32>((2 * ng * batches) as usize)?;
    let mut freq = stream.alloc_zeros::<float2>((2 * nf * batches) as usize)?;
    let mut efreq = stream.alloc_zeros::<float2>((nf * batches) as usize)?;
    let mut e = stream.alloc_zeros::<f32>((ng * batches) as usize)?;
    let mut obs = stream.alloc_zeros::<float2>(s.steps * seeds.len())?;
    let forward = CudaFft::plan_1d(ng, cufftType::CUFFT_R2C, 2 * batches, stream.clone())?;
    let inverse = CudaFft::plan_1d(ng, cufftType::CUFFT_C2R, batches, stream.clone())?;
    let cfgp = LaunchConfig::for_num_elems((n * batches) as u32);
    let cfgdep = LaunchConfig {
        grid_dim: (((n + 255) / 256) as u32, batches as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: (2 * ng * 4) as u32,
    };
    let cfgg = LaunchConfig::for_num_elems((2 * ng * batches) as u32);
    let cfgf = LaunchConfig::for_num_elems((nf * batches) as u32);
    let mode = c.bragg_mode as i32;
    let kd = std::f64::consts::PI * c.bragg_mode as f64 / c.cells as f64;
    let norm = ((n as f64).sqrt() / ng as f64 / (kd.sin() / kd).powi(2)) as f32;
    let mut out: Vec<Run> = (0..seeds.len()).map(|_| Run::default()).collect();
    let steps = s.steps as i32;
    for step in 0..=s.burn_steps + s.steps {
        if step > 0 && step % 100000 == 0 {
            eprintln!(
                "CUDA batch of {}: {:.3}/{:.3} ms",
                seeds.len(),
                step as f64 * s.dt_s * 1e3,
                (s.burn_steps + s.steps) as f64 * s.dt_s * 1e3
            );
        }
        let count = 2 * ng * batches;
        unsafe {
            stream
                .launch_builder(&clear)
                .arg(&mut grid)
                .arg(&count)
                .launch(cfgg)?;
        }
        unsafe {
            stream
                .launch_builder(&deposit)
                .arg(&xe)
                .arg(&xi)
                .arg(&mut grid)
                .arg(&n)
                .arg(&ng)
                .arg(&batches)
                .arg(&length)
                .launch(cfgdep)?;
        }
        forward.exec_r2c(&grid, &mut freq)?;
        let record = if step >= s.burn_steps && step < s.burn_steps + s.steps {
            (step - s.burn_steps) as i32
        } else {
            -1
        };
        unsafe {
            stream
                .launch_builder(&poisson)
                .arg(&freq)
                .arg(&mut efreq)
                .arg(&mut obs)
                .arg(&ng)
                .arg(&batches)
                .arg(&length)
                .arg(&mode)
                .arg(&norm)
                .arg(&record)
                .arg(&steps)
                .launch(cfgf)?;
        }
        inverse.exec_c2r(&mut efreq, &mut e)?;
        if step == 0 {
            let kick = -0.5 * dt;
            let drift = 0f32;
            unsafe {
                stream
                    .launch_builder(&push)
                    .arg(&mut xe)
                    .arg(&mut ve)
                    .arg(&mut xi)
                    .arg(&mut vi)
                    .arg(&e)
                    .arg(&n)
                    .arg(&ng)
                    .arg(&batches)
                    .arg(&length)
                    .arg(&mu)
                    .arg(&kick)
                    .arg(&drift)
                    .launch(cfgp)?;
            }
        }
        if step % c.diagnostic_stride == 0 || step == s.burn_steps + s.steps {
            let host = Particles {
                xe: stream.clone_dtoh(&xe)?,
                ve: stream.clone_dtoh(&ve)?,
                xi: stream.clone_dtoh(&xi)?,
                vi: stream.clone_dtoh(&vi)?,
            };
            let field = stream.clone_dtoh(&e)?;
            let charge = stream.clone_dtoh(&grid)?;
            for (b, run) in out.iter_mut().enumerate() {
                let lo = b * n as usize;
                let hi = lo + n as usize;
                let p = Particles {
                    xe: host.xe[lo..hi].to_vec(),
                    ve: host.ve[lo..hi].to_vec(),
                    xi: host.xi[lo..hi].to_vec(),
                    vi: host.vi[lo..hi].to_vec(),
                };
                diagnostic(run, &p, &field[b * c.cells..(b + 1) * c.cells], s, c, step);
                *run.diagnostics.last_mut().unwrap() = charge
                    [(2 * b + 1) * c.cells..(2 * b + 2) * c.cells]
                    .iter()
                    .map(|v| *v as f64)
                    .sum::<f64>()
                    / c.cells as f64;
            }
            if c.throttle_ms > 0 {
                std::thread::sleep(Duration::from_millis(c.throttle_ms));
            }
        }
        if step == s.burn_steps + s.steps {
            break;
        }
        unsafe {
            stream
                .launch_builder(&push)
                .arg(&mut xe)
                .arg(&mut ve)
                .arg(&mut xi)
                .arg(&mut vi)
                .arg(&e)
                .arg(&n)
                .arg(&ng)
                .arg(&batches)
                .arg(&length)
                .arg(&mu)
                .arg(&dt)
                .arg(&dt)
                .launch(cfgp)?;
        }
    }
    let voltage = stream.clone_dtoh(&obs)?;
    for (b, run) in out.iter_mut().enumerate() {
        run.voltage = voltage[b * s.steps..(b + 1) * s.steps]
            .iter()
            .map(|z| Complex32::new(z.x, z.y))
            .collect();
        run.elapsed_s = timer.elapsed().as_secs_f64();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "Requires a Pascal-or-newer CUDA 12 GPU and NVRTC/cuFFT"]
    fn independent_batches_match_cpu_initial_trajectory() {
        for (cells, dt) in [(64, 0.12), (128, 0.06)] {
            let mut c = Config::parse_from(["test"]);
            c.cells = cells;
            c.dt = dt;
            c.bragg_mode = 2;
            c.ppc = 128;
            c.duration_ms = 0.1;
            c.burn_ms = 0.;
            c.sample_us = 1.;
            c.window = 16;
            let s = Scales::new(&c).unwrap();
            let seeds = [42, 43, 44, 45];
            let gpu = simulate(&c, &s, &seeds).unwrap();
            for (b, seed) in seeds.iter().enumerate() {
                let cpu = cpu::simulate(&c, &s, *seed).unwrap();
                for j in 0..20 {
                    assert!(
                        (cpu.voltage[j] - gpu[b].voltage[j]).norm() < 5e-5,
                        "seed {seed}, step {j}"
                    );
                }
                assert!(gpu[b]
                    .diagnostics
                    .as_chunks::<9>()
                    .0
                    .iter()
                    .all(|r| r[8].abs() < 2e-6));
            }
        }
    }
}
