//! GPU-resident tight loop; no particle arrays cross PCIe except diagnostics.
//! Kernel compilation uses NVRTC, so nvcc is not required on the execution host.
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

pub fn simulate(c: &Config, s: &Scales, seed: u64) -> Result<Run> {
    let timer = Instant::now();
    let ctx = CudaContext::new(c.device)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(compile_ptx_with_opts(
        include_str!("kernels.cu"),
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
    let initial = Particles::new(c, s, seed);
    let n = initial.xe.len() as i32;
    let ng = c.cells as i32;
    let nf = ng / 2 + 1;
    let length = s.length as f32;
    let dt = c.dt as f32;
    let mu = s.mass_ratio as f32;
    let mut xe = stream.clone_htod(&initial.xe)?;
    let mut ve = stream.clone_htod(&initial.ve)?;
    let mut xi = stream.clone_htod(&initial.xi)?;
    let mut vi = stream.clone_htod(&initial.vi)?;
    let mut grid = stream.alloc_zeros::<f32>(2 * c.cells)?;
    let mut freq = stream.alloc_zeros::<float2>(2 * nf as usize)?;
    let mut efreq = stream.alloc_zeros::<float2>(nf as usize)?;
    let mut e = stream.alloc_zeros::<f32>(c.cells)?;
    let mut obs = stream.alloc_zeros::<float2>(s.steps)?;
    let forward = CudaFft::plan_1d(ng, cufftType::CUFFT_R2C, 2, stream.clone())?;
    let inverse = CudaFft::plan_1d(ng, cufftType::CUFFT_C2R, 1, stream.clone())?;
    let cfgp = LaunchConfig::for_num_elems(n as u32);
    let cfgg = LaunchConfig::for_num_elems((2 * ng) as u32);
    let cfgf = LaunchConfig::for_num_elems(nf as u32);
    let mode = c.bragg_mode as i32;
    let kd = std::f64::consts::PI * c.bragg_mode as f64 / c.cells as f64;
    let norm = ((n as f64).sqrt() / ng as f64 / (kd.sin() / kd).powi(2)) as f32;
    let mut out = Run::default();
    // A field solve at x(0), then initialize velocities at t=-dt/2.
    for step in 0..=s.burn_steps + s.steps {
        unsafe {
            stream
                .launch_builder(&clear)
                .arg(&mut grid)
                .arg(&ng)
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
                .arg(&length)
                .launch(cfgp)?;
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
                .arg(&length)
                .arg(&mode)
                .arg(&norm)
                .arg(&record)
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
                    .arg(&length)
                    .arg(&mu)
                    .arg(&kick)
                    .arg(&drift)
                    .launch(cfgp)?;
            }
        }
        if step % c.diagnostic_stride == 0 || step == s.burn_steps + s.steps {
            let p = Particles {
                xe: stream.clone_dtoh(&xe)?,
                ve: stream.clone_dtoh(&ve)?,
                xi: stream.clone_dtoh(&xi)?,
                vi: stream.clone_dtoh(&vi)?,
            };
            let field = stream.clone_dtoh(&e)?;
            diagnostic(&mut out, &p, &field, s, c, step);
            let charge = stream.clone_dtoh(&grid)?;
            *out.diagnostics.last_mut().unwrap() =
                charge[c.cells..].iter().map(|x| *x as f64).sum::<f64>() / c.cells as f64;
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
                .arg(&length)
                .arg(&mu)
                .arg(&dt)
                .arg(&dt)
                .launch(cfgp)?;
        }
    }
    out.voltage = stream
        .clone_dtoh(&obs)?
        .iter()
        .map(|z| Complex32::new(z.x, z.y))
        .collect();
    out.elapsed_s = timer.elapsed().as_secs_f64();
    Ok(out)
}
