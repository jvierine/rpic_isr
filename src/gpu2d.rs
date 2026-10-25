//! Batched 2D cuFFT Poisson solves and CUDA particle push; float32 throughout.
use crate::{pic2d::*, *};
use cudarc::{
    cufft::{
        sys::{cufftType, float2},
        CudaFft, FftDirection,
    },
    driver::*,
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::time::{Duration, Instant};
pub fn simulate(c: &Config, s: &Scales, seeds: &[u64]) -> Result<Vec<Run>> {
    let timer = Instant::now();
    let ctx = CudaContext::new(c.device)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(compile_ptx_with_opts(
        include_str!("kernels_2d.cu"),
        CompileOptions {
            arch: Some("compute_61"),
            fmad: Some(false),
            ..Default::default()
        },
    )?)?;
    let clear = module.load_function("clear2")?;
    let deposit = module.load_function("deposit2")?;
    let poisson = module.load_function("poisson2")?;
    let push = module.load_function("push2")?;
    let trace = module.load_function("trace2")?;
    let mut initial = Particles2::default();
    for &seed in seeds {
        let p = Particles2::new(c, s, seed);
        initial.xe.extend(p.xe);
        initial.ve.extend(p.ve);
        initial.xi.extend(p.xi);
        initial.vi.extend(p.vi);
    }
    let pairs = |v: &[[f32; 2]]| {
        v.iter()
            .map(|v| float2 { x: v[0], y: v[1] })
            .collect::<Vec<_>>()
    };
    let n = (c.cells * c.cells * c.ppc) as i32;
    let ng = c.cells as i32;
    let nn = ng * ng;
    let batches = seeds.len() as i32;
    let length = s.length as f32;
    let dt = c.dt as f32;
    let mu = s.mass_ratio as f32;
    let mut xe = stream.clone_htod(&pairs(&initial.xe))?;
    let mut ve = stream.clone_htod(&pairs(&initial.ve))?;
    let mut xi = stream.clone_htod(&pairs(&initial.xi))?;
    let mut vi = stream.clone_htod(&pairs(&initial.vi))?;
    drop(initial);
    let mut grid = stream.alloc_zeros::<float2>((2 * nn * batches) as usize)?;
    let mut freq = stream.alloc_zeros::<float2>((2 * nn * batches) as usize)?;
    let mut efreq = stream.alloc_zeros::<float2>((2 * nn * batches) as usize)?;
    let mut e = stream.alloc_zeros::<float2>((2 * nn * batches) as usize)?;
    let mut obs = stream.alloc_zeros::<float2>(s.steps * seeds.len())?;
    let frames = (s.burn_steps + s.steps + 1).min(2048) as i32;
    let mut tracks = stream.alloc_zeros::<float2>(frames as usize * 32 * seeds.len())?;
    let fft = CudaFft::plan_many(
        &[ng, ng],
        None,
        1,
        nn,
        None,
        1,
        nn,
        cufftType::CUFFT_C2C,
        2 * batches,
        stream.clone(),
    )?;
    let cfgp = LaunchConfig::for_num_elems((n * batches) as u32);
    let cfgg = LaunchConfig::for_num_elems((2 * nn * batches) as u32);
    let cfgf = LaunchConfig::for_num_elems((nn * batches) as u32);
    let mode = c.bragg_mode as i32;
    let kd = std::f64::consts::PI * c.bragg_mode as f64 / c.cells as f64;
    let norm = ((n as f64).sqrt() / nn as f64 / (kd.sin() / kd).powi(3)) as f32;
    let steps = s.steps as i32;
    let mut out: Vec<_> = seeds.iter().map(|_| Run::default()).collect();
    for step in 0..=s.burn_steps + s.steps {
        if step < frames as usize {
            let j = step as i32;
            let lambda = s.lambda_de as f32;
            unsafe {
                stream
                    .launch_builder(&trace)
                    .arg(&xe)
                    .arg(&xi)
                    .arg(&mut tracks)
                    .arg(&n)
                    .arg(&batches)
                    .arg(&j)
                    .arg(&frames)
                    .arg(&lambda)
                    .launch(LaunchConfig::for_num_elems(32 * batches as u32))?;
            }
        }
        if step > 0 && step % 100000 == 0 {
            eprintln!(
                "2D CUDA device {} batch {}: {:.3}/{:.3} ms, wall {:.1}s",
                c.device,
                batches,
                step as f64 * s.dt_s * 1e3,
                (s.burn_steps + s.steps) as f64 * s.dt_s * 1e3,
                timer.elapsed().as_secs_f64()
            );
        }
        let count = 2 * nn * batches;
        unsafe {
            stream
                .launch_builder(&clear)
                .arg(&mut grid)
                .arg(&count)
                .launch(cfgg)?;
            stream
                .launch_builder(&deposit)
                .arg(&xe)
                .arg(&xi)
                .arg(&mut grid)
                .arg(&n)
                .arg(&ng)
                .arg(&batches)
                .arg(&length)
                .launch(cfgp)?;
        }
        fft.exec_c2c(&mut grid, &mut freq, FftDirection::Forward)?;
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
        fft.exec_c2c(&mut efreq, &mut e, FftDirection::Inverse)?;
        let mut do_push = |kick: f32, drift: f32| -> Result<()> {
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
            Ok(())
        };
        if step == 0 {
            do_push(-0.5 * dt, 0.)?;
        }
        // End the mutable borrow before copying diagnostics.
        drop(do_push);
        if step % c.diagnostic_stride == 0 || step == s.burn_steps + s.steps {
            let unpack = |v: Vec<float2>| v.into_iter().map(|v| [v.x, v.y]).collect::<Vec<_>>();
            let host = Particles2 {
                xe: unpack(stream.clone_dtoh(&xe)?),
                ve: unpack(stream.clone_dtoh(&ve)?),
                xi: unpack(stream.clone_dtoh(&xi)?),
                vi: unpack(stream.clone_dtoh(&vi)?),
            };
            let field = stream.clone_dtoh(&e)?;
            let charge = stream.clone_dtoh(&grid)?;
            for (b, run) in out.iter_mut().enumerate() {
                let lo = b * n as usize;
                let hi = lo + n as usize;
                let p = Particles2 {
                    xe: host.xe[lo..hi].to_vec(),
                    ve: host.ve[lo..hi].to_vec(),
                    xi: host.xi[lo..hi].to_vec(),
                    vi: host.vi[lo..hi].to_vec(),
                };
                let ef: Vec<_> = (0..nn as usize)
                    .map(|j| {
                        [
                            field[2 * b * nn as usize + j].x,
                            field[(2 * b + 1) * nn as usize + j].x,
                        ]
                    })
                    .collect();
                let q = charge[(2 * b + 1) * nn as usize..(2 * b + 2) * nn as usize]
                    .iter()
                    .zip(&charge[2 * b * nn as usize..(2 * b + 1) * nn as usize])
                    .map(|(ion, electron)| ion.x as f64 - electron.x as f64)
                    .sum::<f64>()
                    / nn as f64;
                diagnostic2(run, &p, &ef, c, s, step, q);
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
    let positions = stream.clone_dtoh(&tracks)?;
    for (b, run) in out.iter_mut().enumerate() {
        run.fast_trajectory_x = positions[b * frames as usize * 32..(b + 1) * frames as usize * 32]
            .iter()
            .flat_map(|v| [v.x, v.y])
            .collect();
        run.voltage = voltage[b * s.steps..(b + 1) * s.steps]
            .iter()
            .map(|v| Complex32::new(v.x, v.y))
            .collect();
        run.elapsed_s = timer.elapsed().as_secs_f64();
    }
    Ok(out)
}
