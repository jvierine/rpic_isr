//! Float32 CUDA particles/fields with float64 nonlinear-residual reductions.
//! Unconverged macro-steps and particle steps are rejected, never hidden by
//! velocity rescaling. First prototype uses one realization per device.
use crate::{implicit2d::*, *};
use cudarc::{
    cufft::{
        sys::{cufftType, double2, float2},
        CudaFft, FftDirection,
    },
    driver::*,
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
pub fn simulate(o: &Options, c: &Config, s: &Scales) -> Result<ResultData> {
    let timer = std::time::Instant::now();
    let ctx = CudaContext::new(c.device)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(compile_ptx_with_opts(
        include_str!("implicit_kernels.cu"),
        CompileOptions {
            arch: Some("compute_61"),
            fmad: Some(true),
            ..Default::default()
        },
    )?)?;
    let clear = module.load_function("zero_i")?;
    let charge = module.load_function("charge_i")?;
    let poisson = module.load_function("potential_i")?;
    let mover = module.load_function("midpoint_i")?;
    let residual = module.load_function("residual_i")?;
    let state = State::new(c, s);
    let pack = |v: &[[f64; 2]]| {
        v.iter()
            .map(|v| float2 {
                x: v[0] as f32,
                y: v[1] as f32,
            })
            .collect::<Vec<_>>()
    };
    let mut x = stream.clone_htod(&pack(&state.x))?;
    let mut v = stream.clone_htod(&pack(&state.v))?;
    let mut xn = stream.alloc_zeros::<float2>(state.x.len())?;
    let mut vn = stream.alloc_zeros::<float2>(state.v.len())?;
    let ng = c.cells as i32;
    let nn = ng * ng;
    let np = (state.x.len() / 2) as i32;
    let length = s.length as f32;
    let dt = c.dt as f32;
    let mu = s.mass_ratio as f32;
    let mut grid = stream.alloc_zeros::<float2>(nn as usize)?;
    let mut freq = stream.alloc_zeros::<float2>(nn as usize)?;
    let mut phifreq = stream.alloc_zeros::<float2>(nn as usize)?;
    let mut oldphi = stream.alloc_zeros::<float2>(nn as usize)?;
    let mut newphi = stream.alloc_zeros::<float2>(nn as usize)?;
    let mut mid = stream.alloc_zeros::<float2>(nn as usize)?;
    let mut failures = stream.alloc_zeros::<i32>(1)?;
    let blocks = (nn as u32).div_ceil(256);
    let mut partial = stream.alloc_zeros::<double2>(blocks as usize)?;
    let fft = CudaFft::plan_2d(ng, ng, cufftType::CUFFT_C2C, stream.clone())?;
    let cfgp = LaunchConfig::for_num_elems(2 * np as u32);
    let cfgg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let solve = |positions: &CudaSlice<float2>,
                 grid: &mut CudaSlice<float2>,
                 freq: &mut CudaSlice<float2>,
                 phifreq: &mut CudaSlice<float2>,
                 phi: &mut CudaSlice<float2>,
                 failures: &mut CudaSlice<i32>|
     -> Result<()> {
        unsafe {
            stream
                .launch_builder(&clear)
                .arg(&mut *grid)
                .arg(&nn)
                .arg(failures)
                .launch(cfgg)?;
            stream
                .launch_builder(&charge)
                .arg(positions)
                .arg(&mut *grid)
                .arg(&np)
                .arg(&ng)
                .arg(&length)
                .launch(cfgp)?;
        }
        fft.exec_c2c(grid, freq, FftDirection::Forward)?;
        unsafe {
            stream
                .launch_builder(&poisson)
                .arg(&*freq)
                .arg(&mut *phifreq)
                .arg(&ng)
                .arg(&length)
                .launch(cfgg)?;
        }
        fft.exec_c2c(phifreq, phi, FftDirection::Inverse)?;
        Ok(())
    };
    solve(
        &x,
        &mut grid,
        &mut freq,
        &mut phifreq,
        &mut oldphi,
        &mut failures,
    )?;
    let mut out = ResultData::default();
    let collect = |out: &mut ResultData,
                   x: &CudaSlice<float2>,
                   v: &CudaSlice<float2>,
                   grid: &CudaSlice<float2>,
                   phi: &CudaSlice<float2>,
                   step: usize,
                   it: usize,
                   rms: f64|
     -> Result<()> {
        let unpack = |v: Vec<float2>| {
            v.iter()
                .map(|v| [v.x as f64, v.y as f64])
                .collect::<Vec<_>>()
        };
        let state = State {
            x: unpack(stream.clone_dtoh(x)?),
            v: unpack(stream.clone_dtoh(v)?),
        };
        let rho = stream.clone_dtoh(grid)?;
        let potential = stream.clone_dtoh(phi)?;
        let u = 0.5
            * rho
                .iter()
                .zip(&potential)
                .map(|(r, p)| r.x as f64 * p.x as f64)
                .sum::<f64>()
            / nn as f64;
        let q = rho.iter().map(|r| r.x as f64).sum::<f64>() / nn as f64;
        diagnostics(out, &state, u, c, s, step, it, rms, q);
        let mut reference = Potential::new(c.cells, s.length);
        reference.solve(&state, c.bragg_mode);
        out.density.push(reference.observable);
        Ok(())
    };
    collect(&mut out, &x, &v, &grid, &oldphi, 0, 0, 0.)?;
    let particle_iterations = o.particle_iterations as i32;
    let relaxation = o.relaxation as f32;
    for step in 1..=o.steps {
        stream.memcpy_dtod(&oldphi, &mut mid)?;
        let mut accepted = None;
        let mut last = 0.;
        for iteration in 1..=o.max_iterations {
            // Reset the failure flag BEFORE the mover, not during charge deposition.
            unsafe {
                stream
                    .launch_builder(&clear)
                    .arg(&mut grid)
                    .arg(&nn)
                    .arg(&mut failures)
                    .launch(cfgg)?;
                stream
                    .launch_builder(&mover)
                    .arg(&x)
                    .arg(&v)
                    .arg(&mut xn)
                    .arg(&mut vn)
                    .arg(&mid)
                    .arg(&np)
                    .arg(&ng)
                    .arg(&length)
                    .arg(&mu)
                    .arg(&dt)
                    .arg(&particle_iterations)
                    .arg(&mut failures)
                    .launch(cfgp)?;
            }
            // Charge grid is already zero. Preserve mover failure information.
            unsafe {
                stream
                    .launch_builder(&charge)
                    .arg(&xn)
                    .arg(&mut grid)
                    .arg(&np)
                    .arg(&ng)
                    .arg(&length)
                    .launch(cfgp)?;
            }
            fft.exec_c2c(&mut grid, &mut freq, FftDirection::Forward)?;
            unsafe {
                stream
                    .launch_builder(&poisson)
                    .arg(&freq)
                    .arg(&mut phifreq)
                    .arg(&ng)
                    .arg(&length)
                    .launch(cfgg)?;
            }
            fft.exec_c2c(&mut phifreq, &mut newphi, FftDirection::Inverse)?;
            unsafe {
                stream
                    .launch_builder(&residual)
                    .arg(&mut mid)
                    .arg(&oldphi)
                    .arg(&newphi)
                    .arg(&mut partial)
                    .arg(&nn)
                    .arg(&relaxation)
                    .launch(cfgg)?;
            }
            let norm = stream.clone_dtoh(&partial)?;
            let rms = (norm.iter().map(|p| p.x).sum::<f64>() / nn as f64).sqrt();
            let scale = (norm.iter().map(|p| p.y).sum::<f64>() / nn as f64).sqrt();
            last = rms;
            anyhow::ensure!(
                stream.clone_dtoh(&failures)?[0] == 0,
                "GPU particle orbit failed at step {step}"
            );
            if rms <= o.atol + o.rtol * scale {
                accepted = Some((iteration, rms));
                break;
            }
        }
        let (iterations,rms)=accepted.ok_or_else(||anyhow::anyhow!("CUDA implicit field failed at step {step}; last rms={last:e}; no unconverged step accepted"))?;
        std::mem::swap(&mut x, &mut xn);
        std::mem::swap(&mut v, &mut vn);
        std::mem::swap(&mut oldphi, &mut newphi);
        out.iterations.push(iterations as u32);
        out.residuals.push(rms);
        if step % o.diagnostic_stride == 0 || step == o.steps {
            collect(&mut out, &x, &v, &grid, &oldphi, step, iterations, rms)?;
        }
        if step % 10000 == 0 {
            eprintln!("implicit CUDA {}: {step}/{} steps, {:.1}s, iterations {iterations}, residual {rms:e}",c.device,o.steps,timer.elapsed().as_secs_f64());
        }
    }
    out.elapsed = timer.elapsed().as_secs_f64();
    out.final_rho = stream
        .clone_dtoh(&grid)?
        .iter()
        .map(|v| v.x as f64)
        .collect();
    out.final_phi = stream
        .clone_dtoh(&oldphi)?
        .iter()
        .map(|v| v.x as f64)
        .collect();
    Ok(out)
}
