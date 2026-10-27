//! Batched 2D cuFFT Poisson solves and CUDA particle push; float32 throughout.
use crate::{pic3v::{Setup, Particles3, diagnostic3, magnetic, thermostat, guard}, *};
#[repr(C)]
#[derive(Clone,Copy,Default)]
struct Velocity {x:f32,y:f32,z:f32,w:f32}
unsafe impl DeviceRepr for Velocity {}
unsafe impl ValidAsZeroBits for Velocity {}
use cudarc::{
    cufft::{
        sys::{cufftType, float2},
        CudaFft, FftDirection,
    },
    driver::*,
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::time::{Duration, Instant};
pub fn simulate(c: &Setup, s: &Scales, seeds: &[u64]) -> Result<Vec<Run>> {
    let timer = Instant::now();
    let ctx = CudaContext::new(c.device)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(compile_ptx_with_opts(
        format!("{}\n{}",include_str!("kernels_2d.cu"),include_str!("kernels_3v.cu")),
        CompileOptions {
            arch: Some("compute_61"),
            fmad: Some(false),
            ..Default::default()
        },
    )?)?;
    let clear = module.load_function("clear2")?;
    let deposit = module.load_function("deposit2")?;
    let poisson = module.load_function("poisson2")?;
    let push = module.load_function("push3")?;
    let trace = module.load_function("trace2")?;
    let web_path = std::env::var("RPIC_WEB_CAPTURE").ok();
    let web_pool: usize = std::env::var("RPIC_WEB_POOL").ok().map(|s| s.parse()).transpose()?.unwrap_or(4096);
    anyhow::ensure!((4096..=32768).contains(&web_pool), "Invalid replay particle pool");
    anyhow::ensure!(web_path.is_none() || c.cells*c.cells*c.ppc >= web_pool, "Replay pool exceeds species population");
    let web_frames: usize = std::env::var("RPIC_WEB_FRAMES").ok()
        .map(|s| s.parse()).transpose()?.unwrap_or(1024);
    anyhow::ensure!((2..=8192).contains(&web_frames), "Invalid web frame count");
    let web_crop = std::env::var("RPIC_WEB_CROP").is_ok_and(|v| v == "1");
    let web_grid = if web_crop {
        ((10.0 / s.dx + 2.0).ceil() as usize).next_power_of_two().min(c.cells)
    } else { c.cells };
    let web_start = (c.cells - web_grid) / 2;
    anyhow::ensure!(
        web_path.is_none() || (seeds.len() == 1 && c.cells*c.cells*c.ppc >= 4096 && (c.cells*c.cells*c.ppc).is_multiple_of(4096)),
        "Web capture requires one realization with a particle count divisible by 4096"
    );
    if let Some(ref path) = web_path {
        anyhow::ensure!(
            !std::path::Path::new(path).exists(),
            "Web capture already exists"
        );
    }
    let web_positions = module.load_function("web_positions2")?;
    let web_potential = module.load_function("web_potential2")?;
    let mut initial = Particles3::default();
    for &seed in seeds {
        let p = Particles3::new(c, s, seed);
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
    let velocities = |v: &[[f32;3]]| v.iter().map(|v| Velocity{x:v[0],y:v[1],z:v[2],w:0.}).collect::<Vec<_>>();
    let mut thermostat_work=0.;
    let n = (c.cells * c.cells * c.ppc) as i32;
    let ng = c.cells as i32;
    let nn = ng * ng;
    let batches = seeds.len() as i32;
    let length = s.length as f32;
    let dt = c.dt as f32;
    let mu = s.mass_ratio as f32;
    let mut xe = stream.clone_htod(&pairs(&initial.xe))?;
    let mut ve = stream.clone_htod(&velocities(&initial.ve))?;
    let mut xi = stream.clone_htod(&pairs(&initial.xi))?;
    let mut vi = stream.clone_htod(&velocities(&initial.vi))?;
    drop(initial);
    let mut grid = stream.alloc_zeros::<float2>((2 * nn * batches) as usize)?;
    let mut freq = stream.alloc_zeros::<float2>((2 * nn * batches) as usize)?;
    let mut efreq = stream.alloc_zeros::<float2>((2 * nn * batches) as usize)?;
    let mut e = stream.alloc_zeros::<float2>((2 * nn * batches) as usize)?;
    let total_steps=s.burn_steps+s.steps;
    let mut obs = stream.alloc_zeros::<float2>(total_steps * seeds.len())?;
    let mut modes = stream.alloc_zeros::<float2>(8 * total_steps * seeds.len())?;
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
    let web_fft = if web_path.is_some() {
        Some(CudaFft::plan_2d(
            ng,
            ng,
            cufftType::CUFFT_C2C,
            stream.clone(),
        )?)
    } else {
        None
    };
    let mut web_phi_freq = stream.alloc_zeros::<float2>(nn as usize)?;
    let mut web_phi = stream.alloc_zeros::<float2>(nn as usize)?;
    let mut web_points = stream.alloc_zeros::<float2>(2*web_pool)?;
    let mut web_x = Vec::<f32>::new();
    let mut web_fields = Vec::<f32>::new();
    let cfgp = LaunchConfig::for_num_elems((n * batches) as u32);
    let cfgg = LaunchConfig::for_num_elems((2 * nn * batches) as u32);
    let cfgf = LaunchConfig::for_num_elems((nn * batches) as u32);
    let mode = c.bragg_mode as i32;
    let kd = std::f64::consts::PI * c.bragg_mode as f64 / c.cells as f64;
    let norm = ((n as f64).sqrt() / nn as f64 / (kd.sin() / kd).powi(3)) as f32;
    let steps = total_steps as i32;
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
        if web_path.is_some() && step >= s.burn_steps {
            let elapsed = step - s.burn_steps;
            if elapsed.is_multiple_of(8) && elapsed / 8 < web_frames {
                let pool = web_pool as i32;
                unsafe {
                    stream
                        .launch_builder(&web_positions)
                        .arg(&xe)
                        .arg(&xi)
                        .arg(&mut web_points)
                        .arg(&n)
                        .arg(&pool)
                        .launch(LaunchConfig::for_num_elems((2*web_pool) as u32))?;
                    stream
                        .launch_builder(&web_potential)
                        .arg(&freq)
                        .arg(&mut web_phi_freq)
                        .arg(&ng)
                        .arg(&length)
                        .launch(LaunchConfig::for_num_elems(nn as u32))?;
                }
                web_fft.as_ref().unwrap().exec_c2c(
                    &mut web_phi_freq,
                    &mut web_phi,
                    FftDirection::Inverse,
                )?;
                web_x.extend(
                    stream
                        .clone_dtoh(&web_points)?
                        .into_iter()
                        .flat_map(|p| [p.x, p.y]),
                );
                let native = stream.clone_dtoh(&web_phi)?;
                for row in web_start..web_start+web_grid {
                    web_fields.extend(native[row*c.cells+web_start..row*c.cells+web_start+web_grid].iter().map(|p| p.x));
                }
            }
        }
        let record = if step < total_steps {
            step as i32
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
                .arg(&mut modes)
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
                    .arg(&magnetic(c,s)[0])
                    .arg(&magnetic(c,s)[1])
                    .arg(&magnetic(c,s)[2])
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
            let host = Particles3 {
                xe: unpack(stream.clone_dtoh(&xe)?),
                ve: stream.clone_dtoh(&ve)?.into_iter().map(|v|[v.x,v.y,v.z]).collect(),
                xi: unpack(stream.clone_dtoh(&xi)?),
                vi: stream.clone_dtoh(&vi)?.into_iter().map(|v|[v.x,v.y,v.z]).collect(),
            };
            let field = stream.clone_dtoh(&e)?;
            let charge = stream.clone_dtoh(&grid)?;
            for (b, run) in out.iter_mut().enumerate() {
                let lo = b * n as usize;
                let hi = lo + n as usize;
                let mut p = Particles3 {
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
                diagnostic3(run, &p, &ef, c, s, step, q, thermostat_work);
                crate::pic3v::save_live(run,c,s)?;
                guard(run)?;
                if c.core_thermostat_us>0. && step>0 && step<s.burn_steps {
                    thermostat_work+=thermostat(&mut p,&ef,c,s,step);
                    ve=stream.clone_htod(&velocities(&p.ve))?;
                }
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
                .arg(&magnetic(c,s)[0])
                    .arg(&magnetic(c,s)[1])
                    .arg(&magnetic(c,s)[2])
                .launch(cfgp)?;
        }
    }
    let voltage = stream.clone_dtoh(&obs)?;
    let positions = stream.clone_dtoh(&tracks)?;
    if let Some(path) = web_path {
        let frames = web_x.len() / (2*web_pool) / 2;
        anyhow::ensure!(frames == web_frames, "Incomplete web recording");
        if let Some(parent) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        let f = hdf5::File::create(format!("{path}.tmp"))?;
        for (name, data) in [
            ("config_json", serde_json::to_vec(c)?),
            ("scales_json", serde_json::to_vec(s)?),
            (
                "metadata_json",
                serde_json::to_vec(
                    &serde_json::json!({"source_revision":env!("RPIC_BUILD_REV"),
            "capture":"post-burn positions and endpoint potential; stride 8; fixed IDs p=j*(particles_per_species/pool) for each species",
            "recorded_particles_per_species":web_pool,"magnetic_field_z_t":c.bz_t,
            "solver":"explicit 2D3V TSC CUDA, arbitrary-B Boris","velocity_dimensions":3,"magnetic_field_x_t":c.bx_t,"magnetic_field_y_t":c.by_t,"seed":seeds[0],
            "potential_grid":web_grid,"potential_start_node":web_start,"frames":frames}),
                )?,
            ),
        ] {
            f.new_dataset_builder().with_data(&data).create(name)?;
        }
        f.new_dataset::<f32>()
            .shape([frames, 2*web_pool, 2])
            .create("positions_lambda_de")?
            .write_raw(&web_x)?;
        f.new_dataset::<f32>()
            .shape([frames, web_grid, web_grid])
            .create("potential_kbte_e")?
            .write_raw(&web_fields)?;
        f.new_dataset::<f32>()
            .shape([s.steps, 2])
            .create("raw_signal")?
            .write_raw(&voltage[s.burn_steps..].iter().flat_map(|p| [p.x, p.y]).collect::<Vec<_>>())?;
        let mode_values = stream.clone_dtoh(&modes)?;
        f.new_attr::<f64>().create("density_modes_first_time_s")?.write_scalar(&0.)?;
        f.new_dataset::<f32>().shape([total_steps,8,2])
            .create("electron_density_modes_1_to_8")?
            .write_raw(&mode_values.iter().flat_map(|p| [p.x,p.y]).collect::<Vec<_>>())?;
        f.flush()?;
        drop(f);
        std::fs::rename(format!("{path}.tmp"), path)?;
    }
    for (b, run) in out.iter_mut().enumerate() {
        run.fast_trajectory_x = positions[b * frames as usize * 32..(b + 1) * frames as usize * 32]
            .iter()
            .flat_map(|v| [v.x, v.y])
            .collect();
        run.voltage = voltage[b * total_steps+s.burn_steps..(b + 1) * total_steps]
            .iter()
            .map(|v| Complex32::new(v.x, v.y))
            .collect();
        run.elapsed_s = timer.elapsed().as_secs_f64();
    }
    Ok(out)
}

