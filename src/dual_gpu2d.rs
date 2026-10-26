//! One physical plasma, particle-sharded across one/two GPUs. The shared mesh
//! is reduced on GPU0, cuFFT solved once, and the packed real field broadcast.
//! No particle/mesh copies to the CPU in the timestep loop; GPU FP64 reductions
//! provide live diagnostics. Thermostat work is separately accounted.
use crate::{control2d::*, pic2d::Particles2, *};
use anyhow::{ensure, Context, Result};
use cudarc::{
    cufft::{
        sys::{cufftType, float2},
        CudaFft, FftDirection,
    },
    driver::*,
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::{sync::Arc, time::Instant};

struct Tiles {
    clear: CudaFunction,
    count: CudaFunction,
    prefix: CudaFunction,
    scatter: CudaFunction,
    deposit: CudaFunction,
    counts: CudaSlice<u32>,
    starts: CudaSlice<u32>,
    cursor: CudaSlice<u32>,
    indices: CudaSlice<u32>,
    interval: usize,
}

struct Gpu {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    clear: CudaFunction,
    deposit: CudaFunction,
    push: CudaFunction,
    moments: CudaFunction,
    reduce: CudaFunction,
    thermostat: CudaFunction,
    sample: CudaFunction,
    trace: CudaFunction,
    web: CudaFunction,
    xe: CudaSlice<float2>,
    ve: CudaSlice<float2>,
    xi: CudaSlice<float2>,
    vi: CudaSlice<float2>,
    density: CudaSlice<float2>,
    field: CudaSlice<float2>,
    block_moments: CudaSlice<f64>,
    totals: CudaSlice<f64>,
    samples: CudaSlice<float2>,
    web_points: CudaSlice<float2>,
    n: i32,
    blocks: i32,
    pool: i32,
    push_threads: u32,
    push_variant: String,
    tiles: Option<Tiles>,
}
impl Gpu {
    fn new(
        device: usize,
        p: &Particles2,
        lo: usize,
        hi: usize,
        ng: usize,
        pool: usize,
        length: f32,
    ) -> Result<Self> {
        let ctx = CudaContext::new(device)?;
        let stream = ctx.default_stream();
        let source = format!(
            "{}\n{}",
            include_str!("kernels_2d.cu"),
            include_str!("kernels_dual.cu")
        );
        let module = ctx.load_module(compile_ptx_with_opts(
            source,
            CompileOptions {
                arch: Some("compute_61"),
                fmad: Some(false),
                ..Default::default()
            },
        )?)?;
        {
            let mut symbol = module.get_global("rpic_invcell", &stream)?;
            let mut values = unsafe {
                symbol
                    .transmute_mut::<f32>(1)
                    .context("Invalid geometry constant")?
            };
            stream.memcpy_htod(&[ng as f32 / length], &mut values)?;
        }
        let push_variant = std::env::var("RPIC_PUSH_VARIANT").unwrap_or_else(|_| "packed".into());
        let push_name = match push_variant.as_str() {
            "packed" => "push_packed",
            "split" => "push_split",
            "scaled" => "push_scaled",
            _ => anyhow::bail!("Unknown push variant: {push_variant}"),
        };
        let load = |s| module.load_function(s);
        let upload = |v: &[[f32; 2]]| -> Result<CudaSlice<float2>> {
            Ok(stream.clone_htod(
                &v[lo..hi]
                    .iter()
                    .map(|v| float2 { x: v[0], y: v[1] })
                    .collect::<Vec<_>>(),
            )?)
        };
        let n = hi - lo;
        let blocks = n.max(ng * ng).div_ceil(256);
        let interval = std::env::var("RPIC_TILE_INTERVAL")
            .ok()
            .map(|x| x.parse::<usize>())
            .transpose()?
            .unwrap_or(0);
        let tiles = if interval > 0 {
            let bins = 2 * (ng / 8) * (ng / 8);
            Some(Tiles {
                clear: load("clear_counts")?,
                count: load("count_tiles")?,
                prefix: load("prefix_tiles")?,
                scatter: load("scatter_tiles")?,
                deposit: load("deposit_tiled")?,
                counts: stream.alloc_zeros(bins)?,
                starts: stream.alloc_zeros(bins)?,
                cursor: stream.alloc_zeros(bins)?,
                indices: stream.alloc_zeros(2 * n)?,
                interval,
            })
        } else {
            None
        };
        Ok(Self {
            clear: load("clear2")?,
            deposit: load("deposit_shard")?,
            push: load(push_name)?,
            moments: load("moments_packed")?,
            reduce: load("reduce_moments")?,
            thermostat: load("thermostat_packed")?,
            sample: load("sample_packed")?,
            trace: load("trace2")?,
            web: load("web_positions2")?,
            xe: upload(&p.xe)?,
            ve: upload(&p.ve)?,
            xi: upload(&p.xi)?,
            vi: upload(&p.vi)?,
            density: stream.alloc_zeros(ng * ng)?,
            field: stream.alloc_zeros(ng * ng)?,
            block_moments: stream.alloc_zeros(blocks * 14)?,
            totals: stream.alloc_zeros(14)?,
            samples: stream.alloc_zeros(64)?,
            web_points: stream.alloc_zeros(pool * 2)?,
            n: n as i32,
            blocks: blocks as i32,
            pool: pool as i32,
            push_threads: if push_variant == "packed" {
                n as u32
            } else {
                2 * n as u32
            },
            push_variant,
            stream,
            module,
            tiles,
        })
    }
    fn deposit(&mut self, total: i32, ng: i32, length: f32, step: usize) -> Result<()> {
        unsafe {
            self.stream
                .launch_builder(&self.clear)
                .arg(&mut self.density)
                .arg(&(ng * ng))
                .launch(LaunchConfig::for_num_elems((ng * ng) as u32))?;
            if let Some(ref mut tile) = self.tiles {
                let bins = (ng / 8) * (ng / 8);
                let particles = LaunchConfig::for_num_elems(self.n as u32);
                if step % tile.interval == 0 {
                    self.stream
                        .launch_builder(&tile.clear)
                        .arg(&mut tile.counts)
                        .arg(&(2 * bins))
                        .launch(LaunchConfig::for_num_elems((2 * bins) as u32))?;
                    self.stream
                        .launch_builder(&tile.count)
                        .arg(&self.xe)
                        .arg(&self.xi)
                        .arg(&mut tile.counts)
                        .arg(&self.n)
                        .arg(&ng)
                        .arg(&length)
                        .launch(particles)?;
                    self.stream
                        .launch_builder(&tile.prefix)
                        .arg(&tile.counts)
                        .arg(&mut tile.starts)
                        .arg(&mut tile.cursor)
                        .arg(&bins)
                        .arg(&self.n)
                        .launch(LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                    self.stream
                        .launch_builder(&tile.scatter)
                        .arg(&self.xe)
                        .arg(&self.xi)
                        .arg(&mut tile.cursor)
                        .arg(&mut tile.indices)
                        .arg(&self.n)
                        .arg(&ng)
                        .arg(&length)
                        .launch(particles)?;
                }
                let weight = (ng * ng) as f32 / total as f32;
                self.stream
                    .launch_builder(&tile.deposit)
                    .arg(&self.xe)
                    .arg(&self.xi)
                    .arg(&mut self.density)
                    .arg(&tile.counts)
                    .arg(&tile.starts)
                    .arg(&tile.indices)
                    .arg(&ng)
                    .arg(&length)
                    .arg(&weight)
                    .launch(LaunchConfig {
                        grid_dim: ((2 * bins) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            } else {
                self.stream
                    .launch_builder(&self.deposit)
                    .arg(&self.xe)
                    .arg(&self.xi)
                    .arg(&mut self.density)
                    .arg(&self.n)
                    .arg(&total)
                    .arg(&ng)
                    .arg(&length)
                    .launch(LaunchConfig::for_num_elems(self.n as u32))?;
            }
        }
        Ok(())
    }
    fn push(&mut self, ng: i32, length: f32, mu: f32, kick: f32, drift: f32) -> Result<()> {
        unsafe {
            self.stream
                .launch_builder(&self.push)
                .arg(&mut self.xe)
                .arg(&mut self.ve)
                .arg(&mut self.xi)
                .arg(&mut self.vi)
                .arg(&self.field)
                .arg(&self.n)
                .arg(&ng)
                .arg(&length)
                .arg(&mu)
                .arg(&kick)
                .arg(&drift)
                .launch(LaunchConfig::for_num_elems(self.push_threads))?;
        }
        Ok(())
    }
    fn moments(
        &mut self,
        density: &CudaSlice<float2>,
        ng: i32,
        length: f32,
        mu: f32,
        dt: f32,
        include: i32,
    ) -> Result<Vec<f64>> {
        unsafe {
            self.stream
                .launch_builder(&self.moments)
                .arg(&self.xe)
                .arg(&self.ve)
                .arg(&self.xi)
                .arg(&self.vi)
                .arg(&self.field)
                .arg(density)
                .arg(&mut self.block_moments)
                .arg(&self.n)
                .arg(&ng)
                .arg(&length)
                .arg(&mu)
                .arg(&dt)
                .arg(&include)
                .launch(LaunchConfig {
                    grid_dim: (self.blocks as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            self.stream
                .launch_builder(&self.reduce)
                .arg(&self.block_moments)
                .arg(&mut self.totals)
                .arg(&self.blocks)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(self.stream.clone_dtoh(&self.totals)?)
    }
    fn rescale(
        &mut self,
        state: &State,
        a: [f64; 2],
        ng: i32,
        length: f32,
        mu: f32,
        dt: f32,
    ) -> Result<()> {
        unsafe {
            self.stream
                .launch_builder(&self.thermostat)
                .arg(&self.xe)
                .arg(&mut self.ve)
                .arg(&self.xi)
                .arg(&mut self.vi)
                .arg(&self.field)
                .arg(&self.n)
                .arg(&ng)
                .arg(&length)
                .arg(&mu)
                .arg(&dt)
                .arg(&state.mean[0][0])
                .arg(&state.mean[0][1])
                .arg(&state.mean[1][0])
                .arg(&state.mean[1][1])
                .arg(&a[0])
                .arg(&a[1])
                .launch(LaunchConfig::for_num_elems(self.n as u32))?;
        }
        Ok(())
    }
}

fn state(
    gpus: &mut [Gpu],
    density: &CudaSlice<float2>,
    c: &Config,
    s: &Scales,
    step: usize,
) -> Result<State> {
    let mut sum = vec![0.; 14];
    // Only GPU0 reads the shared density argument. Other GPUs use a local dummy
    // allocation and include_field=0, avoiding cross-context pointer dereference.
    for (index, g) in gpus.iter_mut().enumerate() {
        let m = if index == 0 {
            g.moments(
                density,
                c.cells as i32,
                s.length as f32,
                s.mass_ratio as f32,
                c.dt as f32,
                1,
            )?
        } else {
            let dummy = g.stream.alloc_zeros::<float2>(1)?;
            g.moments(
                &dummy,
                c.cells as i32,
                s.length as f32,
                s.mass_ratio as f32,
                c.dt as f32,
                0,
            )?
        };
        for (k, x) in m.into_iter().enumerate() {
            sum[k] += x;
        }
    }
    Ok(from_moments(
        &sum,
        c.cells * c.cells * c.ppc,
        c.cells,
        c,
        s,
        step as f64 * s.dt_s,
    ))
}

pub fn simulate(original: &Config, options: Controls) -> Result<(Config, Scales, Run)> {
    let timer = Instant::now();
    let mut c = original.clone();
    c.backend = Backend::Cuda;
    ensure!(
        c.realizations == 1 && c.gpu_batch_size == 1,
        "Dual-particle solver currently runs one physical realization"
    );
    let mut control = Controller::new(options)?;
    let s = Scales::new(&c)?;
    let live = control
        .options
        .live_diagnostics
        .clone()
        .unwrap_or_else(|| format!("{}.live.h5", c.output));
    ensure!(
        !std::path::Path::new(&c.output).exists() && !std::path::Path::new(&live).exists(),
        "Output/live checkpoint already exists"
    );
    let total = c.cells * c.cells * c.ppc;
    let devices = control.options.devices.clone();
    let count = devices.len();
    ensure!(
        total % count == 0 && total / count >= 16,
        "Invalid particle partition"
    );
    let web_path = std::env::var("RPIC_WEB_CAPTURE").ok();
    let web_frames = std::env::var("RPIC_WEB_FRAMES")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(4096);
    ensure!(
        (2..=8192).contains(&web_frames),
        "Invalid dense frame count"
    );
    if let Some(ref path) = web_path {
        ensure!(
            !std::path::Path::new(path).exists()
                && !std::path::Path::new(&format!("{path}.tmp")).exists(),
            "Capture exists"
        );
        ensure!(
            total % 4096 == 0 && s.steps >= 8 * (web_frames - 1),
            "Invalid web sampling"
        );
    }
    let initial = Particles2::new(&c, &s, c.seed);
    let mut gpus = vec![];
    for (index, &device) in devices.iter().enumerate() {
        gpus.push(
            Gpu::new(
                device,
                &initial,
                index * total / count,
                (index + 1) * total / count,
                c.cells,
                4096 / count,
                s.length as f32,
            )
            .with_context(|| format!("GPU {device} initialization"))?,
        );
    }
    drop(initial);
    let stream = gpus[0].stream.clone();
    let module = gpus[0].module.clone();
    let fold = module.load_function("combine_density")?;
    let poisson = module.load_function("poisson2")?;
    let pack = module.load_function("pack_field")?;
    let record_mode = module.load_function("record_mode")?;
    let phi_kernel = module.load_function("web_potential2")?;
    let crop_kernel = module.load_function("crop_phi")?;
    let ng = c.cells as i32;
    let nn = ng * ng;
    let n = total as i32;
    let length = s.length as f32;
    let mu = s.mass_ratio as f32;
    let dt = c.dt as f32;
    let mut peer = stream.alloc_zeros::<float2>(nn as usize)?;
    let mut grid = stream.alloc_zeros::<float2>(2 * nn as usize)?;
    let mut freq = stream.alloc_zeros::<float2>(2 * nn as usize)?;
    let mut efreq = stream.alloc_zeros::<float2>(2 * nn as usize)?;
    let mut efull = stream.alloc_zeros::<float2>(2 * nn as usize)?;
    let mut voltage = stream.alloc_zeros::<float2>(s.steps)?;
    let fft = CudaFft::plan_many(
        &[ng, ng],
        None,
        1,
        nn,
        None,
        1,
        nn,
        cufftType::CUFFT_C2C,
        2,
        stream.clone(),
    )?;
    let fast_frames = (s.burn_steps + s.steps + 1).min(2048);
    let mut tracks = stream.alloc_zeros::<float2>(fast_frames * 32)?;
    let mut phi_freq = stream.alloc_zeros::<float2>(nn as usize)?;
    let mut phi = stream.alloc_zeros::<float2>(nn as usize)?;
    let phi_fft = CudaFft::plan_2d(ng, ng, cufftType::CUFFT_C2C, stream.clone())?;
    let crop = ((10. / s.dx + 2.).ceil() as usize)
        .next_power_of_two()
        .min(c.cells);
    let start = (c.cells - crop) / 2;
    let mut cropped_phi = stream.alloc_zeros::<f32>(crop * crop)?;
    let mut web_x = vec![];
    let mut web_phi = vec![];
    let mut run = Run::default();
    let mode = c.bragg_mode as i32;
    let kd = std::f64::consts::PI * c.bragg_mode as f64 / c.cells as f64;
    let norm = ((total as f64).sqrt() / nn as f64 / (kd.sin() / kd).powi(3)) as f32;
    let grid_cfg = LaunchConfig::for_num_elems(nn as u32);
    if control.options.thermostat == Thermostat::Off {
        control.record_step = Some(s.burn_steps);
    }
    let max_burn = s.burn_steps * 2
        + (control.options.thermostat_settle_ms * 2e-3 / s.dt_s).ceil() as usize
        + c.diagnostic_stride * 2;
    let mut step = 0usize;
    let profiling = std::env::var_os("RPIC_PROFILE").is_some();
    let mut stage_seconds = [0.; 6];
    let mut profile_count = 0usize;
    let sync_all = |gpus: &[Gpu]| -> Result<()> {
        for g in gpus {
            g.stream.synchronize()?;
        }
        Ok(())
    };
    loop {
        if step % 256 == 0
            && control
                .options
                .max_wall_seconds
                .is_some_and(|x| timer.elapsed().as_secs_f64() >= x)
        {
            control.checkpoint(
                &live,
                &c,
                &s,
                "aborted: wall-time budget reached",
                &run.diagnostics,
            )?;
            anyhow::bail!("Wall-time budget reached; diagnostics retained in {live}");
        }
        let measured = profiling && step >= 32 && step < 288 && step % 16 == 0;
        if measured {
            sync_all(&gpus)?;
            profile_count += 1;
        }
        let mut stamp = Instant::now();
        for g in &mut gpus {
            g.deposit(n, ng, length, step)?;
        }
        if measured {
            sync_all(&gpus)?;
            stage_seconds[0] += stamp.elapsed().as_secs_f64();
            stamp = Instant::now();
        }
        if count == 2 {
            stream
                .memcpy_dtod(&gpus[1].density, &mut peer)
                .context("Cross-GPU density copy")?;
            // Cross-context read lifetime: GPU1 may not overwrite its grid until copy ends.
            gpus[1].stream.wait(&stream.record_event(None)?)?;
        }
        unsafe {
            stream
                .launch_builder(&fold)
                .arg(&gpus[0].density)
                .arg(&peer)
                .arg(&mut grid)
                .arg(&nn)
                .launch(grid_cfg)?;
        }
        if measured {
            sync_all(&gpus)?;
            stage_seconds[1] += stamp.elapsed().as_secs_f64();
            stamp = Instant::now();
        }
        stream.context().bind_to_thread()?;
        fft.exec_c2c(&mut grid, &mut freq, FftDirection::Forward)?;
        unsafe {
            stream
                .launch_builder(&poisson)
                .arg(&freq)
                .arg(&mut efreq)
                .arg(&mut voltage)
                .arg(&ng)
                .arg(&1i32)
                .arg(&length)
                .arg(&mode)
                .arg(&norm)
                .arg(&-1i32)
                .arg(&(s.steps as i32))
                .launch(grid_cfg)?;
        }
        stream.context().bind_to_thread()?;
        fft.exec_c2c(&mut efreq, &mut efull, FftDirection::Inverse)?;
        unsafe {
            stream
                .launch_builder(&pack)
                .arg(&efull)
                .arg(&mut gpus[0].field)
                .arg(&nn)
                .launch(grid_cfg)?;
        }
        if measured {
            sync_all(&gpus)?;
            stage_seconds[2] += stamp.elapsed().as_secs_f64();
            stamp = Instant::now();
        }
        if count == 2 {
            let (a, b) = gpus.split_at_mut(1);
            let other = b[0].stream.clone();
            other
                .memcpy_dtod(&a[0].field, &mut b[0].field)
                .context("Cross-GPU field copy")?;
            // Protect GPU0's field from overwrite while the remote copy is in flight.
            stream.wait(&other.record_event(None)?)?;
        }
        if step == 0 {
            for g in &mut gpus {
                g.push(ng, length, mu, -0.5 * dt, 0.)?;
            }
        }
        if measured {
            sync_all(&gpus)?;
            stage_seconds[3] += stamp.elapsed().as_secs_f64();
            stamp = Instant::now();
        }
        if step < fast_frames {
            unsafe {
                stream
                    .launch_builder(&gpus[0].trace)
                    .arg(&gpus[0].xe)
                    .arg(&gpus[0].xi)
                    .arg(&mut tracks)
                    .arg(&gpus[0].n)
                    .arg(&1i32)
                    .arg(&(step as i32))
                    .arg(&(fast_frames as i32))
                    .arg(&(s.lambda_de as f32))
                    .launch(LaunchConfig::for_num_elems(32))?;
            }
        }
        if control.options.thermostat == Thermostat::Off && step == s.burn_steps {
            control.phase = Phase::Record;
        }
        let final_step = control.record_step.is_some_and(|b| step == b + s.steps);
        if step % c.diagnostic_stride == 0 || step == s.burn_steps || final_step {
            let pre = state(&mut gpus, &grid, &c, &s, step)?;
            let a = match control.decide(&pre, step, &c) {
                Ok(a) => a,
                Err(error) => {
                    control.checkpoint(
                        &live,
                        &c,
                        &s,
                        &format!("aborted: {error}"),
                        &run.diagnostics,
                    )?;
                    return Err(error);
                }
            };
            if a != [1., 1.] {
                for g in &mut gpus {
                    g.rescale(&pre, a, ng, length, mu, dt)?;
                }
            }
            let post = if a == [1., 1.] {
                pre.clone()
            } else {
                state(&mut gpus, &grid, &c, &s, step)?
            };
            run.diagnostics.extend(post.row);
            let sample_fn = gpus[0].sample.clone();
            let g = &mut gpus[0];
            unsafe {
                stream
                    .launch_builder(&sample_fn)
                    .arg(&g.xe)
                    .arg(&g.ve)
                    .arg(&g.xi)
                    .arg(&g.vi)
                    .arg(&g.field)
                    .arg(&mut g.samples)
                    .arg(&ng)
                    .arg(&length)
                    .arg(&mu)
                    .arg(&dt)
                    .arg(&(s.lambda_de as f32))
                    .arg(&(s.vte as f32))
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            let sample = stream.clone_dtoh(&g.samples)?;
            run.trajectory_x
                .extend(sample[..32].iter().flat_map(|v| [v.x, v.y]));
            run.trajectory_v
                .extend(sample[32..].iter().flat_map(|v| [v.x, v.y]));
            let guard = control.record(&pre, &post, a);
            let status = if let Err(ref e) = guard {
                format!("aborted: {e}")
            } else {
                format!("{:?}", control.phase)
            };
            control.checkpoint(&live, &c, &s, &status, &run.diagnostics)?;
            let last = &control.history[control.history.len() - 16..];
            eprintln!("dual {:?} t={:.6}ms Te={:.4}K Ti={:.4}K balance_drift={:.6}% removed={:.8} phase={:?} wall={:.2}s",devices,post.row[0]*1e3,post.row[5],post.row[6],last[8],control.removed,control.phase,timer.elapsed().as_secs_f64());
            guard?;
            if control.record_step.is_none() && step >= max_burn {
                control.checkpoint(
                    &live,
                    &c,
                    &s,
                    "aborted: stationarity gate not reached",
                    &run.diagnostics,
                )?;
                anyhow::bail!("Stationarity/thermostat settling not reached within burn limit");
            }
        }
        if let Some(burn) = control.record_step {
            if step == burn + s.steps {
                break;
            }
            if step >= burn {
                let record = step - burn;
                unsafe {
                    stream
                        .launch_builder(&record_mode)
                        .arg(&freq)
                        .arg(&mut voltage)
                        .arg(&mode)
                        .arg(&(record as i32))
                        .arg(&norm)
                        .launch(LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
                if web_path.is_some() && record % 8 == 0 && record / 8 < web_frames {
                    let mut chunks = vec![];
                    for g in &mut gpus {
                        unsafe {
                            g.stream
                                .launch_builder(&g.web)
                                .arg(&g.xe)
                                .arg(&g.xi)
                                .arg(&mut g.web_points)
                                .arg(&g.n)
                                .arg(&g.pool)
                                .launch(LaunchConfig::for_num_elems(2 * g.pool as u32))?;
                        }
                        chunks.push(g.stream.clone_dtoh(&g.web_points)?);
                    }
                    for species in 0..2 {
                        for (index, g) in gpus.iter().enumerate() {
                            let lo = species * g.pool as usize;
                            web_x.extend(
                                chunks[index][lo..lo + g.pool as usize]
                                    .iter()
                                    .flat_map(|v| [v.x, v.y]),
                            );
                        }
                    }
                    unsafe {
                        stream
                            .launch_builder(&phi_kernel)
                            .arg(&freq)
                            .arg(&mut phi_freq)
                            .arg(&ng)
                            .arg(&length)
                            .launch(grid_cfg)?;
                    }
                    stream.context().bind_to_thread()?;
                    phi_fft.exec_c2c(&mut phi_freq, &mut phi, FftDirection::Inverse)?;
                    unsafe {
                        stream
                            .launch_builder(&crop_kernel)
                            .arg(&phi)
                            .arg(&mut cropped_phi)
                            .arg(&ng)
                            .arg(&(crop as i32))
                            .arg(&(start as i32))
                            .launch(LaunchConfig::for_num_elems((crop * crop) as u32))?;
                    }
                    web_phi.extend(stream.clone_dtoh(&cropped_phi)?);
                }
            }
        }
        if measured {
            sync_all(&gpus)?;
            stage_seconds[4] += stamp.elapsed().as_secs_f64();
            stamp = Instant::now();
        }
        for g in &mut gpus {
            g.push(ng, length, mu, dt, dt)?;
        }
        if measured {
            sync_all(&gpus)?;
            stage_seconds[5] += stamp.elapsed().as_secs_f64();
        }
        step += 1;
    }
    stream.synchronize()?;
    for g in &gpus {
        g.stream.synchronize()?;
    }
    run.voltage = stream
        .clone_dtoh(&voltage)?
        .into_iter()
        .map(|v| Complex32::new(v.x, v.y))
        .collect();
    run.fast_trajectory_x = stream
        .clone_dtoh(&tracks)?
        .into_iter()
        .flat_map(|v| [v.x, v.y])
        .collect();
    run.elapsed_s = timer.elapsed().as_secs_f64();
    let mut metadata = control.metadata();
    metadata["tile_interval"] = serde_json::json!(gpus[0].tiles.as_ref().map(|x| x.interval));
    metadata["push_variant"] = serde_json::json!(gpus[0].push_variant);
    metadata["elapsed_s"] = serde_json::json!(run.elapsed_s);
    if profile_count > 0 {
        let mean_ms: Vec<_> = stage_seconds
            .iter()
            .map(|x| x * 1000. / profile_count as f64)
            .collect();
        eprintln!(
            "PROFILE synchronized stage ms {:?}: {:?}",
            [
                "deposit",
                "density_combine_copy",
                "fft_poisson_pack",
                "field_copy",
                "record_diagnostics",
                "push"
            ],
            mean_ms
        );
        metadata["profile"] = serde_json::json!({"method":"synchronized host timings; instrumented, not production throughput","samples":profile_count,"stage_order":["deposit","density_combine_copy","fft_poisson_pack","field_copy","record_diagnostics","push"],"mean_ms":mean_ms});
    }
    run.control_metadata = Some(metadata);
    run.control_history = control.history.clone();
    let actual_burn = control.record_step.unwrap();
    c.burn_ms = actual_burn as f64 * s.dt_s * 1e3;
    let mut actual = Scales::new(&c)?;
    actual.burn_steps = actual_burn;
    if let Some(path) = web_path {
        ensure!(
            web_x.len() == web_frames * 8192 * 2,
            "Incomplete web positions"
        );
        if let Some(p) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(p)?;
        }
        let f = hdf5::File::create(format!("{path}.tmp"))?;
        let metadata = serde_json::json!({"source_revision":env!("RPIC_BUILD_REV"),"solver":"explicit packed TSC particle-sharded CUDA PIC","devices":devices,"capture":"fixed global particle IDs and native cropped potential","potential_grid":crop,"potential_start_node":start,"frames":web_frames,"seed":c.seed,"control":control.metadata()});
        for (name, data) in [
            ("config_json", serde_json::to_vec(&c)?),
            ("scales_json", serde_json::to_vec(&actual)?),
            ("metadata_json", serde_json::to_vec(&metadata)?),
        ] {
            f.new_dataset_builder().with_data(&data).create(name)?;
        }
        f.new_dataset::<f32>()
            .shape([web_frames, 8192, 2])
            .create("positions_lambda_de")?
            .write_raw(&web_x)?;
        f.new_dataset::<f32>()
            .shape([web_frames, crop, crop])
            .create("potential_kbte_e")?
            .write_raw(&web_phi)?;
        f.new_dataset::<f32>()
            .shape([run.voltage.len(), 2])
            .create("raw_signal")?
            .write_raw(
                &run.voltage
                    .iter()
                    .flat_map(|z| [z.re, z.im])
                    .collect::<Vec<_>>(),
            )?;
        f.flush()?;
        drop(f);
        std::fs::rename(format!("{path}.tmp"), path)?;
    }
    control.checkpoint(
        &live,
        &c,
        &actual,
        "simulation_complete_pending_spectral_save",
        &run.diagnostics,
    )?;
    Ok((c, actual, run))
}
