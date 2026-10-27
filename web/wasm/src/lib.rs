//! Browser replay and spectral analysis of actual large-GPU PIC recordings.
//! Positions are display-quantized only. Nothing here invents particle motion.
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// Inset legend drawn into the figure canvas, independent of data zoom.
#[wasm_bindgen]
pub fn inset_legend(canvas: web_sys::HtmlCanvasElement, width: f64, height: f64,
    dpr: f64, voltage: bool, theory: bool, plasma: bool) -> Result<(), JsValue> {
    canvas.set_width((width*dpr).round() as u32);
    canvas.set_height((height*dpr).round() as u32);
    let ctx=canvas.get_context("2d")?.unwrap().dyn_into::<web_sys::CanvasRenderingContext2d>()?;
    ctx.scale(dpr,dpr)?;
    let entries=if voltage {vec![("Real", "#126ca4"),("Imaginary", "#b64a16")]}
        else if theory {vec![("PIC",if plasma {"#762a83"} else {"#18555c"}),("Theory","#b66b14")]}
        else {vec![("PIC",if plasma {"#762a83"} else {"#18555c"})]};
    let x=78.;let y=36.;
    ctx.set_fill_style_str("#f5f1e7");ctx.set_global_alpha(0.94);
    ctx.fill_rect(x,y,98.,entries.len() as f64*18.+10.);ctx.set_global_alpha(1.);
    ctx.set_font("12px sans-serif");ctx.set_text_baseline("middle");
    for (j,(label,color)) in entries.iter().enumerate(){let yy=y+14.+j as f64*18.;
        ctx.set_stroke_style_str(color);ctx.set_line_width(2.);
        ctx.begin_path();ctx.move_to(x+7.,yy);ctx.line_to(x+27.,yy);ctx.stroke();
        ctx.set_fill_style_str("#303534");ctx.fill_text(label,x+34.,yy)?;
    }Ok(())
}

#[wasm_bindgen]
pub struct PlasmaView {
    field: rustplotlib_web::FieldView,
    particles: rustplotlib_web::LinePlot,
    canvas: web_sys::HtmlCanvasElement,
}

#[wasm_bindgen]
impl PlasmaView {
    #[wasm_bindgen(constructor)]
    pub fn new(
        field_canvas: web_sys::HtmlCanvasElement,
        particle_canvas: web_sys::HtmlCanvasElement,
    ) -> Result<PlasmaView, JsValue> {
        Ok(Self {
            field: rustplotlib_web::FieldView::new(field_canvas.clone())?,
            particles: rustplotlib_web::LinePlot::new(particle_canvas)?,
            canvas: field_canvas,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        field: &[f32],
        points: &[f32],
        width: f64,
        height: f64,
        dpr: f64,
        length: f32,
        scale: f32,
        visible: bool,
    ) -> Result<(), JsValue> {
        self.canvas.set_width((width * dpr).round() as u32);
        self.canvas.set_height((height * dpr).round() as u32);
        let grid = (field.len() as f64).sqrt() as usize;
        if grid * grid != field.len() || grid < 2 {
            return Err(JsValue::from_str("Potential grid must be square"));
        }
        self.field.set_data(grid as u32, grid as u32, field)?;
        self.field.set_scale(-scale, scale, 0, 0.1, 0)?;
        let palette: &[f32] = if visible {
            &[
                0., 0., 0.3, 0., 0., 1., 1., 1., 0.86, 1., 0., 0., 0.5, 0., 0.,
            ]
        } else {
            &[0.914, 0.894, 0.839, 0.914, 0.894, 0.839]
        };
        self.field.set_colormap(palette)?;
        let lo = (length - 10.) * 0.5 / length;
        let hi = lo + 10. / length;
        self.field.set_view(lo, lo, hi, hi)?;
        self.particles.set_margins(0., 0., 0., 0.)?;
        self.particles.begin(width, height, dpr, 0., 1., 0., 1.)?;
        for species in 0..2 {
            let xy: Vec<f64> = points
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|p| p[2] as usize == species)
                .flat_map(|p| [p[0] as f64, p[1] as f64])
                .collect();
            let size = if species == 0 { 5. } else { 6.5 };
            self.particles
                .series(&xy, 0.025, 0.035, 0.05, true, size, false)?;
            let rgb = if species == 0 {
                [0.12, 0.8, 1.]
            } else {
                [1., 0.68, 0.12]
            };
            self.particles
                .series(&xy, rgb[0], rgb[1], rgb[2], true, size - 2., false)?;
        }
        Ok(())
    }
}

/// Rustplotlib axes and WebGL2 curves on separate, aligned canvases.
#[wasm_bindgen]
pub struct ScientificPlot {
    axes: rustplotlib_web::SeriesView,
    lines: rustplotlib_web::LinePlot,
    viewport: rustplotlib_web::PlotViewport,
}

#[wasm_bindgen]
impl ScientificPlot {
    #[wasm_bindgen(constructor)]
    pub fn new(
        axes: web_sys::HtmlCanvasElement,
        lines: web_sys::HtmlCanvasElement,
    ) -> Result<ScientificPlot, JsValue> {
        Ok(Self {
            axes: rustplotlib_web::SeriesView::new(axes)?,
            lines: rustplotlib_web::LinePlot::new(lines)?,
            viewport: rustplotlib_web::PlotViewport::new(),
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn begin(
        &mut self,
        width: f64,
        height: f64,
        dpr: f64,
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
        xlabel: &str,
        ylabel: &str,
    ) -> Result<(), JsValue> {
        let v = self.viewport.bounds();
        let (x0, x1, y0, y1) = (x0 + v[0]*(x1-x0), x0 + v[2]*(x1-x0), y0 + v[1]*(y1-y0), y0 + v[3]*(y1-y0));
        self.axes
            .begin_axes(width, height, dpr, x0, x1, y0, y1, xlabel, ylabel)?;
        self.axes.x_ticks(false, false)?;
        self.lines.set_margins(70., 18., 30., 48.)?;
        self.lines.begin(width, height, dpr, x0, x1, y0, y1)
    }
    pub fn zoom(&mut self, x: f64, y: f64, factor: f64) { self.viewport.zoom(x, y, factor); }
    pub fn zoom_box(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) -> bool { self.viewport.zoom_box(x0, y0, x1, y1) }
    pub fn pan(&mut self, dx: f64, dy: f64) { self.viewport.pan(dx, dy); }
    pub fn reset(&mut self) { self.viewport.reset(); }
    pub fn viewport_bounds(&self) -> Vec<f64> { self.viewport.bounds() }
    pub fn series(
        &self,
        xy: &[f64],
        red: f32,
        green: f32,
        blue: f32,
        width: f32,
        dashed: bool,
    ) -> Result<(), JsValue> {
        self.lines
            .series(xy, red, green, blue, false, width, dashed)
    }
}

#[wasm_bindgen]
pub struct Replay {
    positions: Vec<u16>,
    potential: Vec<i16>,
    frames: usize,
    pool: usize,
    grid: usize,
    length: f32,
    extent: f32,
    phi_scale: f32,
}

fn periodic_mix(a: f32, b: f32, t: f32, length: f32) -> f32 {
    let d = (b - a + 0.5 * length).rem_euclid(length) - 0.5 * length;
    (a + t * d).rem_euclid(length)
}

#[wasm_bindgen]
impl Replay {
    #[wasm_bindgen(constructor)]
    pub fn new(
        positions: &[u16],
        potential: &[i16],
        pool: usize,
        grid: usize,
        length: f32,
        extent: f32,
        phi_scale: f32,
    ) -> Result<Replay, JsValue> {
        if pool == 0
            || grid < 2
            || length <= extent
            || extent <= 0.
            || !length.is_finite()
            || !phi_scale.is_finite()
            || phi_scale <= 0.
        {
            return Err(JsValue::from_str("Invalid replay geometry"));
        }
        let frames = positions.len() / (4 * pool);
        if frames < 2
            || positions.len() != frames * 4 * pool
            || potential.len() != frames * grid * grid
        {
            return Err(JsValue::from_str("Incomplete replay recording"));
        }
        Ok(Self {
            positions: positions.to_vec(),
            potential: potential.to_vec(),
            frames,
            pool,
            grid,
            length,
            extent,
            phi_scale,
        })
    }

    /// Stable particle-ID priority within the crop. No rescaling of motion.
    /// Output x,y,species,id; equal per-species quotas, up to the display cap.
    pub fn points(&self, frame: f32, count: usize, electrons: bool, ions: bool) -> Vec<f32> {
        let a = (frame.max(0.) as usize).min(self.frames - 2);
        let b = a + 1;
        let t = (frame - a as f32).clamp(0., 1.);
        let origin = (self.length - self.extent) * 0.5;
        let mut out = Vec::with_capacity(count.min(2 * self.pool) * 4);
        let species_count = usize::from(electrons) + usize::from(ions);
        if species_count == 0 {
            return out;
        }
        for species in 0..2 {
            if (species == 0 && !electrons) || (species == 1 && !ions) {
                continue;
            }
            let cap = count.min(2 * self.pool) / species_count;
            let mut shown = 0;
            for p in 0..self.pool {
                let j = species * self.pool + p;
                let mut xy = [0.; 2];
                for (d, v) in xy.iter_mut().enumerate() {
                    let x = self.positions[(a * 2 * self.pool + j) * 2 + d] as f32 / 65535.
                        * self.length;
                    let y = self.positions[(b * 2 * self.pool + j) * 2 + d] as f32 / 65535.
                        * self.length;
                    *v = periodic_mix(x, y, t, self.length) - origin;
                }
                if xy.iter().all(|&v| v >= 0. && v < self.extent) {
                    out.extend([
                        xy[0] / self.extent,
                        xy[1] / self.extent,
                        species as f32,
                        j as f32,
                    ]);
                    shown += 1;
                    if shown >= cap {
                        break;
                    }
                }
            }
        }
        out
    }

    pub fn field(&self, frame: f32) -> Vec<f32> {
        let a = (frame.max(0.) as usize).min(self.frames - 2);
        let t = (frame - a as f32).clamp(0., 1.);
        let nn = self.grid * self.grid;
        (0..nn)
            .map(|j| {
                let x = self.potential[a * nn + j] as f32;
                let y = self.potential[(a + 1) * nn + j] as f32;
                ((1. - t) * x + t * y) * self.phi_scale / 32767.
            })
            .collect()
    }
}

#[wasm_bindgen]
pub struct Welch {
    n: usize,
    dt: f64,
    window: Vec<f32>,
    norm: f64,
    fft: Arc<dyn Fft<f32>>,
    sum: Vec<f64>,
    segments: usize,
}

#[wasm_bindgen]
impl Welch {
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize, dt: f64) -> Result<Welch, JsValue> {
        if !n.is_power_of_two() || !(64..=65536).contains(&n) || !dt.is_finite() || dt <= 0. {
            return Err(JsValue::from_str("Invalid FFT window"));
        }
        let window: Vec<_> = (0..n)
            .map(|j| (0.5 - 0.5 * (2. * std::f64::consts::PI * j as f64 / n as f64).cos()) as f32)
            .collect();
        let norm = window.iter().map(|&v| (v as f64).powi(2)).sum();
        Ok(Self {
            n,
            dt,
            window,
            norm,
            fft: FftPlanner::new().plan_fft_forward(n),
            sum: vec![0.; n],
            segments: 0,
        })
    }
    /// Interleaved real/imag voltage; no demeaning, zero padding, or fabricated samples.
    pub fn add_window(&mut self, signal: &[f32]) -> bool {
        if signal.len() != 2 * self.n || signal.iter().any(|v| !v.is_finite()) {
            return false;
        }
        let mut a: Vec<_> = signal
            .as_chunks::<2>()
            .0
            .iter()
            .zip(&self.window)
            .map(|(z, &w)| Complex32::new(z[0] * w, z[1] * w))
            .collect();
        self.fft.process(&mut a);
        for (s, z) in self.sum.iter_mut().zip(a) {
            *s += z.norm_sqr() as f64 * self.dt / self.norm;
        }
        self.segments += 1;
        true
    }
    pub fn spectrum(&self) -> Vec<f64> {
        (0..self.n)
            .map(|j| self.sum[(j + self.n / 2) % self.n] / self.segments.max(1) as f64)
            .collect()
    }
    pub fn segments(&self) -> usize {
        self.segments
    }
}

/// Zero-phase, finite Gaussian FIR. Cutoff is the -3 dB power frequency.
/// Linear FFT convolution, not circular filtering; unsupported edges are NaN.
#[wasm_bindgen]
pub fn ion_lowpass(input: &[f32], dt: f64, cutoff_hz: f64) -> Result<Vec<f32>, JsValue> {
    if input.len()%2 != 0 || !dt.is_finite() || dt<=0. || !cutoff_hz.is_finite() || cutoff_hz<=0. || cutoff_hz>=0.5/dt {
        return Err(JsValue::from_str("Invalid low-pass parameters"));
    }
    let n=input.len()/2;
    let sigma=std::f64::consts::LN_2.sqrt()/(2.*std::f64::consts::PI*cutoff_hz);
    let radius=(6.*sigma/dt).ceil() as usize;
    if n<=2*radius { return Err(JsValue::from_str("Record too short for low-pass support")); }
    let size=(n+2*radius).next_power_of_two();
    let mut a=vec![rustfft::num_complex::Complex32::new(0.,0.);size];
    let mut h=a.clone();
    for j in 0..n { a[j]=rustfft::num_complex::Complex32::new(input[2*j],input[2*j+1]); }
    let mut total=0.;
    for j in 0..=2*radius {
        let x=(j as f64-radius as f64)*dt/sigma;
        h[j].re=(-0.5*x*x).exp() as f32;total+=h[j].re;
    }
    for v in &mut h { v.re/=total; }
    let mut planner=rustfft::FftPlanner::<f32>::new();
    let forward=planner.plan_fft_forward(size);forward.process(&mut a);forward.process(&mut h);
    for j in 0..size { a[j]*=h[j]; }
    planner.plan_fft_inverse(size).process(&mut a);
    let mut out=vec![f32::NAN;input.len()];
    for j in radius..n-radius { let v=a[j+radius]/size as f32;out[2*j]=v.re;out[2*j+1]=v.im; }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lowpass_rejects_electron_tone_without_phase_shift_or_edge_padding() {
        let n=16384;let dt=1e-7;let fc: f64=30000.;
        for (frequency,expected) in [(10000.,2f64.powf(-0.5*(10000./fc).powi(2))),(1000000.,0.)] {
            let input:Vec<f32>=(0..n).flat_map(|j| {let t=2.*std::f64::consts::PI*frequency*j as f64*dt;[t.cos() as f32,t.sin() as f32]}).collect();
            let y=ion_lowpass(&input,dt,fc).unwrap();
            assert!(y[0].is_nan() && y[y.len()-1].is_nan());
            for j in 1000..n-1000 { assert!((y[2*j] as f64-input[2*j] as f64*expected).abs()<2e-5);assert!((y[2*j+1] as f64-input[2*j+1] as f64*expected).abs()<2e-5); }
        }
    }
    #[test]
    fn periodic_short_path() {
        assert!(
            periodic_mix(21.9, 0.1, 0.5, 22.).min((periodic_mix(21.9, 0.1, 0.5, 22.) - 22.).abs())
                < 1e-5
        );
        assert!((periodic_mix(2., 4., 0.25, 22.) - 2.5).abs() < 1e-6);
    }
    #[test]
    fn fft_complex_tone_and_parseval() {
        let n = 1024;
        let dt = 1e-6;
        let input: Vec<_> = (0..n)
            .flat_map(|j| {
                let a = 2. * std::f32::consts::PI * 31. * j as f32 / n as f32;
                [a.cos(), a.sin()]
            })
            .collect();
        let mut w = Welch::new(n, dt).unwrap();
        assert!(w.add_window(&input));
        let p = w.spectrum();
        let peak = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(peak, n / 2 + 31);
        assert!((p.iter().sum::<f64>() / (n as f64 * dt) - 1.).abs() < 1e-5);
    }
    #[test]
    fn slider_changes_only_display() {
        let pool = 100;
        let positions = vec![32768u16; 2 * 4 * pool];
        let field = vec![1i16; 2 * 16 * 16];
        let r = Replay::new(&positions, &field, pool, 16, 22., 10., 1.).unwrap();
        let a = r.points(0.4, 100, true, true);
        let b = r.points(0.4, 160, true, true);
        assert_eq!(a.len(), 400);
        assert_eq!(b.len(), 640);
        assert_eq!(a[..200], b[..200]);
        assert_eq!(r.field(0.4), r.field(0.4));
    }
}
