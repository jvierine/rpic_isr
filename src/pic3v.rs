//! 2D space, three velocities, arbitrary uniform B. Equal-weight core/tail labels
//! are fixed particle identities, not an energy threshold that changes with time.
use crate::*;
use crate::pic2d::{gather, Solver2};
use std::{ops::Deref,time::Instant};

#[derive(Parser,Debug,Clone,Serialize,Deserialize)]
pub struct Setup {
    #[command(flatten)]
    #[serde(flatten)]
    pub base: Config,
    #[arg(long,default_value_t=0.)]
    pub bx_t: f64,
    #[arg(long,default_value_t=0.)]
    pub by_t: f64,
    /// Fraction of electrons assigned to a zero-drift isotropic hot Maxwellian.
    #[arg(long,default_value_t=0.)]
    pub hot_fraction: f64,
    /// Mean kinetic energy of hot electrons in eV; temperature is 2E/3.
    #[arg(long,default_value_t=20.)]
    pub hot_energy_ev: f64,
    /// Core-only weak rescaling time in microseconds, burn-in only. Zero disables.
    #[arg(long,default_value_t=0.)]
    pub core_thermostat_us: f64,
}
impl Deref for Setup { type Target=Config; fn deref(&self)->&Config {&self.base} }
impl Setup {
    pub fn validate(&self)->Result<()> {
        anyhow::ensure!([self.bx_t,self.by_t,self.hot_fraction,self.hot_energy_ev,self.core_thermostat_us].iter().all(|v|v.is_finite()),"Nonfinite 3V parameters");
        anyhow::ensure!((0.0..0.5).contains(&self.hot_fraction)&&self.hot_energy_ev>0.&&self.core_thermostat_us>=0.,"Invalid tail/thermostat parameters");
        anyhow::ensure!(self.gpu_batch_size==1,"3V diagnostic populations currently require one seed at a time");
        anyhow::ensure!(self.hot_fraction==0.||self.hot_count(self.cells*self.cells*self.ppc)>=32,"Tail requires at least 32 macroparticles");
        Ok(())
    }
    pub fn hot_count(&self,n:usize)->usize {(self.hot_fraction*n as f64).round() as usize}
}
pub fn magnetic(c:&Setup,s:&Scales)->[f32;3] {
    [c.bx_t,c.by_t,c.bz_t].map(|b|(QE*b/(ME*s.omega_pe)) as f32)
}
fn cross(a:[f32;3],b:[f32;3])->[f32;3] {[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
pub fn boris(v:[f32;3],e:[f32;2],qm:f32,dt:f32,b:[f32;3])->[f32;3] {
    let h=0.5*dt*qm;let u=[v[0]+h*e[0],v[1]+h*e[1],v[2]];
    let t=b.map(|v|h*v);let fac=2./(1.+t.iter().map(|v|v*v).sum::<f32>());
    let ux=cross(u,t);let prime=std::array::from_fn(|j|u[j]+ux[j]);
    let rot=cross(prime,t.map(|v|fac*v));
    [u[0]+rot[0]+h*e[0],u[1]+rot[1]+h*e[1],u[2]+rot[2]]
}
#[derive(Clone,Default)]
pub struct Particles3 {pub xe:Vec<[f32;2]>,pub xi:Vec<[f32;2]>,pub ve:Vec<[f32;3]>,pub vi:Vec<[f32;3]>}
impl Particles3 {
    pub fn new(c:&Setup,s:&Scales,seed:u64)->Self {
        let n=c.cells*c.cells*c.ppc;let mut rng=StdRng::seed_from_u64(seed);
        let mut pos=|| (0..n).map(|_|[rng.random::<f32>()*s.length as f32,rng.random::<f32>()*s.length as f32]).collect();
        let xe=pos();let xi=pos();
        let mut vel=|| (0..n).map(|_|std::array::from_fn(|_|StandardNormal.sample(&mut rng))).collect::<Vec<[f32;3]>>();
        let mut ve=vel();let mut vi=vel();let nh=c.hot_count(n);
        fn normalize(v:&mut [[f32;3]],sigma:f64) {
            if v.is_empty(){return;}
            for d in 0..3 {let mean=v.iter().map(|v|v[d] as f64).sum::<f64>()/v.len() as f64;
                let var=v.iter().map(|v|(v[d] as f64-mean).powi(2)).sum::<f64>()/v.len() as f64;
                for v in v.iter_mut(){v[d]=((v[d] as f64-mean)*sigma/var.sqrt()) as f32;}}
        }
        normalize(&mut ve[nh..],1.);
        normalize(&mut ve[..nh],(2.*c.hot_energy_ev*QE/(3.*KB*c.te)).sqrt());
        normalize(&mut vi,(c.ti/c.te/s.mass_ratio).sqrt());
        Self{xe,xi,ve,vi}
    }
}
pub fn push(p:&mut Particles3,e:&[[f32;2]],c:&Setup,s:&Scales,kick:f32,drift:f32) {
    for(x,v,qm) in [(&mut p.xe,&mut p.ve,-1.),(&mut p.xi,&mut p.vi,1./s.mass_ratio as f32)] {
        for(x,v) in x.iter_mut().zip(v){*v=boris(*v,gather(e,*x,c.cells,s.length as f32),qm,kick,magnetic(c,s));
            for d in 0..2{x[d]=(x[d]+drift*v[d]).rem_euclid(s.length as f32);}}
    }
}
// All energies normalized by N_e kB T_core, powers by N_e kB T_core omega_pe.
// Columns: t, core_K, tail_K, ion_K, core_T, tail_T, ion_T,
// core_JE, tail_JE, ion_JE, cumulative_thermostat_energy_added.
pub fn diagnostic3(run:&mut Run,p:&Particles3,e:&[[f32;2]],c:&Setup,s:&Scales,step:usize,charge:f64,work:f64) {
    let n=p.xe.len();let nh=c.hot_count(n);let b=magnetic(c,s);
    let norm=b.iter().map(|v|v*v).sum::<f32>().sqrt();let unit=if norm>0.{b.map(|v|v/norm)}else{[0.,0.,1.]};
    let mut temperatures=[0.;6];
    let mut energy=[0.;3];let mut temp=[0.;3];let mut power=[0.;3];let mut momentum=0.;
    for(g,x,v,m,q) in [(0,&p.xe[nh..],&p.ve[nh..],1.,-1.),(1,&p.xe[..nh],&p.ve[..nh],1.,-1.),(2,&p.xi[..],&p.vi[..],s.mass_ratio,1.)] {
        if x.is_empty(){continue;}
        let mut mean=[0.;3];let mut vv=0.;let mut vpar2=0.;
        for(&x,&v) in x.iter().zip(v){let ef=gather(e,x,c.cells,s.length as f32);let v=boris(v,ef,(q/m) as f32,0.5*c.dt as f32,b);
            for d in 0..3{mean[d]+=v[d] as f64;vv+=(v[d] as f64).powi(2);}
            vpar2+=(0..3).map(|d|v[d] as f64*unit[d] as f64).sum::<f64>().powi(2);
            power[g]+=q*(v[0] as f64*ef[0] as f64+v[1] as f64*ef[1] as f64)/n as f64;
        }
        energy[g]=0.5*m*vv/n as f64;
        temp[g]=m*c.te/3.*(vv/x.len() as f64-mean.iter().map(|v|(v/x.len() as f64).powi(2)).sum::<f64>());
        let meanpar=(0..3).map(|d|mean[d]/x.len() as f64*unit[d] as f64).sum::<f64>();
        temperatures[2*g]=m*c.te*(vpar2/x.len() as f64-meanpar*meanpar);
        temperatures[2*g+1]=(3.*temp[g]-temperatures[2*g])/2.;
        momentum+=m*mean[0]/n as f64;
    }
    let field=0.5*e.iter().map(|v|(v[0] as f64).powi(2)+(v[1] as f64).powi(2)).sum::<f64>()/e.len() as f64;
    let total=energy.iter().sum::<f64>()+field;
    run.diagnostics.extend([step as f64*s.dt_s,total,energy[0]+energy[1],energy[2],field,(temp[0]*(n-nh) as f64+temp[1]*nh as f64)/n as f64,temp[2],momentum,charge]);
    run.interaction_history.extend([step as f64*s.dt_s,energy[0],energy[1],energy[2],temp[0],temp[1],temp[2],power[0],power[1],power[2],work]);
    run.directional_temperatures.extend(temperatures);
    for(x,v,qm) in [(&p.xe,&p.ve,-1.),(&p.xi,&p.vi,1./s.mass_ratio as f32)] {
        for(&x,&v) in x.iter().zip(v).take(16){run.trajectory_x.extend(x.map(|x|(x as f64*s.lambda_de) as f32));
            run.trajectory_v.extend(boris(v,gather(e,x,c.cells,s.length as f32),qm,0.5*c.dt as f32,b).map(|v|(v as f64*s.vte) as f32));}
    }
}
pub fn thermostat(p:&mut Particles3,e:&[[f32;2]],c:&Setup,s:&Scales,step:usize)->f64 {
    if c.core_thermostat_us==0.||step==0||step>=s.burn_steps{return 0.;}
    let nh=c.hot_count(p.xe.len());let n=p.xe.len();let b=magnetic(c,s);
    let mut center:Vec<_>=p.xe[nh..].iter().zip(&p.ve[nh..]).map(|(&x,&v)|boris(v,gather(e,x,c.cells,s.length as f32),-1.,0.5*c.dt as f32,b)).collect();
    let mean:[f64;3]=std::array::from_fn(|d|center.iter().map(|v|v[d] as f64).sum::<f64>()/center.len() as f64);
    let vv=center.iter().map(|v|(0..3).map(|d|(v[d] as f64-mean[d]).powi(2)).sum::<f64>()).sum::<f64>();
    let alpha=1.-(-(c.diagnostic_stride as f64)*s.dt_s/(c.core_thermostat_us*1e-6)).exp();
    let scale=(1.+alpha*(3.*center.len() as f64/vv-1.)).sqrt();let mut work=0.;
    for(j,v) in center.iter_mut().enumerate(){for d in 0..3{let old=v[d] as f64;v[d]=(mean[d]+scale*(old-mean[d])) as f32;work+=0.5*((v[d] as f64).powi(2)-old*old)/n as f64;}
        p.ve[j+nh]=boris(*v,gather(e,p.xe[j+nh],c.cells,s.length as f32),-1.,-0.5*c.dt as f32,b);}
    work
}
pub fn guard(run:&Run)->Result<()> {
    let a=&run.diagnostics;let first=a[1];let last=&a[a.len()-9..];let h=&run.interaction_history;
    let corrected=last[1]-h[h.len()-1];
    anyhow::ensure!(last.iter().all(|v|v.is_finite())&&((corrected-first)/first).abs()<0.01,"3V numerical energy guard exceeded 1%; thermostat work removed from balance");Ok(())
}
pub fn save_live(run:&Run,c:&Setup,s:&Scales)->Result<()> {
    let Ok(path)=std::env::var("RPIC_LIVE_PATH") else{return Ok(())};
    let f=hdf5::File::create(format!("{path}.tmp"))?;
    f.new_dataset::<f64>().shape([run.diagnostics.len()/9,9]).create("diagnostics")?.write_raw(&run.diagnostics)?;
    f.new_dataset::<f64>().shape([run.interaction_history.len()/11,11]).create("population_energy")?.write_raw(&run.interaction_history)?;
    f.new_dataset::<f64>().shape([run.directional_temperatures.len()/6,6]).create("parallel_perpendicular_temperatures_K")?.write_raw(&run.directional_temperatures)?;
    f.new_dataset_builder().with_data(&serde_json::to_vec(c)?).create("config_json")?;
    f.new_dataset_builder().with_data(&serde_json::to_vec(s)?).create("scales_json")?;
    f.flush()?;drop(f);std::fs::rename(format!("{path}.tmp"),path)?;Ok(())
}
pub fn simulate(c:&Setup,s:&Scales,seed:u64)->Result<Run>{
    let start=Instant::now();let mut p=Particles3::new(c,s,seed);let mut solver=Solver2::new(c.cells);let mut run=Run::default();let mut work=0.;
    for step in 0..=s.burn_steps+s.steps {let obs=solver.fields_positions(&p.xe,&p.xi,c,s);
        if step==0{push(&mut p,&solver.e,c,s,-0.5*c.dt as f32,0.);}
        if step>=s.burn_steps&&step<s.burn_steps+s.steps{run.voltage.push(obs);}
        if step%c.diagnostic_stride==0||step==s.burn_steps+s.steps{diagnostic3(&mut run,&p,&solver.e,c,s,step,solver.charge,work);guard(&run)?;work+=thermostat(&mut p,&solver.e,c,s,step);}
        if step==s.burn_steps+s.steps{break;}push(&mut p,&solver.e,c,s,c.dt as f32,c.dt as f32);
    }
    run.elapsed_s=start.elapsed().as_secs_f64();Ok(run)
}
#[cfg(test)] mod tests {
 use super::*;
 #[test]fn arbitrary_b_preserves_speed_parallel_and_reverses(){let b=[0.02,0.33,0.07];let v=[0.7,-0.3,1.2];let mut u=v;for _ in 0..20000{u=boris(u,[0.,0.],-1.,0.06,b);}let norm=|v:[f32;3]|v.iter().map(|x|x*x).sum::<f32>();assert!((norm(u)-norm(v)).abs()<0.001);assert!((0..3).map(|j|(u[j]-v[j])*b[j]).sum::<f32>().abs()<0.001);let back=boris(boris(v,[0.2,-0.1],-1.,0.06,b),[0.2,-0.1],-1.,-0.06,b);assert!((0..3).all(|d|(back[d]-v[d]).abs()<1e-6));}
 #[test]fn old_bz_limit_matches(){let v=[0.7,-0.3];let old=crate::pic2d::boris(v,[0.2,-0.1],-1.,0.06,0.33);let new=boris([v[0],v[1],1.1],[0.2,-0.1],-1.,0.06,[0.,0.,0.33]);assert!((0..2).all(|d|(old[d]-new[d]).abs()<1e-6));assert_eq!(new[2],1.1);}
 #[test]fn population_targets(){let c=Setup::parse_from(["t","--cells","32","--ppc","4","--bragg-mode","1","--hot-fraction","0.1"]);let s=Scales::new(&c).unwrap();let p=Particles3::new(&c,&s,3);let mut run=Run::default();diagnostic3(&mut run,&p,&vec![[0.,0.];1024],&c,&s,0,0.,0.);let h=run.interaction_history;assert!((h[4]-c.te).abs()<0.01);assert!((h[5]-2.*20.*QE/(3.*KB)).abs()<0.1);assert!((h[6]-c.ti).abs()<0.01);}
 #[test]fn thermostat_only_core_during_burn_with_exact_ledger(){
 let c=Setup::parse_from(["t","--cells","32","--ppc","4","--bragg-mode","1","--hot-fraction","0.1","--core-thermostat-us","10"]);let s=Scales::new(&c).unwrap();let mut p=Particles3::new(&c,&s,3);let nh=c.hot_count(p.xe.len());for v in &mut p.ve[nh..]{for d in 0..3{v[d]*=1.1;}}let hot=p.ve[..nh].to_vec();let ions=p.vi.clone();let energy=|p:&Particles3|p.ve.iter().flat_map(|v|v.iter()).map(|v|0.5*(*v as f64).powi(2)).sum::<f64>()/p.ve.len() as f64;let before=energy(&p);let field=vec![[0.,0.];1024];let work=thermostat(&mut p,&field,&c,&s,c.diagnostic_stride);assert!(work<0.);assert!((energy(&p)-before-work).abs()<1e-10);assert_eq!(hot,p.ve[..nh]);assert_eq!(ions,p.vi);let prior=p.ve.clone();assert_eq!(thermostat(&mut p,&field,&c,&s,s.burn_steps),0.);assert_eq!(prior,p.ve);
 }
 #[test]fn prescribed_wave_resonance_selects_parallel_velocity(){
 let b=[0.1_f32,(0.99_f32).sqrt(),0.];let k=0.3;let omega=1.05;let dt=0.02;
 let measure=|vp:f32|{let mut variance=0.;for phase in 0..12{let theta=phase as f32*std::f32::consts::TAU/12.;let mut x=theta/k;let mut v=[vp*b[0],vp*b[1],1.];let initial=v.iter().map(|v|0.5*v*v).sum::<f32>();for step in 0..20000{let ef=0.0001*(k*x-omega*step as f32*dt).cos();v=boris(v,[ef,0.],-1.,dt,b);x+=dt*v[0];}let end=v.iter().map(|v|0.5*v*v).sum::<f32>();variance+=(end-initial).powi(2);}variance/12.};
 let resonant=(omega-1.)/(k*b[0]);let at=measure(resonant);let off=measure(resonant+2.);assert!(at>20.*off,"resonant {at}, detuned {off}");
 }
}
