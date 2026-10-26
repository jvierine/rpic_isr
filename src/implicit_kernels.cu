// Discrete-work 2D TSC potential gather; independent particle midpoint solves.
struct Pair {float x,y;};
struct DPair {double x,y;};
extern "C" __global__ void zero_i(Pair* grid,int nn,int* failures) {
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<nn){grid[j].x=0;grid[j].y=0;}if(j==0)*failures=0;
}
extern "C" __global__ void charge_i(const Pair* x,Pair* grid,int np,int ng,float length) {
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=2*np)return;
 float u=x[p].x*ng/length,v=x[p].y*ng/length;int i=(int)floorf(u+.5f),j=(int)floorf(v+.5f);
 float a=u-i,b=v-j,wx[3]={.5f*(.5f-a)*(.5f-a),.75f-a*a,.5f*(.5f+a)*(.5f+a)},wy[3]={.5f*(.5f-b)*(.5f-b),.75f-b*b,.5f*(.5f+b)*(.5f+b)};
 float w=(p<np?-1.f:1.f)*ng*ng/np;
 #pragma unroll
 for(int y=0;y<3;y++) {
 #pragma unroll
 for(int x=0;x<3;x++)atomicAdd(&grid[((j+y-1)&(ng-1))*ng+((i+x-1)&(ng-1))].x,w*wx[x]*wy[y]);}
}
extern "C" __global__ void potential_i(const Pair* freq,Pair* out,int ng,float length) {
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=ng*ng)return;int x=j%ng,y=j/ng;
 float kx=6.283185307179586f*(x<=ng/2?x:x-ng)/length,ky=6.283185307179586f*(y<=ng/2?y:y-ng)/length;
 float k2=kx*kx+ky*ky;float a=k2==0?0:1.f/(k2*ng*ng);out[j].x=freq[j].x*a;out[j].y=freq[j].y*a;
}
__device__ Pair grad_i(const Pair* phi,Pair pos,int ng,float length) {
 float u=pos.x*ng/length,v=pos.y*ng/length;int i=(int)floorf(u+.5f),j=(int)floorf(v+.5f);
 float a=u-i,b=v-j,wx[3]={.5f*(.5f-a)*(.5f-a),.75f-a*a,.5f*(.5f+a)*(.5f+a)},wy[3]={.5f*(.5f-b)*(.5f-b),.75f-b*b,.5f*(.5f+b)*(.5f+b)};
 float dx[3]={a-.5f,-2*a,a+.5f},dy[3]={b-.5f,-2*b,b+.5f};Pair e={0,0};
 #pragma unroll
 for(int y=0;y<3;y++) {
 #pragma unroll
 for(int x=0;x<3;x++){float q=phi[((j+y-1)&(ng-1))*ng+((i+x-1)&(ng-1))].x;e.x-=q*dx[x]*wy[y];e.y-=q*wx[x]*dy[y];}}
 e.x*=ng/length;e.y*=ng/length;return e;
}
__device__ Pair orbit_i(const Pair* phi,Pair x,Pair delta,int ng,float length,bool* valid) {
 float u=x.x*ng/length,v=x.y*ng/length,du=delta.x*ng/length,dv=delta.y*ng/length;
 float tx=du==0?1e30f:((du>0?floorf(u+.5f)+.5f:ceilf(u-.5f)-.5f)-u)/du;
 float ty=dv==0?1e30f:((dv>0?floorf(v+.5f)+.5f:ceilf(v-.5f)-.5f)-v)/dv;
 float sx=du==0?1e30f:fabsf(1.f/du),sy=dv==0?1e30f:fabsf(1.f/dv),t=0;Pair out={0,0};
 for(int piece=0;piece<64;piece++) {
  float end=fminf(1.f,fminf(tx,ty)),h=.5f*(end-t),center=.5f*(end+t);
  if(h>0){float a=center-h*.5773502691896258f,b=center+h*.5773502691896258f;
   Pair pa={x.x+a*delta.x,x.y+a*delta.y},pb={x.x+b*delta.x,x.y+b*delta.y};
   Pair ea=grad_i(phi,pa,ng,length),eb=grad_i(phi,pb,ng,length);out.x+=h*(ea.x+eb.x);out.y+=h*(ea.y+eb.y);}
  if(end>=1.f)return out;
  if(tx<=end+1e-7f)tx+=sx;if(ty<=end+1e-7f)ty+=sy;t=end;
 }
 *valid=false;return out;
}
extern "C" __global__ void midpoint_i(const Pair* x,const Pair* v,Pair* xn,Pair* vn,const Pair* mid,int np,int ng,float length,float mu,float dt,int particle_iterations,int* failures) {
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=2*np)return;float qm=p<np?-1.f:1.f/mu;Pair guess=v[p];bool converged=false,valid=true;
 for(int it=0;it<particle_iterations;it++) {
  Pair delta={.5f*dt*(v[p].x+guess.x),.5f*dt*(v[p].y+guess.y)};
  Pair e=orbit_i(mid,x[p],delta,ng,length,&valid),next={v[p].x+dt*qm*e.x,v[p].y+dt*qm*e.y};
  float error=fmaxf(fabsf(next.x-guess.x),fabsf(next.y-guess.y));guess=next;
  if(error<=2e-7f*(1.f+fmaxf(fabsf(next.x),fabsf(next.y)))){converged=true;break;}
 }
 if(!converged||!valid)atomicAdd(failures,1);
 vn[p]=guess;Pair next={x[p].x+.5f*dt*(v[p].x+guess.x),x[p].y+.5f*dt*(v[p].y+guess.y)};
 next.x-=floorf(next.x/length)*length;next.y-=floorf(next.y/length)*length;xn[p]=next;
}
extern "C" __global__ void residual_i(Pair* mid,const Pair* old,const Pair* next,DPair* partial,int nn,float relaxation) {
 __shared__ double rr[256],pp[256];int t=threadIdx.x,j=blockIdx.x*blockDim.x+t;double r=0,p=0;
 if(j<nn){float target=.5f*(old[j].x+next[j].x);r=(double)target-mid[j].x;p=next[j].x;mid[j].x+=relaxation*(float)r;mid[j].y=0;}
 rr[t]=r*r;pp[t]=p*p;__syncthreads();for(int offset=128;offset>0;offset/=2){if(t<offset){rr[t]+=rr[t+offset];pp[t]+=pp[t+offset];}__syncthreads();}
 if(t==0){partial[blockIdx.x].x=rr[0];partial[blockIdx.x].y=pp[0];}
}
