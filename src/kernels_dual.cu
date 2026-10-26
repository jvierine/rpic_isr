// Appended to kernels_2d.cu. Packed real density/field halves mesh traffic.
__device__ void deposit_component(Pair* g,Pair x,int ng,float length,float w,int component) {
 float u=x.x/length*ng,v=x.y/length*ng;int i=(int)floorf(u+.5f),j=(int)floorf(v+.5f);
 float a=u-i,b=v-j;
 float wx[3]={.5f*(.5f-a)*(.5f-a),.75f-a*a,.5f*(.5f+a)*(.5f+a)};
 float wy[3]={.5f*(.5f-b)*(.5f-b),.75f-b*b,.5f*(.5f+b)*(.5f+b)};
 #pragma unroll
 for(int y=0;y<3;y++) {
 #pragma unroll
 for(int x=0;x<3;x++){int z=((j+y-1+ng)&(ng-1))*ng+((i+x-1+ng)&(ng-1));float* p=component?&g[z].y:&g[z].x;atomicAdd(p,w*wx[x]*wy[y]);}}
}
extern "C" __global__ void deposit_shard(const Pair* xe,const Pair* xi,Pair* g,int n,int total,int ng,float length){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
 float w=(float)(ng*ng)/total;
 deposit_component(g,xe[p],ng,length,w,0);deposit_component(g,xi[p],ng,length,w,1);
}
extern "C" __global__ void combine_density(const Pair* a,const Pair* b,Pair* out,int nn){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nn)return;
 out[j]={a[j].x+b[j].x,0};out[nn+j]={a[j].y+b[j].y,0};
}
extern "C" __global__ void pack_field(const Pair* e,Pair* packed,int nn){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<nn)packed[j]={e[j].x,e[nn+j].x};
}
__device__ Pair gather_uv(const Pair* __restrict__ e,float u,float v,int ng){
 int i=(int)floorf(u+.5f),j=(int)floorf(v+.5f);
 float a=u-i,b=v-j;float wx[3]={.5f*(.5f-a)*(.5f-a),.75f-a*a,.5f*(.5f+a)*(.5f+a)};
 float wy[3]={.5f*(.5f-b)*(.5f-b),.75f-b*b,.5f*(.5f+b)*(.5f+b)};Pair r={0,0};
 #pragma unroll
 for(int y=0;y<3;y++) {
 #pragma unroll
 for(int x=0;x<3;x++){int z=((j+y-1+ng)&(ng-1))*ng+((i+x-1+ng)&(ng-1));Pair f=e[z];float w=wx[x]*wy[y];r.x+=f.x*w;r.y+=f.y*w;}}
 return r;
}
__device__ Pair gather_packed(const Pair* __restrict__ e,Pair x,int ng,float length){return gather_uv(e,x.x/length*ng,x.y/length*ng,ng);}
extern "C" {__constant__ float rpic_invcell;}
// Almost all stable PIC pushes remain inside the box. Avoid a floating divide
// for wrapping these particles; retain the general fallback for large jumps.
__device__ float wrap_packed(float x,float length){
 if(x>=0.f&&x<length)return x;
 if(x<0.f&&x>=-length)return x+length;
 if(x>=length&&x<2.f*length)return x-length;
 return wrap2(x,length);
}
__device__ void packed_move(Pair* x,Pair* v,const Pair* e,int ng,float length,float qm,float kick,float drift){
 Pair f=gather_packed(e,*x,ng,length);v->x+=kick*qm*f.x;v->y+=kick*qm*f.y;
 x->x=wrap_packed(x->x+drift*v->x,length);x->y=wrap_packed(x->y+drift*v->y,length);
}
extern "C" __global__ void push_packed(Pair* xe,Pair* ve,Pair* xi,Pair* vi,const Pair* e,int n,int ng,float length,float mu,float kick,float drift){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
 packed_move(xe+p,ve+p,e,ng,length,-1,kick,drift);packed_move(xi+p,vi+p,e,ng,length,1/mu,kick,drift);
}
extern "C" __global__ void push_split(Pair* xe,Pair* ve,Pair* xi,Pair* vi,const Pair* e,int n,int ng,float length,float mu,float kick,float drift){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=2*n)return;int p=j%n;
 if(j<n)packed_move(xe+p,ve+p,e,ng,length,-1,kick,drift);else packed_move(xi+p,vi+p,e,ng,length,1/mu,kick,drift);
}
extern "C" __global__ void push_scaled(Pair* xe,Pair* ve,Pair* xi,Pair* vi,const Pair* e,int n,int ng,float length,float mu,float kick,float drift){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=2*n)return;int p=j%n,s=j>=n;Pair* x=s?xi+p:xe+p;Pair* v=s?vi+p:ve+p;
 Pair f=gather_uv(e,x->x*rpic_invcell,x->y*rpic_invcell,ng);float q=s?1/mu:-1;
 v->x+=kick*q*f.x;v->y+=kick*q*f.y;x->x=wrap_packed(x->x+drift*v->x,length);x->y=wrap_packed(x->y+drift*v->y,length);
}
// Integer-time moments from leapfrog velocities. Double block reductions only.
// For each species: vx,vy,vx²,vy²,vx*vy,(vx²+vy²)²; then field² and charge.
extern "C" __global__ void moments_packed(const Pair* xe,const Pair* ve,const Pair* xi,const Pair* vi,const Pair* e,const Pair* density,double* blocks,int n,int ng,float length,float mu,float dt,int include_field){
 __shared__ double shared[14][256];int t=threadIdx.x,p=blockIdx.x*blockDim.x+t,nn=ng*ng;
 for(int k=0;k<14;k++)shared[k][t]=0;
 if(p<n){for(int s=0;s<2;s++){Pair x=s?xi[p]:xe[p],v=s?vi[p]:ve[p],f=gather_packed(e,x,ng,length);double q=s?1.0/mu:-1.;double vx=(double)v.x+.5*dt*q*f.x,vy=(double)v.y+.5*dt*q*f.y;
  int k=6*s;double r2=vx*vx+vy*vy;shared[k][t]=vx;shared[k+1][t]=vy;shared[k+2][t]=vx*vx;shared[k+3][t]=vy*vy;shared[k+4][t]=vx*vy;shared[k+5][t]=r2*r2;}}
 if(include_field&&p<nn){Pair f=e[p];shared[12][t]=(double)f.x*f.x+(double)f.y*f.y;shared[13][t]=(double)density[nn+p].x-density[p].x;}
 __syncthreads();for(int d=128;d;d/=2){if(t<d)for(int k=0;k<14;k++)shared[k][t]+=shared[k][t+d];__syncthreads();}
 if(t==0)for(int k=0;k<14;k++)blocks[blockIdx.x*14+k]=shared[k][0];
}
extern "C" __global__ void reduce_moments(const double* blocks,double* out,int n){
 __shared__ double shared[14][256];int t=threadIdx.x;for(int k=0;k<14;k++){double x=0;for(int j=t;j<n;j+=256)x+=blocks[j*14+k];shared[k][t]=x;}
 __syncthreads();for(int d=128;d;d/=2){if(t<d)for(int k=0;k<14;k++)shared[k][t]+=shared[k][t+d];__syncthreads();}
 if(t==0)for(int k=0;k<14;k++)out[k]=shared[k][0];
}
// Rescale central INTEGER-time thermal velocities, preserving species mean flow.
// Return to staggered representation after applying the thermostat.
extern "C" __global__ void thermostat_packed(const Pair* xe,Pair* ve,const Pair* xi,Pair* vi,const Pair* e,int n,int ng,float length,float mu,float dt,double ux_e,double uy_e,double ux_i,double uy_i,double ae,double ai){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
 for(int s=0;s<2;s++){Pair x=s?xi[p]:xe[p],v=s?vi[p]:ve[p],f=gather_packed(e,x,ng,length);double q=s?1.0/mu:-1.,ux=s?ux_i:ux_e,uy=s?uy_i:uy_e,a=s?ai:ae;
  v.x=(float)(ux+a*((double)v.x+.5*dt*q*f.x-ux)-.5*dt*q*f.x);v.y=(float)(uy+a*((double)v.y+.5*dt*q*f.y-uy)-.5*dt*q*f.y);if(s)vi[p]=v;else ve[p]=v;}
}
extern "C" __global__ void sample_packed(const Pair* xe,const Pair* ve,const Pair* xi,const Pair* vi,const Pair* e,Pair* out,int ng,float length,float mu,float dt,float lambda,float vte){
 int j=threadIdx.x;if(j>=32)return;int s=j/16,p=j%16;Pair x=s?xi[p]:xe[p],v=s?vi[p]:ve[p],f=gather_packed(e,x,ng,length);float q=s?1/mu:-1;
 out[j]={x.x*lambda,x.y*lambda};out[32+j]={(v.x+.5f*dt*q*f.x)*vte,(v.y+.5f*dt*q*f.y)*vte};
}
extern "C" __global__ void crop_phi(const Pair* native,float* out,int ng,int crop,int start){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<crop*crop)out[j]=native[(start+j/crop)*ng+start+j%crop].x;
}
extern "C" __global__ void record_mode(const Pair* freq,Pair* out,int mode,int sample,float norm){if(threadIdx.x==0){out[sample]={norm*freq[mode].x,norm*freq[mode].y};}}

// Tile bins contain original particle indices: stable trajectory IDs never change.
__device__ int tile_of(Pair x,int ng,float length){
 int i=(int)floorf(x.x/length*ng+.5f),j=(int)floorf(x.y/length*ng+.5f);
 return ((j&(ng-1))/8)*(ng/8)+(i&(ng-1))/8;
}
extern "C" __global__ void clear_counts(unsigned* counts,int n){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)counts[j]=0;}
extern "C" __global__ void count_tiles(const Pair* xe,const Pair* xi,unsigned* counts,int n,int ng,float length){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;int tiles=(ng/8)*(ng/8);
 atomicAdd(counts+tile_of(xe[p],ng,length),1u);atomicAdd(counts+tiles+tile_of(xi[p],ng,length),1u);
}
extern "C" __global__ void prefix_tiles(const unsigned* counts,unsigned* starts,unsigned* cursor,int tiles,int n){
 int s=threadIdx.x;if(s>=2)return;unsigned sum=s*n;
 for(int j=0;j<tiles;j++){int k=s*tiles+j;starts[k]=sum;cursor[k]=sum;sum+=counts[k];}
}
extern "C" __global__ void scatter_tiles(const Pair* xe,const Pair* xi,unsigned* cursor,unsigned* indices,int n,int ng,float length){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;int tiles=(ng/8)*(ng/8);
 unsigned a=atomicAdd(cursor+tile_of(xe[p],ng,length),1u),b=atomicAdd(cursor+tiles+tile_of(xi[p],ng,length),1u);indices[a]=p;indices[b]=p;
}
extern "C" __global__ void deposit_tiled(const Pair* xe,const Pair* xi,Pair* density,const unsigned* counts,const unsigned* starts,const unsigned* indices,int ng,float length,float w){
 __shared__ float rho[100];int tiles=(ng/8)*(ng/8),s=blockIdx.x/tiles,tile=blockIdx.x%tiles,t=threadIdx.x;
 if(t<100)rho[t]=0;__syncthreads();
 int ox=(tile%(ng/8))*8,oy=(tile/(ng/8))*8;
 unsigned begin=starts[blockIdx.x],end=begin+counts[blockIdx.x];
 for(unsigned p=begin+t;p<end;p+=blockDim.x){Pair v=s?xi[indices[p]]:xe[indices[p]];
  float u=v.x/length*ng,z=v.y/length*ng;int i=(int)floorf(u+.5f),j=(int)floorf(z+.5f);float a=u-i,b=z-j;
  int lx=(i&(ng-1))-ox+1,ly=(j&(ng-1))-oy+1;
  // Bins may be reused for several steps; particles leaving their original tile
  // deposit globally rather than being lost or written outside shared memory.
  if(lx<1||lx>8||ly<1||ly>8){deposit_component(density,v,ng,length,w,s);continue;}
  float wx[3]={.5f*(.5f-a)*(.5f-a),.75f-a*a,.5f*(.5f+a)*(.5f+a)},wy[3]={.5f*(.5f-b)*(.5f-b),.75f-b*b,.5f*(.5f+b)*(.5f+b)};
  #pragma unroll
  for(int y=0;y<3;y++){
   #pragma unroll
   for(int x=0;x<3;x++)atomicAdd(rho+(ly+y-1)*10+lx+x-1,w*wx[x]*wy[y]);}
 }
 __syncthreads();if(t<100){int x=(ox+t%10-1+ng)&(ng-1),y=(oy+t/10-1+ng)&(ng-1);float* out=s?&density[y*ng+x].y:&density[y*ng+x].x;atomicAdd(out,rho[t]);}
}
