// Square, periodic, quadratic TSC 2D2V. Layout [batch][y][x].
struct Pair {float x,y;};
extern "C" __global__ void trace2(const Pair* xe,const Pair* xi,Pair* tracks,int n,int batches,int step,int frames,float lambda) {
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=32*batches)return;int b=j/32,p=j%32;
 Pair v=p<16?xe[b*n+p]:xi[b*n+p-16];v.x*=lambda;v.y*=lambda;tracks[(b*frames+step)*32+p]=v;
}
extern "C" __global__ void clear2(Pair* g,int count) {
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<count){g[j].x=0;g[j].y=0;}
}
__device__ void add2(Pair* g,Pair x,int ng,float length,float w) {
 float u=x.x/length*ng,v=x.y/length*ng;int i=(int)floorf(u+0.5f),j=(int)floorf(v+0.5f);
 float a=u-i,b=v-j;float wx[3]={0.5f*(0.5f-a)*(0.5f-a),0.75f-a*a,0.5f*(0.5f+a)*(0.5f+a)};
 float wy[3]={0.5f*(0.5f-b)*(0.5f-b),0.75f-b*b,0.5f*(0.5f+b)*(0.5f+b)};
 // ng is validated as a power of two; masks avoid integer divisions.
 #pragma unroll
 for(int y=0;y<3;y++) {
 #pragma unroll
 for(int x=0;x<3;x++)atomicAdd(&g[((j+y-1+ng)&(ng-1))*ng+((i+x-1+ng)&(ng-1))].x,w*wx[x]*wy[y]);}
}
extern "C" __global__ void deposit2(const Pair* xe,const Pair* xi,Pair* g,int n,int ng,int batches,float length) {
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n*batches)return;
 int nn=ng*ng,b=p/n;Pair* ne=g+2*b*nn;Pair* rho=ne+nn;float w=(float)nn/n;
 add2(ne,xe[p],ng,length,w);add2(rho,xi[p],ng,length,w);
}
extern "C" __global__ void poisson2(const Pair* all,Pair* field,Pair* obs,int ng,int batches,float length,int mode,float norm,int record,int steps) {
 int z=blockIdx.x*blockDim.x+threadIdx.x,nn=ng*ng;if(z>=nn*batches)return;
 int b=z/nn,j=z%nn,x=j%ng,y=j/ng;const Pair* f=all+2*b*nn;
 if(j==0&&record>=0){obs[b*steps+record].x=norm*f[mode].x;obs[b*steps+record].y=norm*f[mode].y;}
 float kx=6.2831853071795864769f*(x<=ng/2?x:x-ng)/length;
 float ky=6.2831853071795864769f*(y<=ng/2?y:y-ng)/length;float k2=kx*kx+ky*ky;
 float ax=(k2==0||x==ng/2)?0:kx/(k2*nn),ay=(k2==0||y==ng/2)?0:ky/(k2*nn);
 Pair r={f[nn+j].x-f[j].x,f[nn+j].y-f[j].y};field[2*b*nn+j].x=r.y*ax;field[2*b*nn+j].y=-r.x*ax;
 field[(2*b+1)*nn+j].x=r.y*ay;field[(2*b+1)*nn+j].y=-r.x*ay;
}
__device__ Pair gather2(const Pair* e,Pair x,int ng,float length) {
 float u=x.x/length*ng,v=x.y/length*ng;int i=(int)floorf(u+0.5f),j=(int)floorf(v+0.5f);
 float a=u-i,b=v-j;float wx[3]={0.5f*(0.5f-a)*(0.5f-a),0.75f-a*a,0.5f*(0.5f+a)*(0.5f+a)};
 float wy[3]={0.5f*(0.5f-b)*(0.5f-b),0.75f-b*b,0.5f*(0.5f+b)*(0.5f+b)};
 Pair out={0,0};
 #pragma unroll
 for(int y=0;y<3;y++) {
 #pragma unroll
 for(int x=0;x<3;x++){int z=((j+y-1+ng)&(ng-1))*ng+((i+x-1+ng)&(ng-1));
 out.x+=e[z].x*wx[x]*wy[y];out.y+=e[ng*ng+z].x*wx[x]*wy[y];}}
 return out;
}
__device__ float wrap2(float x,float l){return x-floorf(x/l)*l;}
__device__ void move2(Pair* x,Pair* v,const Pair* e,int ng,float length,float qm,float kick,float drift) {
 Pair field=gather2(e,*x,ng,length);
 v->x+=kick*qm*field.x;v->y+=kick*qm*field.y;
 x->x=wrap2(x->x+drift*v->x,length);x->y=wrap2(x->y+drift*v->y,length);
}
extern "C" __global__ void push2(Pair* xe,Pair* ve,Pair* xi,Pair* vi,const Pair* fields,int n,int ng,int batches,float length,float mu,float kick,float drift) {
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n*batches)return;const Pair* e=fields+2*(p/n)*ng*ng;
 move2(xe+p,ve+p,e,ng,length,-1,kick,drift);move2(xi+p,vi+p,e,ng,length,1/mu,kick,drift);
}
// Optional web recording: evenly indexed, fixed particle IDs, both species.
extern "C" __global__ void web_positions2(const Pair* xe,const Pair* xi,Pair* out,int n,int pool) {
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=2*pool)return;
 int p=(j%pool)*(n/pool);out[j]=j<pool?xe[p]:xi[p];
}
// Same endpoint charge and Fourier Poisson operator as the explicit mover.
extern "C" __global__ void web_potential2(const Pair* f,Pair* phi,int ng,float length) {
 int j=blockIdx.x*blockDim.x+threadIdx.x,nn=ng*ng;if(j>=nn)return;
 int x=j%ng,y=j/ng;float kx=6.283185307179586f*(x<=ng/2?x:x-ng)/length;
 float ky=6.283185307179586f*(y<=ng/2?y:y-ng)/length,k2=kx*kx+ky*ky;
 phi[j].x=k2==0?0:(f[nn+j].x-f[j].x)/(k2*nn);
 phi[j].y=k2==0?0:(f[nn+j].y-f[j].y)/(k2*nn);
}
