// CIC deposit/gather, spectral periodic Poisson, and leapfrog particle push.
// float32 throughout the tight loop. No fast-math flags are required.
struct Pair { float x,y; };
extern "C" __global__ void clear_grid(float* g,int ng) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<2*ng)g[j]=0.f;
}
extern "C" __global__ void deposit(const float* xe,const float* xi,float* g,int n,int ng,float length) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
    float w=(float)ng/n;
    float u=xe[p]/length*ng;int j=(int)floorf(u);float f=u-floorf(u);j=(j%ng+ng)%ng;
    atomicAdd(g+j,w*(1-f));atomicAdd(g+(j+1)%ng,w*f);
    atomicAdd(g+ng+j,-w*(1-f));atomicAdd(g+ng+(j+1)%ng,-w*f);
    u=xi[p]/length*ng;j=(int)floorf(u);f=u-floorf(u);j=(j%ng+ng)%ng;
    atomicAdd(g+ng+j,w*(1-f));atomicAdd(g+ng+(j+1)%ng,w*f);
}
extern "C" __global__ void poisson(const Pair* f,Pair* out,Pair* obs,int ng,float length,int mode,float norm,int record) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;int nf=ng/2+1;if(j>=nf)return;
    if(j==0 && record>=0){obs[record].x=norm*f[mode].x;obs[record].y=norm*f[mode].y;}
    if(j==0 || j==ng/2){out[j].x=0;out[j].y=0;return;}
    float k=6.2831853071795864769f*j/length;
    // Inverse cuFFT is unnormalized; include 1/ng here.
    out[j].x=f[nf+j].y/(k*ng);out[j].y=-f[nf+j].x/(k*ng);
}
__device__ float gather(const float* e,float x,int ng,float length) {
    float u=x/length*ng;int j=(int)floorf(u);float f=u-floorf(u);j=(j%ng+ng)%ng;
    return e[j]*(1-f)+e[(j+1)%ng]*f;
}
__device__ float wrap(float x,float length){return x-floorf(x/length)*length;}
extern "C" __global__ void push(float* xe,float* ve,float* xi,float* vi,const float* e,int n,int ng,float length,float mu,float kick,float drift) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
    ve[p]-=kick*gather(e,xe[p],ng,length);vi[p]+=kick*gather(e,xi[p],ng,length)/mu;
    xe[p]=wrap(xe[p]+drift*ve[p],length);xi[p]=wrap(xi[p]+drift*vi[p],length);
}
