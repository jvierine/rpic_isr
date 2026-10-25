// Independent PIC realizations in disjoint grids. No cross-realization forces.
struct Pair {float x,y;};
extern "C" __global__ void clear_grid(float* g,int count) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<count)g[j]=0;
}
extern "C" __global__ void deposit(const float* xe,const float* xi,float* grid,int n,int ng,int batches,float length) {
    // Block-private CIC accumulation avoids serializing every particle on a
    // tiny global grid. The block belongs to exactly one realization.
    extern __shared__ float local[];
    for(int j=threadIdx.x;j<2*ng;j+=blockDim.x)local[j]=0;
    __syncthreads();
    int b=blockIdx.y,p=blockIdx.x*blockDim.x+threadIdx.x;
    float* g=local;float w=(float)ng/n;
    if(p<n){p+=b*n;
    float u=xe[p]/length*ng;int j=(int)floorf(u);float f=u-floorf(u);j=(j%ng+ng)%ng;
    atomicAdd(g+j,w*(1-f));atomicAdd(g+(j+1)%ng,w*f);
    atomicAdd(g+ng+j,-w*(1-f));atomicAdd(g+ng+(j+1)%ng,-w*f);
    u=xi[p]/length*ng;j=(int)floorf(u);f=u-floorf(u);j=(j%ng+ng)%ng;
    atomicAdd(g+ng+j,w*(1-f));atomicAdd(g+ng+(j+1)%ng,w*f);
    }
    __syncthreads();
    for(int j=threadIdx.x;j<2*ng;j+=blockDim.x)atomicAdd(grid+2*b*ng+j,local[j]);
}
extern "C" __global__ void poisson(const Pair* all,Pair* field,Pair* obs,int ng,int batches,float length,int mode,float norm,int record,int steps) {
    int z=blockIdx.x*blockDim.x+threadIdx.x;int nf=ng/2+1;if(z>=nf*batches)return;
    int b=z/nf,j=z%nf;const Pair* f=all+2*b*nf;
    if(j==0&&record>=0){obs[b*steps+record].x=norm*f[mode].x;obs[b*steps+record].y=norm*f[mode].y;}
    if(j==0||j==ng/2){field[z].x=0;field[z].y=0;return;}
    float k=6.2831853071795864769f*j/length;
    field[z].x=f[nf+j].y/(k*ng);field[z].y=-f[nf+j].x/(k*ng);
}
__device__ float gather(const float* e,float x,int ng,float length) {
    float u=x/length*ng;int j=(int)floorf(u);float f=u-floorf(u);j=(j%ng+ng)%ng;
    return e[j]*(1-f)+e[(j+1)%ng]*f;
}
__device__ float wrap(float x,float length){return x-floorf(x/length)*length;}
extern "C" __global__ void push(float* xe,float* ve,float* xi,float* vi,const float* fields,int n,int ng,int batches,float length,float mu,float kick,float drift) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n*batches)return;const float* e=fields+(p/n)*ng;
    ve[p]-=kick*gather(e,xe[p],ng,length);vi[p]+=kick*gather(e,xi[p],ng,length)/mu;
    xe[p]=wrap(xe[p]+drift*ve[p],length);xi[p]=wrap(xi[p]+drift*vi[p],length);
}
