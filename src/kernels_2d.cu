// Square, periodic, bilinear CIC 2D2V. Layout [batch][y][x].
struct Pair {float x,y;};
extern "C" __global__ void clear2(Pair* g,int count) {
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<count){g[j].x=0;g[j].y=0;}
}
__device__ void add2(Pair* g,Pair x,int ng,float length,float w) {
 float u=x.x/length*ng,v=x.y/length*ng;int i=(int)floorf(u),j=(int)floorf(v);
 float a=u-floorf(u),b=v-floorf(v);i=(i%ng+ng)%ng;j=(j%ng+ng)%ng;
 atomicAdd(&g[j*ng+i].x,w*(1-a)*(1-b));atomicAdd(&g[j*ng+(i+1)%ng].x,w*a*(1-b));
 atomicAdd(&g[((j+1)%ng)*ng+i].x,w*(1-a)*b);atomicAdd(&g[((j+1)%ng)*ng+(i+1)%ng].x,w*a*b);
}
extern "C" __global__ void deposit2(const Pair* xe,const Pair* xi,Pair* g,int n,int ng,int batches,float length) {
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n*batches)return;
 int nn=ng*ng,b=p/n;Pair* ne=g+2*b*nn;Pair* rho=ne+nn;float w=(float)nn/n;
 add2(ne,xe[p],ng,length,w);add2(rho,xe[p],ng,length,-w);add2(rho,xi[p],ng,length,w);
}
extern "C" __global__ void poisson2(const Pair* all,Pair* field,Pair* obs,int ng,int batches,float length,int mode,float norm,int record,int steps) {
 int z=blockIdx.x*blockDim.x+threadIdx.x,nn=ng*ng;if(z>=nn*batches)return;
 int b=z/nn,j=z%nn,x=j%ng,y=j/ng;const Pair* f=all+2*b*nn;
 if(j==0&&record>=0){obs[b*steps+record].x=norm*f[mode].x;obs[b*steps+record].y=norm*f[mode].y;}
 float kx=6.2831853071795864769f*(x<=ng/2?x:x-ng)/length;
 float ky=6.2831853071795864769f*(y<=ng/2?y:y-ng)/length;float k2=kx*kx+ky*ky;
 float ax=(k2==0||x==ng/2)?0:kx/(k2*nn),ay=(k2==0||y==ng/2)?0:ky/(k2*nn);
 Pair r=f[nn+j];field[2*b*nn+j].x=r.y*ax;field[2*b*nn+j].y=-r.x*ax;
 field[(2*b+1)*nn+j].x=r.y*ay;field[(2*b+1)*nn+j].y=-r.x*ay;
}
__device__ float gather2(const Pair* e,Pair x,int ng,float length) {
 float u=x.x/length*ng,v=x.y/length*ng;int i=(int)floorf(u),j=(int)floorf(v);
 float a=u-floorf(u),b=v-floorf(v);i=(i%ng+ng)%ng;j=(j%ng+ng)%ng;
 return e[j*ng+i].x*(1-a)*(1-b)+e[j*ng+(i+1)%ng].x*a*(1-b)+e[((j+1)%ng)*ng+i].x*(1-a)*b+e[((j+1)%ng)*ng+(i+1)%ng].x*a*b;
}
__device__ float wrap2(float x,float l){return x-floorf(x/l)*l;}
__device__ void move2(Pair* x,Pair* v,const Pair* e,int ng,float length,float qm,float kick,float drift) {
 float ex=gather2(e,*x,ng,length),ey=gather2(e+ng*ng,*x,ng,length);
 v->x+=kick*qm*ex;v->y+=kick*qm*ey;
 x->x=wrap2(x->x+drift*v->x,length);x->y=wrap2(x->y+drift*v->y,length);
}
extern "C" __global__ void push2(Pair* xe,Pair* ve,Pair* xi,Pair* vi,const Pair* fields,int n,int ng,int batches,float length,float mu,float kick,float drift) {
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n*batches)return;const Pair* e=fields+2*(p/n)*ng*ng;
 move2(xe+p,ve+p,e,ng,length,-1,kick,drift);move2(xi+p,vi+p,e,ng,length,1/mu,kick,drift);
}
