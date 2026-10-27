// Included after the existing 2D TSC/Poisson kernels. Positions remain Pair;
// velocities use 16-byte aligned storage with all three physical components.
struct Velocity {float x,y,z,w;};
__device__ void move3(Pair* x,Velocity* v,const Pair* e,int ng,float length,float qm,float kick,float drift,float bx,float by,float bz) {
 Pair f=gather2(e,*x,ng,length);
 float h=0.5f*kick*qm,tx=h*bx,ty=h*by,tz=h*bz;
 float ux=v->x+h*f.x,uy=v->y+h*f.y,uz=v->z;
 float px=ux+uy*tz-uz*ty,py=uy+uz*tx-ux*tz,pz=uz+ux*ty-uy*tx;
 float scale=2.f/(1.f+tx*tx+ty*ty+tz*tz);
 v->x=ux+scale*(py*tz-pz*ty)+h*f.x;
 v->y=uy+scale*(pz*tx-px*tz)+h*f.y;
 v->z=uz+scale*(px*ty-py*tx);
 x->x=wrap2(x->x+drift*v->x,length);x->y=wrap2(x->y+drift*v->y,length);
}
extern "C" __global__ void push3(Pair* xe,Velocity* ve,Pair* xi,Velocity* vi,const Pair* fields,int n,int ng,int batches,float length,float mu,float kick,float drift,float bx,float by,float bz) {
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n*batches)return;const Pair* e=fields+2*(p/n)*ng*ng;
 move3(xe+p,ve+p,e,ng,length,-1,kick,drift,bx,by,bz);
 move3(xi+p,vi+p,e,ng,length,1/mu,kick,drift,bx,by,bz);
}
