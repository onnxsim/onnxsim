/* hexagon-sim check of hmx_gemm_u8 (int8 cm path) vs an exact reference: gemm_u8_sim M K N [pow2 1|0]
 * pow2=1: power-of-two column scales, where the HMX conversion is exact (must be 0 mismatches);
 * pow2=0: random fp16 scales, reports how many outputs are off and by how much. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../hmx_gemm_u8.h"
static unsigned cfg(int off){ unsigned b; __asm__ volatile("%0 = cfgbase":"=r"(b)); b<<=16; return *(volatile unsigned*)(b+off); }
static double h2d(uint16_t h){ __fp16 x; memcpy(&x,&h,2); return (double)x; }
int main(int argc,char**argv){
  int M=atoi(argv[1]),K=atoi(argv[2]),N=atoi(argv[3]),pow2=argc>4?atoi(argv[4]):1;
  uint8_t* v=(uint8_t*)(cfg(0x38)<<16); unsigned vs=cfg(0x3c)*1024;
  unsigned r; __asm__ volatile("%0 = ssr":"=r"(r)); r|=1u<<26; __asm__ volatile("ssr = %0; isync"::"r"(r));
  uint8_t *A=malloc((size_t)M*K),*C=malloc((size_t)M*N); int8_t *W=malloc((size_t)K*N),*Wp=malloc((size_t)K*N); uint16_t* S=malloc(N*2);
  srand(5);
  for(size_t i=0;i<(size_t)M*K;i++) A[i]=rand()%256;
  for(size_t i=0;i<(size_t)K*N;i++){ int x=rand()%256-128+40; W[i]=(int8_t)(x>127?127:x); }
  for(int j=0;j<N;j++){ double m=pow2?1.0:1.0+(rand()%1024)/1024.0; __fp16 x=(__fp16)(m/(double)(1<<(8+rand()%4))); memcpy(&S[j],&x,2); }
  hmx_pack_w_u8cm(W,K,N,Wp);
  int rc=hmx_gemm_u8_prof(A,Wp,S,C,M,K,N,v,vs,0,NULL);
  int bad=0,sat=0,zero=0; long worst=0;
  for(int i=0;i<M;i++) for(int j=0;j<N;j++){ long acc=0; for(int k=0;k<K;k++) acc+=(long)A[(size_t)i*K+k]*W[(size_t)k*N+j];
    double e=acc<0?0:floor((double)acc*h2d(S[j])/512); if(e>=255){e=255;sat++;} if(e==0) zero++;
    long d=(long)C[(size_t)i*N+j]-(long)e; if(d){ if(bad<4) printf("bad %d,%d got %d want %g acc %ld\n",i,j,C[(size_t)i*N+j],e,acc); bad++; if(labs(d)>worst) worst=labs(d);} }
  printf("gemm_u8 %dx%dx%d pow2 %d rc %d vtcm %u: %d mismatches of %d (max |diff| %ld; %d saturated, %d zero) %s\n",M,K,N,pow2,rc,vs,bad,M*N,worst,sat,zero,(pow2&&bad)||rc?"FAIL":"PASS");
  return 0;
}
