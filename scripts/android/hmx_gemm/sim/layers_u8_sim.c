/* hexagon-sim check of hmx_layer_u8cm: L chained C -> C int8 layers with activations in crouton form in VTCM,
 * vs an exact reference (power-of-two scales). layers_u8_sim M C L */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../hmx_gemm_u8.h"
static unsigned cfg(int off){ unsigned b; __asm__ volatile("%0 = cfgbase":"=r"(b)); b<<=16; return *(volatile unsigned*)(b+off); }
int main(int argc,char**argv){
  int M=atoi(argv[1]),C=atoi(argv[2]),L=atoi(argv[3]),mt=(M+63)/64,kt=C/32;
  uint8_t* v=(uint8_t*)(cfg(0x38)<<16);
  unsigned r; __asm__ volatile("%0 = ssr":"=r"(r)); r|=1u<<26; __asm__ volatile("ssr = %0; isync"::"r"(r));
  size_t act=(size_t)mt*kt*2048; uint8_t *X=v,*Y=v+act,*Wv=Y+act; uint32_t* T=(uint32_t*)(Wv+(size_t)L*C*C);
  uint8_t *A=malloc((size_t)M*C),*R=malloc((size_t)M*C),*R2=malloc((size_t)M*C),*O=malloc((size_t)M*C); int8_t *W=malloc((size_t)C*C);
  srand(9);
  for(size_t i=0;i<(size_t)M*C;i++) A[i]=rand()%256;
  memcpy(R,A,(size_t)M*C);
  for(int l=0;l<L;l++){
    for(size_t i=0;i<(size_t)C*C;i++){ int x=rand()%256-128+30; W[i]=(int8_t)(x>127?127:x); }
    hmx_pack_w_u8cm(W,C,C,(int8_t*)(Wv+(size_t)l*C*C));
    /* reference layer + pick a power-of-two scale per layer that puts the median output mid-range */
    static long acc[1<<16]; long mx=1;
    for(int i=0;i<M;i++) for(int j=0;j<C;j++){ long s=0; for(int k=0;k<C;k++) s+=(long)R[(size_t)i*C+k]*W[(size_t)k*C+j]; acc[(size_t)i*C+j]=s; if(s>mx)mx=s; }
    int e=(int)floor(log2((double)mx/512/255))+1; __fp16 sc=(__fp16)ldexp(1.0,-e); uint16_t sh; memcpy(&sh,&sc,2);
    for(int j=0;j<C/32;j++) for(int k=0;k<64;k++) T[((size_t)l*(C/32)+j)*64+k]=k<32?sh:0;
    for(size_t i=0;i<(size_t)M*C;i++){ double y=acc[i]<0?0:floor((double)acc[i]*ldexp(1.0,-e)/512); R2[i]=y>255?255:(uint8_t)y; }
    memcpy(R,R2,(size_t)M*C);
  }
  for(int mb=0;mb<mt;mb++) hmx_pack_a_u8cm(A,M,C,mb*64,X+(size_t)mb*kt*2048);
  uint8_t *src=X,*dst=Y;
  for(int l=0;l<L;l++){ hmx_layer_u8cm(src,dst,Wv+(size_t)l*C*C,T+(size_t)l*(C/32)*64,mt,kt,C); uint8_t* t=src; src=dst; dst=t; }
  hmx_unpack_rows_u8cm(src,O,M,C);
  int bad=0,zero=0; for(size_t i=0;i<(size_t)M*C;i++){ if(O[i]!=R[i]){ if(bad<4) printf("bad %zu got %d want %d\n",i,O[i],R[i]); bad++; } if(!R[i]) zero++; }
  printf("layers_u8 M %d C %d L %d: %d mismatches of %d (%d zero) %s\n",M,C,L,bad,M*C,zero,bad?"FAIL":"PASS");
  return 0;
}
