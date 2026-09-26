/* hexagon-sim --timing: pcycles per HMX instruction for the loop shapes compared in the README, operands resident
 * in VTCM (the simulator's HMX timing matched the phone's MAC/pcycle for the fp16 and int8 cm :deep loops).
 * rate_sim <var> <kt> <tiles>: 0 int8 cm, one crouton per instruction + store; 1 int8 cm, activation :deep over
 * kt croutons; 2 as 0 without the store; 3 fp16 (activation :deep, as hmx_gemm.h); 4 fp16 one crouton per
 * instruction; 5 int8 cm + weight :deep (64 columns, hmx_gemm_u8.h); 6 fp16 + weight :deep.
 * Build/run: hexagon-clang -mv69 -mhmx -mhvx -O2 rate_sim.c -o r.elf -lhexagon;
 *            hexagon-sim -mv69 --mhmx 1 --timing r.elf -- <var> <kt> 64 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <hexagon_sim_timer.h>
static unsigned cfg(int off){ unsigned b; __asm__ volatile("%0 = cfgbase":"=r"(b)); b<<=16; return *(volatile unsigned*)(b+off); }
int main(int argc,char**argv){
  int var=atoi(argv[1]), kt=atoi(argv[2]), tiles=atoi(argv[3]);
  uint8_t* v=(uint8_t*)(cfg(0x38)<<16);
  unsigned r; __asm__ volatile("%0 = ssr":"=r"(r)); r|=1u<<26; __asm__ volatile("ssr = %0; isync"::"r"(r));
  uint8_t *a=v, *w=v+256*1024, *c=v+512*1024, *t=v+768*1024;
  memset(v,0,800*1024);
  __asm__ volatile("bias = mxmem(%0)"::"r"(t):"memory");
  unsigned long long t0=hexagon_sim_read_pcycles();
  for(int i=0;i<tiles;i++){
    uint8_t* ct = c + (i&7)*2048;
    if(var==0){ for(int k=0;k<kt;k++) __asm__ volatile("{ activation.ub = mxmem(%0,%1):cm\n weight.b = mxmem(%2,%3) }"::"r"(a+2048*k),"r"(0x7ff),"r"(w+1024*k),"r"(0x3ff):"memory");
      __asm__ volatile("mxmem(%0,%1):after:cm:sat.ub = acc"::"r"(ct),"r"(0):"memory"); }
    else if(var==1){ __asm__ volatile("{ activation.ub = mxmem(%0,%1):deep:cm\n weight.b = mxmem(%2,%3) }"::"r"(a),"r"(kt*2048-1),"r"(w),"r"(kt*2048-1):"memory");
      __asm__ volatile("mxmem(%0,%1):after:cm:sat.ub = acc"::"r"(ct),"r"(0):"memory"); }
    else if(var==2){ for(int k=0;k<kt;k++) __asm__ volatile("{ activation.ub = mxmem(%0,%1):cm\n weight.b = mxmem(%2,%3) }"::"r"(a+2048*k),"r"(0x7ff),"r"(w+1024*k),"r"(0x3ff):"memory"); }
    else if(var==3){ /* fp16 reference: our current */ __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }"::"r"(a),"r"(kt*2048-1),"r"(w),"r"(kt*2048-1):"memory");
      __asm__ volatile("mxmem(%0,%1):after.hf = acc"::"r"(ct),"r"(0):"memory"); }
    else if(var==4){ for(int k=0;k<kt;k++) __asm__ volatile("{ activation.hf = mxmem(%0,%1)\n weight.hf = mxmem(%2,%3) }"::"r"(a+2048*k),"r"(0x7ff),"r"(w+2048*k),"r"(0x7ff):"memory");
      __asm__ volatile("mxmem(%0,%1):after.hf = acc"::"r"(ct),"r"(0):"memory"); }
    else if(var==5){ for(int k=0;k<kt;k++) __asm__ volatile("{ activation.ub = mxmem(%0,%1):cm\n weight.b = mxmem(%2,%3):deep }"::"r"(a+2048*k),"r"(0x7ff),"r"(w+2048*k),"r"(0x7ff):"memory");
      __asm__ volatile("mxmem(%0,%1):after:cm:sat.ub = acc"::"r"(ct),"r"(0):"memory");
      __asm__ volatile("mxmem(%0,%1):after:cm:sat.ub = acc"::"r"(ct+2048),"r"(0):"memory"); }
    else if(var==6){ for(int k=0;k<kt;k++) __asm__ volatile("{ activation.hf = mxmem(%0,%1)\n weight.hf = mxmem(%2,%3):deep }"::"r"(a+2048*k),"r"(0x7ff),"r"(w+4096*k),"r"(0xfff):"memory");
      __asm__ volatile("mxmem(%0,%1):after.hf = acc"::"r"(ct),"r"(0):"memory");
      __asm__ volatile("mxmem(%0,%1):after.hf = acc"::"r"(ct+2048),"r"(0):"memory"); }
  }
  unsigned long long t1=hexagon_sim_read_pcycles();
  double macs=(double)tiles*kt*(var==5?131072:var==6?65536:(var>=3?32768:65536));
  printf("var %d kt %d tiles %d: %.2f pcycles/tile, %.2f pcycles per instruction-crouton, %.0f MAC/cycle\n",var,kt,tiles,(double)(t1-t0)/tiles,(double)(t1-t0)/tiles/kt,macs/(t1-t0));
  return 0;
}
