// ResNet-50 stage-3 3x3 conv, 256 ch @14x14: 2.21 ms fp16 (0.231 GFLOP = 105 GFLOPS)
// plan.txt line: kern r_7_7_4_2_2_16_4_64_3_3_4_v44 4 7 7 2 2 16 4 i172,1,13328,2 i170,1,13328,2 i60,64,2880,2 i173,1,64,4
// (g = work groups, l = local size; NDRange = g*l; iN,h,w,itemsize = image2d_t over a buffer, RGBA, h x w texels, itemsize 2 = half)
__kernel void r_7_7_4_2_2_16_4_64_3_3_4_v44(write_only image2d_t data0_1_13328_4, read_only image2d_t data1_1_13328_4, read_only image2d_t data2_64_2880_4, read_only image2d_t data3_1_64_4) {
const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  float buf0[4];
  int gidx0 = get_group_id(0); /* 4 */
  int gidx1 = get_group_id(1); /* 7 */
  int gidx2 = get_group_id(2); /* 7 */
  int lidx0 = get_local_id(0); /* 2 */
  int lidx1 = get_local_id(1); /* 2 */
  int lidx2 = get_local_id(2); /* 16 */
  int alu0 = (lidx2+(gidx0<<4));
  *(buf0+0) = 0.0f;
  *(buf0+1) = 0.0f;
  *(buf0+2) = 0.0f;
  *(buf0+3) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    for (int Ridx2 = 0; Ridx2 < 3; Ridx2++) {
      int alu5 = (lidx0+(gidx2<<1)+Ridx2);
      for (int Ridx3 = 0; Ridx3 < 3; Ridx3++) {
        int alu6 = (lidx1+(gidx1<<1)+Ridx3);
        float4 val0 = (((0<(gidx1+lidx1+Ridx3))&(alu6<15)&(0<(gidx2+lidx0+Ridx2))&(alu5<15))?read_imagef(data1_1_13328_4, smp, (int2)(((((alu6+13)%14)*68)+Ridx0+(((alu5+((gidx1+((lidx1+Ridx3+1)>>1)+6)/7)+12)%14)*952)),0)):(float4)(0.0f,0.0f,0.0f,0.0f));
        int alu7 = ((Ridx0*12)+(Ridx3<<2)+(Ridx2*960));
        float4 val1 = read_imagef(data2_64_2880_4, smp, (int2)((alu7+1),alu0));
        float4 val2 = read_imagef(data2_64_2880_4, smp, (int2)((alu7+2),alu0));
        float4 val3 = read_imagef(data2_64_2880_4, smp, (int2)((alu7+3),alu0));
        float4 val4 = read_imagef(data2_64_2880_4, smp, (int2)(alu7,alu0));
        *(buf0+0) = ((*(buf0+0))+(val0.x*val4.x)+(val0.y*val1.x)+(val0.z*val2.x)+(val0.w*val3.x));
        *(buf0+1) = ((*(buf0+1))+(val0.x*val4.y)+(val0.y*val1.y)+(val0.z*val2.y)+(val0.w*val3.y));
        *(buf0+2) = ((*(buf0+2))+(val0.x*val4.z)+(val0.y*val1.z)+(val0.z*val2.z)+(val0.w*val3.z));
        *(buf0+3) = ((*(buf0+3))+(val0.x*val4.w)+(val0.y*val1.w)+(val0.z*val2.w)+(val0.w*val3.w));
      }
    }
  }
  float4 val5 = read_imagef(data3_1_64_4, smp, (int2)(alu0,0));
  float alu15 = ((*(buf0+0))+val5.x);
  float alu16 = ((*(buf0+1))+val5.y);
  float alu17 = ((*(buf0+2))+val5.z);
  float alu18 = ((*(buf0+3))+val5.w);
  float alu19 = ((0.0f<alu15)?alu15:0.0f);
  float alu20 = ((0.0f<alu16)?alu16:0.0f);
  float alu21 = ((0.0f<alu17)?alu17:0.0f);
  float alu22 = ((0.0f<alu18)?alu18:0.0f);
  write_imagef(data0_1_13328_4, (int2)((alu0+(gidx1*136)+(lidx1*68)+(gidx2*1904)+(lidx0*952)),0), (float4)(alu19,alu20,alu21,alu22));
}
