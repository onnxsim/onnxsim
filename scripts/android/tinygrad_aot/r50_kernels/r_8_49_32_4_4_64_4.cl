// ResNet-50 1x1 conv + residual add + ReLU (5 inputs incl. the residual image): 0.86 ms fp16
// plan.txt line: kern r_8_49_32_4_4_64_4 49 8 1 32 1 1 5 i174,8,6272,4 i172,1,13328,2 i62,8,10240,2 i175,1,256,4 i166,8,6272,4
// (g = work groups, l = local size; NDRange = g*l; iN,h,w,itemsize = image2d_t over a buffer, RGBA, h x w texels, itemsize 2 = half)
__kernel void r_8_49_32_4_4_64_4(write_only image2d_t data0_8_6272_4, read_only image2d_t data1_1_13328_4, read_only image2d_t data2_8_10240_4, read_only image2d_t data3_1_256_4, read_only image2d_t data4_8_6272_4) {
const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  float buf0[16];
  int gidx0 = get_group_id(0); /* 49 */
  int gidx1 = get_group_id(1); /* 8 */
  int lidx0 = get_local_id(0); /* 32 */
  *(buf0+0) = 0.0f;
  *(buf0+1) = 0.0f;
  *(buf0+10) = 0.0f;
  *(buf0+11) = 0.0f;
  *(buf0+12) = 0.0f;
  *(buf0+13) = 0.0f;
  *(buf0+14) = 0.0f;
  *(buf0+15) = 0.0f;
  *(buf0+2) = 0.0f;
  *(buf0+3) = 0.0f;
  *(buf0+4) = 0.0f;
  *(buf0+5) = 0.0f;
  *(buf0+6) = 0.0f;
  *(buf0+7) = 0.0f;
  *(buf0+8) = 0.0f;
  *(buf0+9) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    int alu16 = ((gidx0*272)+Ridx0);
    float4 val0 = read_imagef(data1_1_13328_4, smp, (int2)((alu16+136),0));
    float4 val1 = read_imagef(data1_1_13328_4, smp, (int2)((alu16+204),0));
    float4 val2 = read_imagef(data1_1_13328_4, smp, (int2)((alu16+68),0));
    float4 val3 = read_imagef(data1_1_13328_4, smp, (int2)(alu16,0));
    int alu17 = ((lidx0*320)+(Ridx0<<2));
    float4 val4 = read_imagef(data2_8_10240_4, smp, (int2)((alu17+1),gidx1));
    float4 val5 = read_imagef(data2_8_10240_4, smp, (int2)((alu17+2),gidx1));
    float4 val6 = read_imagef(data2_8_10240_4, smp, (int2)((alu17+3),gidx1));
    float4 val7 = read_imagef(data2_8_10240_4, smp, (int2)(alu17,gidx1));
    *(buf0+0) = ((*(buf0+0))+(val3.x*val7.x)+(val3.y*val4.x)+(val3.z*val5.x)+(val3.w*val6.x));
    *(buf0+1) = ((*(buf0+1))+(val2.x*val7.x)+(val2.y*val4.x)+(val2.z*val5.x)+(val2.w*val6.x));
    *(buf0+10) = ((*(buf0+10))+(val0.x*val7.z)+(val0.y*val4.z)+(val0.z*val5.z)+(val0.w*val6.z));
    *(buf0+11) = ((*(buf0+11))+(val1.x*val7.z)+(val1.y*val4.z)+(val1.z*val5.z)+(val1.w*val6.z));
    *(buf0+12) = ((*(buf0+12))+(val3.x*val7.w)+(val3.y*val4.w)+(val3.z*val5.w)+(val3.w*val6.w));
    *(buf0+13) = ((*(buf0+13))+(val2.x*val7.w)+(val2.y*val4.w)+(val2.z*val5.w)+(val2.w*val6.w));
    *(buf0+14) = ((*(buf0+14))+(val0.x*val7.w)+(val0.y*val4.w)+(val0.z*val5.w)+(val0.w*val6.w));
    *(buf0+15) = ((*(buf0+15))+(val1.x*val7.w)+(val1.y*val4.w)+(val1.z*val5.w)+(val1.w*val6.w));
    *(buf0+2) = ((*(buf0+2))+(val0.x*val7.x)+(val0.y*val4.x)+(val0.z*val5.x)+(val0.w*val6.x));
    *(buf0+3) = ((*(buf0+3))+(val1.x*val7.x)+(val1.y*val4.x)+(val1.z*val5.x)+(val1.w*val6.x));
    *(buf0+4) = ((*(buf0+4))+(val3.x*val7.y)+(val3.y*val4.y)+(val3.z*val5.y)+(val3.w*val6.y));
    *(buf0+5) = ((*(buf0+5))+(val2.x*val7.y)+(val2.y*val4.y)+(val2.z*val5.y)+(val2.w*val6.y));
    *(buf0+6) = ((*(buf0+6))+(val0.x*val7.y)+(val0.y*val4.y)+(val0.z*val5.y)+(val0.w*val6.y));
    *(buf0+7) = ((*(buf0+7))+(val1.x*val7.y)+(val1.y*val4.y)+(val1.z*val5.y)+(val1.w*val6.y));
    *(buf0+8) = ((*(buf0+8))+(val3.x*val7.z)+(val3.y*val4.z)+(val3.z*val5.z)+(val3.w*val6.z));
    *(buf0+9) = ((*(buf0+9))+(val2.x*val7.z)+(val2.y*val4.z)+(val2.z*val5.z)+(val2.w*val6.z));
  }
  float4 val8 = read_imagef(data3_1_256_4, smp, (int2)((lidx0+(gidx1<<5)),0));
  int alu35 = (gidx0+(lidx0*196));
  float4 val9 = read_imagef(data4_8_6272_4, smp, (int2)(alu35,gidx1));
  int alu36 = (alu35+147);
  float4 val10 = read_imagef(data4_8_6272_4, smp, (int2)(alu36,gidx1));
  int alu37 = (alu35+49);
  float4 val11 = read_imagef(data4_8_6272_4, smp, (int2)(alu37,gidx1));
  int alu38 = (alu35+98);
  float4 val12 = read_imagef(data4_8_6272_4, smp, (int2)(alu38,gidx1));
  float alu39 = ((*(buf0+0))+val8.x+val9.x);
  float alu40 = ((*(buf0+1))+val8.x+val9.y);
  float alu41 = ((*(buf0+2))+val8.x+val9.z);
  float alu42 = ((*(buf0+3))+val8.x+val9.w);
  float alu43 = ((0.0f<alu39)?alu39:0.0f);
  float alu44 = ((0.0f<alu40)?alu40:0.0f);
  float alu45 = ((0.0f<alu41)?alu41:0.0f);
  float alu46 = ((0.0f<alu42)?alu42:0.0f);
  float alu47 = ((*(buf0+12))+val8.w+val10.x);
  float alu48 = ((*(buf0+13))+val8.w+val10.y);
  float alu49 = ((*(buf0+14))+val8.w+val10.z);
  float alu50 = ((*(buf0+15))+val8.w+val10.w);
  float alu51 = ((0.0f<alu47)?alu47:0.0f);
  float alu52 = ((0.0f<alu48)?alu48:0.0f);
  float alu53 = ((0.0f<alu49)?alu49:0.0f);
  float alu54 = ((0.0f<alu50)?alu50:0.0f);
  float alu55 = ((*(buf0+4))+val8.y+val11.x);
  float alu56 = ((*(buf0+5))+val8.y+val11.y);
  float alu57 = ((*(buf0+6))+val8.y+val11.z);
  float alu58 = ((*(buf0+7))+val8.y+val11.w);
  float alu59 = ((0.0f<alu55)?alu55:0.0f);
  float alu60 = ((0.0f<alu56)?alu56:0.0f);
  float alu61 = ((0.0f<alu57)?alu57:0.0f);
  float alu62 = ((0.0f<alu58)?alu58:0.0f);
  float alu63 = ((*(buf0+10))+val8.z+val12.z);
  float alu64 = ((*(buf0+11))+val8.z+val12.w);
  float alu65 = ((*(buf0+8))+val8.z+val12.x);
  float alu66 = ((*(buf0+9))+val8.z+val12.y);
  float alu67 = ((0.0f<alu63)?alu63:0.0f);
  float alu68 = ((0.0f<alu64)?alu64:0.0f);
  float alu69 = ((0.0f<alu65)?alu65:0.0f);
  float alu70 = ((0.0f<alu66)?alu66:0.0f);
  write_imagef(data0_8_6272_4, (int2)(alu35,gidx1), (float4)(alu43,alu44,alu45,alu46));
  write_imagef(data0_8_6272_4, (int2)(alu36,gidx1), (float4)(alu51,alu52,alu53,alu54));
  write_imagef(data0_8_6272_4, (int2)(alu37,gidx1), (float4)(alu59,alu60,alu61,alu62));
  write_imagef(data0_8_6272_4, (int2)(alu38,gidx1), (float4)(alu69,alu70,alu67,alu68));
}
