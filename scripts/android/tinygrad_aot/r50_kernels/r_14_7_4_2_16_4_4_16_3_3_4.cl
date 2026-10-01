// ResNet-50 stage-1 3x3 conv, 64 ch @56x56 (+bias+ReLU): 1.03 ms fp16 (0.231 GFLOP = 225 GFLOPS)
// plan.txt line: kern r_14_7_4_2_16_4_4_16_3_3_4 7 14 1 4 2 16 4 i115,14,5376,2 i113,56,1344,2 i6,1,10368,2 i116,1,16,4
// (g = work groups, l = local size; NDRange = g*l; iN,h,w,itemsize = image2d_t over a buffer, RGBA, h x w texels, itemsize 2 = half)
__kernel void r_14_7_4_2_16_4_4_16_3_3_4(write_only image2d_t data0_14_5376_4, read_only image2d_t data1_56_1344_4, read_only image2d_t data2_1_10368_4, read_only image2d_t data3_1_16_4) {
const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  float buf0[16];
  int gidx0 = get_group_id(0); /* 7 */
  int gidx1 = get_group_id(1); /* 14 */
  int lidx0 = get_local_id(0); /* 4 */
  int lidx1 = get_local_id(1); /* 2 */
  int lidx2 = get_local_id(2); /* 16 */
  float4 cast0 = (float4)(0.0f,0.0f,0.0f,0.0f);
  int alu0 = (gidx0*192);
  int alu1 = (lidx1*96);
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
  for (int Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    for (int Ridx2 = 0; Ridx2 < 3; Ridx2++) {
      int alu18 = (lidx0+(gidx1<<2)+Ridx2);
      int alu19 = ((alu18+55)%56);
      bool alu20 = ((0<(gidx1+lidx0+Ridx2))&(alu18<57));
      for (int Ridx3 = 0; Ridx3 < 3; Ridx3++) {
        int alu21 = ((gidx0+((lidx1+((Ridx3+2)>>2))>>1))/7);
        int alu22 = (alu0+alu1+(Ridx3*24)+Ridx0);
        int alu23 = ((gidx0<<3)+(lidx1<<2)+Ridx3);
        float4 val0 = (((alu23<54)&alu20)?read_imagef(data1_56_1344_4, smp, (int2)((alu22+48),(alu21+((alu18+alu21+55)%56)))):cast0);
        float4 val1 = (alu20?read_imagef(data1_56_1344_4, smp, (int2)((alu22+24),alu19)):cast0);
        float4 val2 = (alu20?read_imagef(data1_56_1344_4, smp, (int2)(alu22,alu19)):cast0);
        float4 val3 = (((0<(gidx0+lidx1+Ridx3))&alu20)?read_imagef(data1_56_1344_4, smp, (int2)(((((alu23+55)%56)*24)+Ridx0),alu19)):cast0);
        int alu24 = ((Ridx0*12)+(Ridx3<<2)+(Ridx2*216)+(lidx2*648));
        float4 val4 = read_imagef(data2_1_10368_4, smp, (int2)((alu24+1),0));
        float4 val5 = read_imagef(data2_1_10368_4, smp, (int2)((alu24+2),0));
        float4 val6 = read_imagef(data2_1_10368_4, smp, (int2)((alu24+3),0));
        float4 val7 = read_imagef(data2_1_10368_4, smp, (int2)(alu24,0));
        *(buf0+0) = ((*(buf0+0))+(val3.x*val7.x)+(val3.y*val4.x)+(val3.z*val5.x)+(val3.w*val6.x));
        *(buf0+1) = ((*(buf0+1))+(val3.x*val7.y)+(val3.y*val4.y)+(val3.z*val5.y)+(val3.w*val6.y));
        *(buf0+10) = ((*(buf0+10))+(val1.x*val7.z)+(val1.y*val4.z)+(val1.z*val5.z)+(val1.w*val6.z));
        *(buf0+11) = ((*(buf0+11))+(val1.x*val7.w)+(val1.y*val4.w)+(val1.z*val5.w)+(val1.w*val6.w));
        *(buf0+12) = ((*(buf0+12))+(val0.x*val7.x)+(val0.y*val4.x)+(val0.z*val5.x)+(val0.w*val6.x));
        *(buf0+13) = ((*(buf0+13))+(val0.x*val7.y)+(val0.y*val4.y)+(val0.z*val5.y)+(val0.w*val6.y));
        *(buf0+14) = ((*(buf0+14))+(val0.x*val7.z)+(val0.y*val4.z)+(val0.z*val5.z)+(val0.w*val6.z));
        *(buf0+15) = ((*(buf0+15))+(val0.x*val7.w)+(val0.y*val4.w)+(val0.z*val5.w)+(val0.w*val6.w));
        *(buf0+2) = ((*(buf0+2))+(val3.x*val7.z)+(val3.y*val4.z)+(val3.z*val5.z)+(val3.w*val6.z));
        *(buf0+3) = ((*(buf0+3))+(val3.x*val7.w)+(val3.y*val4.w)+(val3.z*val5.w)+(val3.w*val6.w));
        *(buf0+4) = ((*(buf0+4))+(val2.x*val7.x)+(val2.y*val4.x)+(val2.z*val5.x)+(val2.w*val6.x));
        *(buf0+5) = ((*(buf0+5))+(val2.x*val7.y)+(val2.y*val4.y)+(val2.z*val5.y)+(val2.w*val6.y));
        *(buf0+6) = ((*(buf0+6))+(val2.x*val7.z)+(val2.y*val4.z)+(val2.z*val5.z)+(val2.w*val6.z));
        *(buf0+7) = ((*(buf0+7))+(val2.x*val7.w)+(val2.y*val4.w)+(val2.z*val5.w)+(val2.w*val6.w));
        *(buf0+8) = ((*(buf0+8))+(val1.x*val7.x)+(val1.y*val4.x)+(val1.z*val5.x)+(val1.w*val6.x));
        *(buf0+9) = ((*(buf0+9))+(val1.x*val7.y)+(val1.y*val4.y)+(val1.z*val5.y)+(val1.w*val6.y));
      }
    }
  }
  float4 val8 = read_imagef(data3_1_16_4, smp, (int2)(lidx2,0));
  float alu44 = ((*(buf0+0))+val8.x);
  float alu45 = ((*(buf0+1))+val8.y);
  float alu46 = ((*(buf0+2))+val8.z);
  float alu47 = ((*(buf0+3))+val8.w);
  float alu48 = ((0.0f<alu44)?alu44:0.0f);
  float alu49 = ((0.0f<alu45)?alu45:0.0f);
  float alu50 = ((0.0f<alu46)?alu46:0.0f);
  float alu51 = ((0.0f<alu47)?alu47:0.0f);
  float alu52 = ((*(buf0+12))+val8.x);
  float alu53 = ((*(buf0+13))+val8.y);
  float alu54 = ((*(buf0+14))+val8.z);
  float alu55 = ((*(buf0+15))+val8.w);
  float alu56 = ((0.0f<alu52)?alu52:0.0f);
  float alu57 = ((0.0f<alu53)?alu53:0.0f);
  float alu58 = ((0.0f<alu54)?alu54:0.0f);
  float alu59 = ((0.0f<alu55)?alu55:0.0f);
  float alu60 = ((*(buf0+4))+val8.x);
  float alu61 = ((*(buf0+5))+val8.y);
  float alu62 = ((*(buf0+6))+val8.z);
  float alu63 = ((*(buf0+7))+val8.w);
  float alu64 = ((0.0f<alu60)?alu60:0.0f);
  float alu65 = ((0.0f<alu61)?alu61:0.0f);
  float alu66 = ((0.0f<alu62)?alu62:0.0f);
  float alu67 = ((0.0f<alu63)?alu63:0.0f);
  float alu68 = ((*(buf0+10))+val8.z);
  float alu69 = ((*(buf0+11))+val8.w);
  float alu70 = ((*(buf0+8))+val8.x);
  float alu71 = ((*(buf0+9))+val8.y);
  float alu72 = ((0.0f<alu68)?alu68:0.0f);
  float alu73 = ((0.0f<alu69)?alu69:0.0f);
  float alu74 = ((0.0f<alu70)?alu70:0.0f);
  float alu75 = ((0.0f<alu71)?alu71:0.0f);
  int alu76 = (lidx2+alu0+alu1+(lidx0*1344));
  write_imagef(data0_14_5376_4, (int2)(alu76,gidx1), (float4)(alu48,alu49,alu50,alu51));
  write_imagef(data0_14_5376_4, (int2)((alu76+24),gidx1), (float4)(alu64,alu65,alu66,alu67));
  write_imagef(data0_14_5376_4, (int2)((alu76+48),gidx1), (float4)(alu74,alu75,alu72,alu73));
  write_imagef(data0_14_5376_4, (int2)((alu76+72),gidx1), (float4)(alu56,alu57,alu58,alu59));
}
