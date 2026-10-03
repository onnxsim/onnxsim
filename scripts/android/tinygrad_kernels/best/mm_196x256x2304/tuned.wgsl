fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,2048>;
@group(0) @binding(1)var<storage,read_write>data0_50176:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_451584:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_589824:array<f32>;
@compute @workgroup_size(16,8) fn r_4_49_16_8_4_4_72_4(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var buf1: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 49 */
  var gidx1 = i32(gindex.y); /* 4 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var cast0 = bitcast<u32>(lidx0);
  var cast1 = bitcast<u32>(lidx1);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx1)<<6u))+bitcast<i32>((cast0<<2u)));
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 72; Ridx0++) {
    var cast2 = bitcast<u32>(Ridx0);
    var alu17 = (bitcast<i32>((cast1<<2u))+bitcast<i32>((cast2<<5u))+(gidx0*9216));
    var val0 = data1_451584[alu17];
    var val1 = data1_451584[(alu17+1)];
    var val2 = data1_451584[(alu17+2)];
    var val3 = data1_451584[(alu17+3)];
    var val4 = data1_451584[(alu17+2304)];
    var val5 = data1_451584[(alu17+2305)];
    var val6 = data1_451584[(alu17+2306)];
    var val7 = data1_451584[(alu17+2307)];
    var val8 = data1_451584[(alu17+4608)];
    var val9 = data1_451584[(alu17+4609)];
    var val10 = data1_451584[(alu17+4610)];
    var val11 = data1_451584[(alu17+4611)];
    var val12 = data1_451584[(alu17+6912)];
    var val13 = data1_451584[(alu17+6913)];
    var val14 = data1_451584[(alu17+6914)];
    var val15 = data1_451584[(alu17+6915)];
    var alu18 = (alu0+bitcast<i32>((cast1<<10u))+bitcast<i32>((cast2<<13u)));
    var val16 = data2_589824[alu18];
    var val17 = data2_589824[(alu18+1)];
    var val18 = data2_589824[(alu18+2)];
    var val19 = data2_589824[(alu18+3)];
    var val20 = data2_589824[(alu18+256)];
    var val21 = data2_589824[(alu18+257)];
    var val22 = data2_589824[(alu18+258)];
    var val23 = data2_589824[(alu18+259)];
    var val24 = data2_589824[(alu18+512)];
    var val25 = data2_589824[(alu18+513)];
    var val26 = data2_589824[(alu18+514)];
    var val27 = data2_589824[(alu18+515)];
    var val28 = data2_589824[(alu18+768)];
    var val29 = data2_589824[(alu18+769)];
    var val30 = data2_589824[(alu18+770)];
    var val31 = data2_589824[(alu18+771)];
    buf0[0] = (buf0[0]+((val0*val16)+(val1*val20)+(val2*val24)+(val3*val28)));
    buf0[1] = (buf0[1]+((val4*val16)+(val5*val20)+(val6*val24)+(val7*val28)));
    buf0[2] = (buf0[2]+((val8*val16)+(val9*val20)+(val10*val24)+(val11*val28)));
    buf0[3] = (buf0[3]+((val12*val16)+(val13*val20)+(val14*val24)+(val15*val28)));
    buf0[4] = (buf0[4]+((val0*val17)+(val1*val21)+(val2*val25)+(val3*val29)));
    buf0[5] = (buf0[5]+((val4*val17)+(val5*val21)+(val6*val25)+(val7*val29)));
    buf0[6] = (buf0[6]+((val8*val17)+(val9*val21)+(val10*val25)+(val11*val29)));
    buf0[7] = (buf0[7]+((val12*val17)+(val13*val21)+(val14*val25)+(val15*val29)));
    buf0[8] = (buf0[8]+((val0*val18)+(val1*val22)+(val2*val26)+(val3*val30)));
    buf0[9] = (buf0[9]+((val4*val18)+(val5*val22)+(val6*val26)+(val7*val30)));
    buf0[10] = (buf0[10]+((val8*val18)+(val9*val22)+(val10*val26)+(val11*val30)));
    buf0[11] = (buf0[11]+((val12*val18)+(val13*val22)+(val14*val26)+(val15*val30)));
    buf0[12] = (buf0[12]+((val0*val19)+(val1*val23)+(val2*val27)+(val3*val31)));
    buf0[13] = (buf0[13]+((val4*val19)+(val5*val23)+(val6*val27)+(val7*val31)));
    buf0[14] = (buf0[14]+((val8*val19)+(val9*val23)+(val10*val27)+(val11*val31)));
    buf0[15] = (buf0[15]+((val12*val19)+(val13*val23)+(val14*val27)+(val15*val31)));
  }
  var cast3 = bitcast<i32>((cast0<<7u));
  var alu36 = (cast3+bitcast<i32>((cast1<<4u)));
  buf2[alu36] = buf0[0];
  buf2[(alu36+1)] = buf0[1];
  buf2[(alu36+2)] = buf0[2];
  buf2[(alu36+3)] = buf0[3];
  buf2[(alu36+4)] = buf0[4];
  buf2[(alu36+5)] = buf0[5];
  buf2[(alu36+6)] = buf0[6];
  buf2[(alu36+7)] = buf0[7];
  buf2[(alu36+8)] = buf0[8];
  buf2[(alu36+9)] = buf0[9];
  buf2[(alu36+10)] = buf0[10];
  buf2[(alu36+11)] = buf0[11];
  buf2[(alu36+12)] = buf0[12];
  buf2[(alu36+13)] = buf0[13];
  buf2[(alu36+14)] = buf0[14];
  buf2[(alu36+15)] = buf0[15];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  buf1[4] = 0.0f;
  buf1[5] = 0.0f;
  buf1[6] = 0.0f;
  buf1[7] = 0.0f;
  buf1[8] = 0.0f;
  buf1[9] = 0.0f;
  buf1[10] = 0.0f;
  buf1[11] = 0.0f;
  buf1[12] = 0.0f;
  buf1[13] = 0.0f;
  buf1[14] = 0.0f;
  buf1[15] = 0.0f;
  for (var Ridx107 = 0; Ridx107 < 8; Ridx107++) {
    var alu70 = (cast3+bitcast<i32>((bitcast<u32>(Ridx107)<<4u)));
    var val32 = buf2[alu70];
    var val33 = buf2[(alu70+1)];
    var val34 = buf2[(alu70+2)];
    var val35 = buf2[(alu70+3)];
    var val36 = buf2[(alu70+4)];
    var val37 = buf2[(alu70+5)];
    var val38 = buf2[(alu70+6)];
    var val39 = buf2[(alu70+7)];
    var val40 = buf2[(alu70+8)];
    var val41 = buf2[(alu70+9)];
    var val42 = buf2[(alu70+10)];
    var val43 = buf2[(alu70+11)];
    var val44 = buf2[(alu70+12)];
    var val45 = buf2[(alu70+13)];
    var val46 = buf2[(alu70+14)];
    var val47 = buf2[(alu70+15)];
    buf1[0] = (buf1[0]+val32);
    buf1[1] = (buf1[1]+val33);
    buf1[2] = (buf1[2]+val34);
    buf1[3] = (buf1[3]+val35);
    buf1[4] = (buf1[4]+val36);
    buf1[5] = (buf1[5]+val37);
    buf1[6] = (buf1[6]+val38);
    buf1[7] = (buf1[7]+val39);
    buf1[8] = (buf1[8]+val40);
    buf1[9] = (buf1[9]+val41);
    buf1[10] = (buf1[10]+val42);
    buf1[11] = (buf1[11]+val43);
    buf1[12] = (buf1[12]+val44);
    buf1[13] = (buf1[13]+val45);
    buf1[14] = (buf1[14]+val46);
    buf1[15] = (buf1[15]+val47);
  }
  var alu88 = (alu0+bitcast<i32>((bitcast<u32>(gidx0)<<10u)));
  var alu89 = (lidx1==0);
  if (alu89) {
    data0_50176[alu88] = buf1[0];
  }
  if (alu89) {
    data0_50176[(alu88+1)] = buf1[4];
  }
  if (alu89) {
    data0_50176[(alu88+2)] = buf1[8];
  }
  if (alu89) {
    data0_50176[(alu88+3)] = buf1[12];
  }
  if (alu89) {
    data0_50176[(alu88+256)] = buf1[1];
  }
  if (alu89) {
    data0_50176[(alu88+257)] = buf1[5];
  }
  if (alu89) {
    data0_50176[(alu88+258)] = buf1[9];
  }
  if (alu89) {
    data0_50176[(alu88+259)] = buf1[13];
  }
  if (alu89) {
    data0_50176[(alu88+512)] = buf1[2];
  }
  if (alu89) {
    data0_50176[(alu88+513)] = buf1[6];
  }
  if (alu89) {
    data0_50176[(alu88+514)] = buf1[10];
  }
  if (alu89) {
    data0_50176[(alu88+515)] = buf1[14];
  }
  if (alu89) {
    data0_50176[(alu88+768)] = buf1[3];
  }
  if (alu89) {
    data0_50176[(alu88+769)] = buf1[7];
  }
  if (alu89) {
    data0_50176[(alu88+770)] = buf1[11];
  }
  if (alu89) {
    data0_50176[(alu88+771)] = buf1[15];
  }
}