fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,1792>;
@group(0) @binding(1)var<storage,read_write>data0_25088:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_225792:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2359296:array<f32>;
@compute @workgroup_size(16,4) fn r_8_7_16_4_4_7_288_4(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,28>;
  var buf1: array<f32,28>;
  var gidx0 = i32(gindex.x); /* 7 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(lidx1);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx1)<<6u))+bitcast<i32>((bitcast<u32>(lidx0)<<2u)));
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
  buf0[16] = 0.0f;
  buf0[17] = 0.0f;
  buf0[18] = 0.0f;
  buf0[19] = 0.0f;
  buf0[20] = 0.0f;
  buf0[21] = 0.0f;
  buf0[22] = 0.0f;
  buf0[23] = 0.0f;
  buf0[24] = 0.0f;
  buf0[25] = 0.0f;
  buf0[26] = 0.0f;
  buf0[27] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 288; Ridx0++) {
    var cast1 = bitcast<u32>(Ridx0);
    var alu29 = (bitcast<i32>((cast0<<2u))+bitcast<i32>((cast1<<4u))+(gidx0*32256));
    var val0 = data1_225792[alu29];
    var val1 = data1_225792[(alu29+1)];
    var val2 = data1_225792[(alu29+2)];
    var val3 = data1_225792[(alu29+3)];
    var val4 = data1_225792[(alu29+4608)];
    var val5 = data1_225792[(alu29+4609)];
    var val6 = data1_225792[(alu29+4610)];
    var val7 = data1_225792[(alu29+4611)];
    var val8 = data1_225792[(alu29+9216)];
    var val9 = data1_225792[(alu29+9217)];
    var val10 = data1_225792[(alu29+9218)];
    var val11 = data1_225792[(alu29+9219)];
    var val12 = data1_225792[(alu29+13824)];
    var val13 = data1_225792[(alu29+13825)];
    var val14 = data1_225792[(alu29+13826)];
    var val15 = data1_225792[(alu29+13827)];
    var val16 = data1_225792[(alu29+18432)];
    var val17 = data1_225792[(alu29+18433)];
    var val18 = data1_225792[(alu29+18434)];
    var val19 = data1_225792[(alu29+18435)];
    var val20 = data1_225792[(alu29+23040)];
    var val21 = data1_225792[(alu29+23041)];
    var val22 = data1_225792[(alu29+23042)];
    var val23 = data1_225792[(alu29+23043)];
    var val24 = data1_225792[(alu29+27648)];
    var val25 = data1_225792[(alu29+27649)];
    var val26 = data1_225792[(alu29+27650)];
    var val27 = data1_225792[(alu29+27651)];
    var alu30 = (alu0+bitcast<i32>((cast0<<11u))+bitcast<i32>((cast1<<13u)));
    var val28 = data2_2359296[alu30];
    var val29 = data2_2359296[(alu30+1)];
    var val30 = data2_2359296[(alu30+2)];
    var val31 = data2_2359296[(alu30+3)];
    var val32 = data2_2359296[(alu30+512)];
    var val33 = data2_2359296[(alu30+513)];
    var val34 = data2_2359296[(alu30+514)];
    var val35 = data2_2359296[(alu30+515)];
    var val36 = data2_2359296[(alu30+1024)];
    var val37 = data2_2359296[(alu30+1025)];
    var val38 = data2_2359296[(alu30+1026)];
    var val39 = data2_2359296[(alu30+1027)];
    var val40 = data2_2359296[(alu30+1536)];
    var val41 = data2_2359296[(alu30+1537)];
    var val42 = data2_2359296[(alu30+1538)];
    var val43 = data2_2359296[(alu30+1539)];
    buf0[0] = (buf0[0]+((val0*val28)+(val1*val32)+(val2*val36)+(val3*val40)));
    buf0[1] = (buf0[1]+((val4*val28)+(val5*val32)+(val6*val36)+(val7*val40)));
    buf0[2] = (buf0[2]+((val8*val28)+(val9*val32)+(val10*val36)+(val11*val40)));
    buf0[3] = (buf0[3]+((val12*val28)+(val13*val32)+(val14*val36)+(val15*val40)));
    buf0[4] = (buf0[4]+((val16*val28)+(val17*val32)+(val18*val36)+(val19*val40)));
    buf0[5] = (buf0[5]+((val20*val28)+(val21*val32)+(val22*val36)+(val23*val40)));
    buf0[6] = (buf0[6]+((val24*val28)+(val25*val32)+(val26*val36)+(val27*val40)));
    buf0[7] = (buf0[7]+((val0*val29)+(val1*val33)+(val2*val37)+(val3*val41)));
    buf0[8] = (buf0[8]+((val4*val29)+(val5*val33)+(val6*val37)+(val7*val41)));
    buf0[9] = (buf0[9]+((val8*val29)+(val9*val33)+(val10*val37)+(val11*val41)));
    buf0[10] = (buf0[10]+((val12*val29)+(val13*val33)+(val14*val37)+(val15*val41)));
    buf0[11] = (buf0[11]+((val16*val29)+(val17*val33)+(val18*val37)+(val19*val41)));
    buf0[12] = (buf0[12]+((val20*val29)+(val21*val33)+(val22*val37)+(val23*val41)));
    buf0[13] = (buf0[13]+((val24*val29)+(val25*val33)+(val26*val37)+(val27*val41)));
    buf0[14] = (buf0[14]+((val0*val30)+(val1*val34)+(val2*val38)+(val3*val42)));
    buf0[15] = (buf0[15]+((val4*val30)+(val5*val34)+(val6*val38)+(val7*val42)));
    buf0[16] = (buf0[16]+((val8*val30)+(val9*val34)+(val10*val38)+(val11*val42)));
    buf0[17] = (buf0[17]+((val12*val30)+(val13*val34)+(val14*val38)+(val15*val42)));
    buf0[18] = (buf0[18]+((val16*val30)+(val17*val34)+(val18*val38)+(val19*val42)));
    buf0[19] = (buf0[19]+((val20*val30)+(val21*val34)+(val22*val38)+(val23*val42)));
    buf0[20] = (buf0[20]+((val24*val30)+(val25*val34)+(val26*val38)+(val27*val42)));
    buf0[21] = (buf0[21]+((val0*val31)+(val1*val35)+(val2*val39)+(val3*val43)));
    buf0[22] = (buf0[22]+((val4*val31)+(val5*val35)+(val6*val39)+(val7*val43)));
    buf0[23] = (buf0[23]+((val8*val31)+(val9*val35)+(val10*val39)+(val11*val43)));
    buf0[24] = (buf0[24]+((val12*val31)+(val13*val35)+(val14*val39)+(val15*val43)));
    buf0[25] = (buf0[25]+((val16*val31)+(val17*val35)+(val18*val39)+(val19*val43)));
    buf0[26] = (buf0[26]+((val20*val31)+(val21*val35)+(val22*val39)+(val23*val43)));
    buf0[27] = (buf0[27]+((val24*val31)+(val25*val35)+(val26*val39)+(val27*val43)));
  }
  var alu60 = (lidx0*112);
  var alu61 = (alu60+(lidx1*28));
  buf2[(alu61+1)] = buf0[1];
  buf2[(alu61+2)] = buf0[2];
  buf2[(alu61+3)] = buf0[3];
  buf2[(alu61+4)] = buf0[4];
  buf2[(alu61+5)] = buf0[5];
  buf2[(alu61+6)] = buf0[6];
  buf2[(alu61+7)] = buf0[7];
  buf2[(alu61+8)] = buf0[8];
  buf2[(alu61+9)] = buf0[9];
  buf2[(alu61+10)] = buf0[10];
  buf2[(alu61+11)] = buf0[11];
  buf2[(alu61+12)] = buf0[12];
  buf2[(alu61+13)] = buf0[13];
  buf2[(alu61+14)] = buf0[14];
  buf2[(alu61+15)] = buf0[15];
  buf2[(alu61+16)] = buf0[16];
  buf2[(alu61+17)] = buf0[17];
  buf2[(alu61+18)] = buf0[18];
  buf2[(alu61+19)] = buf0[19];
  buf2[(alu61+20)] = buf0[20];
  buf2[(alu61+21)] = buf0[21];
  buf2[(alu61+22)] = buf0[22];
  buf2[(alu61+23)] = buf0[23];
  buf2[(alu61+24)] = buf0[24];
  buf2[(alu61+25)] = buf0[25];
  buf2[(alu61+26)] = buf0[26];
  buf2[(alu61+27)] = buf0[27];
  buf2[alu61] = buf0[0];
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
  buf1[16] = 0.0f;
  buf1[17] = 0.0f;
  buf1[18] = 0.0f;
  buf1[19] = 0.0f;
  buf1[20] = 0.0f;
  buf1[21] = 0.0f;
  buf1[22] = 0.0f;
  buf1[23] = 0.0f;
  buf1[24] = 0.0f;
  buf1[25] = 0.0f;
  buf1[26] = 0.0f;
  buf1[27] = 0.0f;
  for (var Ridx107 = 0; Ridx107 < 4; Ridx107++) {
    var alu119 = (alu60+(Ridx107*28));
    var val44 = buf2[(alu119+1)];
    var val45 = buf2[(alu119+2)];
    var val46 = buf2[(alu119+3)];
    var val47 = buf2[(alu119+4)];
    var val48 = buf2[(alu119+5)];
    var val49 = buf2[(alu119+6)];
    var val50 = buf2[(alu119+7)];
    var val51 = buf2[(alu119+8)];
    var val52 = buf2[(alu119+9)];
    var val53 = buf2[(alu119+10)];
    var val54 = buf2[(alu119+11)];
    var val55 = buf2[(alu119+12)];
    var val56 = buf2[(alu119+13)];
    var val57 = buf2[(alu119+14)];
    var val58 = buf2[(alu119+15)];
    var val59 = buf2[(alu119+16)];
    var val60 = buf2[(alu119+17)];
    var val61 = buf2[(alu119+18)];
    var val62 = buf2[(alu119+19)];
    var val63 = buf2[(alu119+20)];
    var val64 = buf2[(alu119+21)];
    var val65 = buf2[(alu119+22)];
    var val66 = buf2[(alu119+23)];
    var val67 = buf2[(alu119+24)];
    var val68 = buf2[(alu119+25)];
    var val69 = buf2[(alu119+26)];
    var val70 = buf2[(alu119+27)];
    var val71 = buf2[alu119];
    buf1[0] = (buf1[0]+val71);
    buf1[1] = (buf1[1]+val44);
    buf1[2] = (buf1[2]+val45);
    buf1[3] = (buf1[3]+val46);
    buf1[4] = (buf1[4]+val47);
    buf1[5] = (buf1[5]+val48);
    buf1[6] = (buf1[6]+val49);
    buf1[7] = (buf1[7]+val50);
    buf1[8] = (buf1[8]+val51);
    buf1[9] = (buf1[9]+val52);
    buf1[10] = (buf1[10]+val53);
    buf1[11] = (buf1[11]+val54);
    buf1[12] = (buf1[12]+val55);
    buf1[13] = (buf1[13]+val56);
    buf1[14] = (buf1[14]+val57);
    buf1[15] = (buf1[15]+val58);
    buf1[16] = (buf1[16]+val59);
    buf1[17] = (buf1[17]+val60);
    buf1[18] = (buf1[18]+val61);
    buf1[19] = (buf1[19]+val62);
    buf1[20] = (buf1[20]+val63);
    buf1[21] = (buf1[21]+val64);
    buf1[22] = (buf1[22]+val65);
    buf1[23] = (buf1[23]+val66);
    buf1[24] = (buf1[24]+val67);
    buf1[25] = (buf1[25]+val68);
    buf1[26] = (buf1[26]+val69);
    buf1[27] = (buf1[27]+val70);
  }
  var alu149 = (alu0+(gidx0*3584));
  var alu150 = (lidx1==0);
  if (alu150) {
    data0_25088[alu149] = buf1[0];
  }
  if (alu150) {
    data0_25088[(alu149+1)] = buf1[7];
  }
  if (alu150) {
    data0_25088[(alu149+2)] = buf1[14];
  }
  if (alu150) {
    data0_25088[(alu149+3)] = buf1[21];
  }
  if (alu150) {
    data0_25088[(alu149+512)] = buf1[1];
  }
  if (alu150) {
    data0_25088[(alu149+513)] = buf1[8];
  }
  if (alu150) {
    data0_25088[(alu149+514)] = buf1[15];
  }
  if (alu150) {
    data0_25088[(alu149+515)] = buf1[22];
  }
  if (alu150) {
    data0_25088[(alu149+1024)] = buf1[2];
  }
  if (alu150) {
    data0_25088[(alu149+1025)] = buf1[9];
  }
  if (alu150) {
    data0_25088[(alu149+1026)] = buf1[16];
  }
  if (alu150) {
    data0_25088[(alu149+1027)] = buf1[23];
  }
  if (alu150) {
    data0_25088[(alu149+1536)] = buf1[3];
  }
  if (alu150) {
    data0_25088[(alu149+1537)] = buf1[10];
  }
  if (alu150) {
    data0_25088[(alu149+1538)] = buf1[17];
  }
  if (alu150) {
    data0_25088[(alu149+1539)] = buf1[24];
  }
  if (alu150) {
    data0_25088[(alu149+2048)] = buf1[4];
  }
  if (alu150) {
    data0_25088[(alu149+2049)] = buf1[11];
  }
  if (alu150) {
    data0_25088[(alu149+2050)] = buf1[18];
  }
  if (alu150) {
    data0_25088[(alu149+2051)] = buf1[25];
  }
  if (alu150) {
    data0_25088[(alu149+2560)] = buf1[5];
  }
  if (alu150) {
    data0_25088[(alu149+2561)] = buf1[12];
  }
  if (alu150) {
    data0_25088[(alu149+2562)] = buf1[19];
  }
  if (alu150) {
    data0_25088[(alu149+2563)] = buf1[26];
  }
  if (alu150) {
    data0_25088[(alu149+3072)] = buf1[6];
  }
  if (alu150) {
    data0_25088[(alu149+3073)] = buf1[13];
  }
  if (alu150) {
    data0_25088[(alu149+3074)] = buf1[20];
  }
  if (alu150) {
    data0_25088[(alu149+3075)] = buf1[27];
  }
}