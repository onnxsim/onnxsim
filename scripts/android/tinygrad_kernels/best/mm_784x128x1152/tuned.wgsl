fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,4096>;
@group(0) @binding(1)var<storage,read_write>data0_100352:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_903168:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_147456:array<f32>;
@compute @workgroup_size(4,8,4) fn r_49_2_4_8_4_4_4_2_72_4(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,32>;
  var buf1: array<f32,32>;
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 49 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 8 */
  var lidx2 = i32(lindex.z); /* 4 */
  var cast0 = bitcast<u32>(lidx1);
  var cast1 = bitcast<u32>(lidx2);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((cast0<<3u)));
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
  buf0[28] = 0.0f;
  buf0[29] = 0.0f;
  buf0[30] = 0.0f;
  buf0[31] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 72; Ridx0++) {
    var cast2 = bitcast<u32>(Ridx0);
    var alu33 = ((gidx1*18432)+(lidx0*4608)+bitcast<i32>((cast1<<2u))+bitcast<i32>((cast2<<4u)));
    var val0 = data1_903168[(alu33+1)];
    var val1 = data1_903168[(alu33+2)];
    var val2 = data1_903168[(alu33+3)];
    var val3 = data1_903168[(alu33+1152)];
    var val4 = data1_903168[(alu33+1153)];
    var val5 = data1_903168[(alu33+1154)];
    var val6 = data1_903168[(alu33+1155)];
    var val7 = data1_903168[(alu33+2304)];
    var val8 = data1_903168[(alu33+2305)];
    var val9 = data1_903168[(alu33+2306)];
    var val10 = data1_903168[(alu33+2307)];
    var val11 = data1_903168[(alu33+3456)];
    var val12 = data1_903168[(alu33+3457)];
    var val13 = data1_903168[(alu33+3458)];
    var val14 = data1_903168[(alu33+3459)];
    var val15 = data1_903168[alu33];
    var alu34 = (alu0+bitcast<i32>((cast1<<9u))+bitcast<i32>((cast2<<11u)));
    var val16 = data2_147456[alu34];
    var val17 = data2_147456[(alu34+1)];
    var val18 = data2_147456[(alu34+2)];
    var val19 = data2_147456[(alu34+3)];
    var val20 = data2_147456[(alu34+4)];
    var val21 = data2_147456[(alu34+5)];
    var val22 = data2_147456[(alu34+6)];
    var val23 = data2_147456[(alu34+7)];
    var val24 = data2_147456[(alu34+128)];
    var val25 = data2_147456[(alu34+129)];
    var val26 = data2_147456[(alu34+130)];
    var val27 = data2_147456[(alu34+131)];
    var val28 = data2_147456[(alu34+132)];
    var val29 = data2_147456[(alu34+133)];
    var val30 = data2_147456[(alu34+134)];
    var val31 = data2_147456[(alu34+135)];
    var val32 = data2_147456[(alu34+256)];
    var val33 = data2_147456[(alu34+257)];
    var val34 = data2_147456[(alu34+258)];
    var val35 = data2_147456[(alu34+259)];
    var val36 = data2_147456[(alu34+260)];
    var val37 = data2_147456[(alu34+261)];
    var val38 = data2_147456[(alu34+262)];
    var val39 = data2_147456[(alu34+263)];
    var val40 = data2_147456[(alu34+384)];
    var val41 = data2_147456[(alu34+385)];
    var val42 = data2_147456[(alu34+386)];
    var val43 = data2_147456[(alu34+387)];
    var val44 = data2_147456[(alu34+388)];
    var val45 = data2_147456[(alu34+389)];
    var val46 = data2_147456[(alu34+390)];
    var val47 = data2_147456[(alu34+391)];
    buf0[0] = (buf0[0]+((val15*val16)+(val0*val24)+(val1*val32)+(val2*val40)));
    buf0[1] = (buf0[1]+((val3*val16)+(val4*val24)+(val5*val32)+(val6*val40)));
    buf0[2] = (buf0[2]+((val7*val16)+(val8*val24)+(val9*val32)+(val10*val40)));
    buf0[3] = (buf0[3]+((val11*val16)+(val12*val24)+(val13*val32)+(val14*val40)));
    buf0[4] = (buf0[4]+((val15*val17)+(val0*val25)+(val1*val33)+(val2*val41)));
    buf0[5] = (buf0[5]+((val3*val17)+(val4*val25)+(val5*val33)+(val6*val41)));
    buf0[6] = (buf0[6]+((val7*val17)+(val8*val25)+(val9*val33)+(val10*val41)));
    buf0[7] = (buf0[7]+((val11*val17)+(val12*val25)+(val13*val33)+(val14*val41)));
    buf0[8] = (buf0[8]+((val15*val18)+(val0*val26)+(val1*val34)+(val2*val42)));
    buf0[9] = (buf0[9]+((val3*val18)+(val4*val26)+(val5*val34)+(val6*val42)));
    buf0[10] = (buf0[10]+((val7*val18)+(val8*val26)+(val9*val34)+(val10*val42)));
    buf0[11] = (buf0[11]+((val11*val18)+(val12*val26)+(val13*val34)+(val14*val42)));
    buf0[12] = (buf0[12]+((val15*val19)+(val0*val27)+(val1*val35)+(val2*val43)));
    buf0[13] = (buf0[13]+((val3*val19)+(val4*val27)+(val5*val35)+(val6*val43)));
    buf0[14] = (buf0[14]+((val7*val19)+(val8*val27)+(val9*val35)+(val10*val43)));
    buf0[15] = (buf0[15]+((val11*val19)+(val12*val27)+(val13*val35)+(val14*val43)));
    buf0[16] = (buf0[16]+((val15*val20)+(val0*val28)+(val1*val36)+(val2*val44)));
    buf0[17] = (buf0[17]+((val3*val20)+(val4*val28)+(val5*val36)+(val6*val44)));
    buf0[18] = (buf0[18]+((val7*val20)+(val8*val28)+(val9*val36)+(val10*val44)));
    buf0[19] = (buf0[19]+((val11*val20)+(val12*val28)+(val13*val36)+(val14*val44)));
    buf0[20] = (buf0[20]+((val15*val21)+(val0*val29)+(val1*val37)+(val2*val45)));
    buf0[21] = (buf0[21]+((val3*val21)+(val4*val29)+(val5*val37)+(val6*val45)));
    buf0[22] = (buf0[22]+((val7*val21)+(val8*val29)+(val9*val37)+(val10*val45)));
    buf0[23] = (buf0[23]+((val11*val21)+(val12*val29)+(val13*val37)+(val14*val45)));
    buf0[24] = (buf0[24]+((val15*val22)+(val0*val30)+(val1*val38)+(val2*val46)));
    buf0[25] = (buf0[25]+((val3*val22)+(val4*val30)+(val5*val38)+(val6*val46)));
    buf0[26] = (buf0[26]+((val7*val22)+(val8*val30)+(val9*val38)+(val10*val46)));
    buf0[27] = (buf0[27]+((val11*val22)+(val12*val30)+(val13*val38)+(val14*val46)));
    buf0[28] = (buf0[28]+((val15*val23)+(val0*val31)+(val1*val39)+(val2*val47)));
    buf0[29] = (buf0[29]+((val3*val23)+(val4*val31)+(val5*val39)+(val6*val47)));
    buf0[30] = (buf0[30]+((val7*val23)+(val8*val31)+(val9*val39)+(val10*val47)));
    buf0[31] = (buf0[31]+((val11*val23)+(val12*val31)+(val13*val39)+(val14*val47)));
  }
  var cast3 = bitcast<u32>(lidx0);
  var cast4 = bitcast<i32>((cast3<<10u));
  var cast5 = bitcast<i32>((cast0<<7u));
  var alu68 = (cast5+bitcast<i32>((cast1<<5u))+cast4);
  buf2[alu68] = buf0[0];
  buf2[(alu68+1)] = buf0[1];
  buf2[(alu68+2)] = buf0[2];
  buf2[(alu68+3)] = buf0[3];
  buf2[(alu68+4)] = buf0[4];
  buf2[(alu68+5)] = buf0[5];
  buf2[(alu68+6)] = buf0[6];
  buf2[(alu68+7)] = buf0[7];
  buf2[(alu68+8)] = buf0[8];
  buf2[(alu68+9)] = buf0[9];
  buf2[(alu68+10)] = buf0[10];
  buf2[(alu68+11)] = buf0[11];
  buf2[(alu68+12)] = buf0[12];
  buf2[(alu68+13)] = buf0[13];
  buf2[(alu68+14)] = buf0[14];
  buf2[(alu68+15)] = buf0[15];
  buf2[(alu68+16)] = buf0[16];
  buf2[(alu68+17)] = buf0[17];
  buf2[(alu68+18)] = buf0[18];
  buf2[(alu68+19)] = buf0[19];
  buf2[(alu68+20)] = buf0[20];
  buf2[(alu68+21)] = buf0[21];
  buf2[(alu68+22)] = buf0[22];
  buf2[(alu68+23)] = buf0[23];
  buf2[(alu68+24)] = buf0[24];
  buf2[(alu68+25)] = buf0[25];
  buf2[(alu68+26)] = buf0[26];
  buf2[(alu68+27)] = buf0[27];
  buf2[(alu68+28)] = buf0[28];
  buf2[(alu68+29)] = buf0[29];
  buf2[(alu68+30)] = buf0[30];
  buf2[(alu68+31)] = buf0[31];
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
  buf1[28] = 0.0f;
  buf1[29] = 0.0f;
  buf1[30] = 0.0f;
  buf1[31] = 0.0f;
  for (var Ridx108 = 0; Ridx108 < 4; Ridx108++) {
    var alu134 = (cast5+bitcast<i32>((bitcast<u32>(Ridx108)<<5u))+cast4);
    var val48 = buf2[alu134];
    var val49 = buf2[(alu134+1)];
    var val50 = buf2[(alu134+2)];
    var val51 = buf2[(alu134+3)];
    var val52 = buf2[(alu134+4)];
    var val53 = buf2[(alu134+5)];
    var val54 = buf2[(alu134+6)];
    var val55 = buf2[(alu134+7)];
    var val56 = buf2[(alu134+8)];
    var val57 = buf2[(alu134+9)];
    var val58 = buf2[(alu134+10)];
    var val59 = buf2[(alu134+11)];
    var val60 = buf2[(alu134+12)];
    var val61 = buf2[(alu134+13)];
    var val62 = buf2[(alu134+14)];
    var val63 = buf2[(alu134+15)];
    var val64 = buf2[(alu134+16)];
    var val65 = buf2[(alu134+17)];
    var val66 = buf2[(alu134+18)];
    var val67 = buf2[(alu134+19)];
    var val68 = buf2[(alu134+20)];
    var val69 = buf2[(alu134+21)];
    var val70 = buf2[(alu134+22)];
    var val71 = buf2[(alu134+23)];
    var val72 = buf2[(alu134+24)];
    var val73 = buf2[(alu134+25)];
    var val74 = buf2[(alu134+26)];
    var val75 = buf2[(alu134+27)];
    var val76 = buf2[(alu134+28)];
    var val77 = buf2[(alu134+29)];
    var val78 = buf2[(alu134+30)];
    var val79 = buf2[(alu134+31)];
    buf1[0] = (buf1[0]+val48);
    buf1[1] = (buf1[1]+val49);
    buf1[2] = (buf1[2]+val50);
    buf1[3] = (buf1[3]+val51);
    buf1[4] = (buf1[4]+val52);
    buf1[5] = (buf1[5]+val53);
    buf1[6] = (buf1[6]+val54);
    buf1[7] = (buf1[7]+val55);
    buf1[8] = (buf1[8]+val56);
    buf1[9] = (buf1[9]+val57);
    buf1[10] = (buf1[10]+val58);
    buf1[11] = (buf1[11]+val59);
    buf1[12] = (buf1[12]+val60);
    buf1[13] = (buf1[13]+val61);
    buf1[14] = (buf1[14]+val62);
    buf1[15] = (buf1[15]+val63);
    buf1[16] = (buf1[16]+val64);
    buf1[17] = (buf1[17]+val65);
    buf1[18] = (buf1[18]+val66);
    buf1[19] = (buf1[19]+val67);
    buf1[20] = (buf1[20]+val68);
    buf1[21] = (buf1[21]+val69);
    buf1[22] = (buf1[22]+val70);
    buf1[23] = (buf1[23]+val71);
    buf1[24] = (buf1[24]+val72);
    buf1[25] = (buf1[25]+val73);
    buf1[26] = (buf1[26]+val74);
    buf1[27] = (buf1[27]+val75);
    buf1[28] = (buf1[28]+val76);
    buf1[29] = (buf1[29]+val77);
    buf1[30] = (buf1[30]+val78);
    buf1[31] = (buf1[31]+val79);
  }
  var alu168 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<11u))+bitcast<i32>((cast3<<9u)));
  var alu169 = (lidx2==0);
  if (alu169) {
    data0_100352[alu168] = buf1[0];
  }
  if (alu169) {
    data0_100352[(alu168+1)] = buf1[4];
  }
  if (alu169) {
    data0_100352[(alu168+2)] = buf1[8];
  }
  if (alu169) {
    data0_100352[(alu168+3)] = buf1[12];
  }
  if (alu169) {
    data0_100352[(alu168+4)] = buf1[16];
  }
  if (alu169) {
    data0_100352[(alu168+5)] = buf1[20];
  }
  if (alu169) {
    data0_100352[(alu168+6)] = buf1[24];
  }
  if (alu169) {
    data0_100352[(alu168+7)] = buf1[28];
  }
  if (alu169) {
    data0_100352[(alu168+128)] = buf1[1];
  }
  if (alu169) {
    data0_100352[(alu168+129)] = buf1[5];
  }
  if (alu169) {
    data0_100352[(alu168+130)] = buf1[9];
  }
  if (alu169) {
    data0_100352[(alu168+131)] = buf1[13];
  }
  if (alu169) {
    data0_100352[(alu168+132)] = buf1[17];
  }
  if (alu169) {
    data0_100352[(alu168+133)] = buf1[21];
  }
  if (alu169) {
    data0_100352[(alu168+134)] = buf1[25];
  }
  if (alu169) {
    data0_100352[(alu168+135)] = buf1[29];
  }
  if (alu169) {
    data0_100352[(alu168+256)] = buf1[2];
  }
  if (alu169) {
    data0_100352[(alu168+257)] = buf1[6];
  }
  if (alu169) {
    data0_100352[(alu168+258)] = buf1[10];
  }
  if (alu169) {
    data0_100352[(alu168+259)] = buf1[14];
  }
  if (alu169) {
    data0_100352[(alu168+260)] = buf1[18];
  }
  if (alu169) {
    data0_100352[(alu168+261)] = buf1[22];
  }
  if (alu169) {
    data0_100352[(alu168+262)] = buf1[26];
  }
  if (alu169) {
    data0_100352[(alu168+263)] = buf1[30];
  }
  if (alu169) {
    data0_100352[(alu168+384)] = buf1[3];
  }
  if (alu169) {
    data0_100352[(alu168+385)] = buf1[7];
  }
  if (alu169) {
    data0_100352[(alu168+386)] = buf1[11];
  }
  if (alu169) {
    data0_100352[(alu168+387)] = buf1[15];
  }
  if (alu169) {
    data0_100352[(alu168+388)] = buf1[19];
  }
  if (alu169) {
    data0_100352[(alu168+389)] = buf1[23];
  }
  if (alu169) {
    data0_100352[(alu168+390)] = buf1[27];
  }
  if (alu169) {
    data0_100352[(alu168+391)] = buf1[31];
  }
}