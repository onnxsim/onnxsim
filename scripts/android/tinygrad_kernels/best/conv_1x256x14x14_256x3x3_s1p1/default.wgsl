fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_50176:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_50176:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_589824:array<f32>;
@compute @workgroup_size(32,2,2) fn r_2_7_7_32_2_2_4_256_3_3(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 7 */
  var gidx1 = i32(gindex.y); /* 7 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 32 */
  var lidx1 = i32(lindex.y); /* 2 */
  var lidx2 = i32(lindex.z); /* 2 */
  var alu0 = (lidx2+bitcast<i32>((bitcast<u32>(gidx0)<<1u)));
  var alu1 = (alu0+(gidx1*28)+(lidx1*14));
  var alu2 = (0<(gidx0+lidx2));
  var alu3 = (0<(gidx1+lidx1));
  var alu4 = ((lidx1+bitcast<i32>((bitcast<u32>(gidx1)<<1u)))<13);
  var alu5 = (alu0<13);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu10 = (alu1+(Ridx0*196));
    var val0 = data1_50176[alu10];
    var val1 = select(0.0f, data1_50176[(alu10+-15)], (alu2&alu3));
    var val2 = select(0.0f, data1_50176[(alu10+-14)], alu3);
    var val3 = select(0.0f, data1_50176[(alu10+-13)], (alu5&alu3));
    var val4 = select(0.0f, data1_50176[(alu10+-1)], alu2);
    var val5 = select(0.0f, data1_50176[(alu10+1)], alu5);
    var val6 = select(0.0f, data1_50176[(alu10+13)], (alu2&alu4));
    var val7 = select(0.0f, data1_50176[(alu10+14)], alu4);
    var val8 = select(0.0f, data1_50176[(alu10+15)], (alu5&alu4));
    var alu11 = ((gidx2*294912)+(lidx0*9216)+(Ridx0*9));
    var val9 = data2_589824[(alu11+1)];
    var val10 = data2_589824[(alu11+2)];
    var val11 = data2_589824[(alu11+3)];
    var val12 = data2_589824[(alu11+4)];
    var val13 = data2_589824[(alu11+5)];
    var val14 = data2_589824[(alu11+6)];
    var val15 = data2_589824[(alu11+7)];
    var val16 = data2_589824[(alu11+8)];
    var val17 = data2_589824[(alu11+2304)];
    var val18 = data2_589824[(alu11+2305)];
    var val19 = data2_589824[(alu11+2306)];
    var val20 = data2_589824[(alu11+2307)];
    var val21 = data2_589824[(alu11+2308)];
    var val22 = data2_589824[(alu11+2309)];
    var val23 = data2_589824[(alu11+2310)];
    var val24 = data2_589824[(alu11+2311)];
    var val25 = data2_589824[(alu11+2312)];
    var val26 = data2_589824[(alu11+4608)];
    var val27 = data2_589824[(alu11+4609)];
    var val28 = data2_589824[(alu11+4610)];
    var val29 = data2_589824[(alu11+4611)];
    var val30 = data2_589824[(alu11+4612)];
    var val31 = data2_589824[(alu11+4613)];
    var val32 = data2_589824[(alu11+4614)];
    var val33 = data2_589824[(alu11+4615)];
    var val34 = data2_589824[(alu11+4616)];
    var val35 = data2_589824[(alu11+6912)];
    var val36 = data2_589824[(alu11+6913)];
    var val37 = data2_589824[(alu11+6914)];
    var val38 = data2_589824[(alu11+6915)];
    var val39 = data2_589824[(alu11+6916)];
    var val40 = data2_589824[(alu11+6917)];
    var val41 = data2_589824[(alu11+6918)];
    var val42 = data2_589824[(alu11+6919)];
    var val43 = data2_589824[(alu11+6920)];
    var val44 = data2_589824[alu11];
    buf0[0] = (buf0[0]+((val1*val44)+(val2*val9)+(val3*val10)+(val4*val11)+(val0*val12)+(val5*val13)+(val6*val14)+(val7*val15)+(val8*val16)));
    buf0[1] = (buf0[1]+((val1*val17)+(val2*val18)+(val3*val19)+(val4*val20)+(val0*val21)+(val5*val22)+(val6*val23)+(val7*val24)+(val8*val25)));
    buf0[2] = (buf0[2]+((val1*val26)+(val2*val27)+(val3*val28)+(val4*val29)+(val0*val30)+(val5*val31)+(val6*val32)+(val7*val33)+(val8*val34)));
    buf0[3] = (buf0[3]+((val1*val35)+(val2*val36)+(val3*val37)+(val4*val38)+(val0*val39)+(val5*val40)+(val6*val41)+(val7*val42)+(val8*val43)));
  }
  var alu17 = (alu1+(gidx2*25088)+(lidx0*784));
  data0_50176[alu17] = buf0[0];
  data0_50176[(alu17+196)] = buf0[1];
  data0_50176[(alu17+392)] = buf0[2];
  data0_50176[(alu17+588)] = buf0[3];
}