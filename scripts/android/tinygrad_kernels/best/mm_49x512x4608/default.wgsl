fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_25088:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_225792:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2359296:array<f32>;
@compute @workgroup_size(16) fn r_49_8_16_4_1152_4(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 49 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx0)<<2u)));
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 1152; Ridx0++) {
    var cast0 = bitcast<u32>(Ridx0);
    var alu5 = ((gidx1*4608)+bitcast<i32>((cast0<<2u)));
    var val0 = data1_225792[(alu5+1)];
    var val1 = data1_225792[(alu5+2)];
    var val2 = data1_225792[(alu5+3)];
    var val3 = data1_225792[alu5];
    var alu6 = (alu0+bitcast<i32>((cast0<<11u)));
    var val4 = data2_2359296[alu6];
    var val5 = data2_2359296[(alu6+1)];
    var val6 = data2_2359296[(alu6+2)];
    var val7 = data2_2359296[(alu6+3)];
    var val8 = data2_2359296[(alu6+512)];
    var val9 = data2_2359296[(alu6+513)];
    var val10 = data2_2359296[(alu6+514)];
    var val11 = data2_2359296[(alu6+515)];
    var val12 = data2_2359296[(alu6+1024)];
    var val13 = data2_2359296[(alu6+1025)];
    var val14 = data2_2359296[(alu6+1026)];
    var val15 = data2_2359296[(alu6+1027)];
    var val16 = data2_2359296[(alu6+1536)];
    var val17 = data2_2359296[(alu6+1537)];
    var val18 = data2_2359296[(alu6+1538)];
    var val19 = data2_2359296[(alu6+1539)];
    buf0[0] = (buf0[0]+((val3*val4)+(val0*val8)+(val1*val12)+(val2*val16)));
    buf0[1] = (buf0[1]+((val3*val5)+(val0*val9)+(val1*val13)+(val2*val17)));
    buf0[2] = (buf0[2]+((val3*val6)+(val0*val10)+(val1*val14)+(val2*val18)));
    buf0[3] = (buf0[3]+((val3*val7)+(val0*val11)+(val1*val15)+(val2*val19)));
  }
  var alu12 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<9u)));
  data0_25088[alu12] = buf0[0];
  data0_25088[(alu12+1)] = buf0[1];
  data0_25088[(alu12+2)] = buf0[2];
  data0_25088[(alu12+3)] = buf0[3];
}