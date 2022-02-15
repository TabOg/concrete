// RUN: concretecompiler --passes MANP --action=dump-fhe --split-input-file %s 2>&1 | FileCheck %s

func @single_cst_add_eint_int(%t: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  %cst = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>

  // CHECK: %[[ret:.*]] = "FHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  %0 = "FHELinalg.add_eint_int"(%t, %cst) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func @single_dyn_add_eint_int(%e: tensor<8x!FHE.eint<2>>, %i: tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
{
  // CHECK: %[[ret:.*]] = "FHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 9 : ui{{[0-9]+}}} : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  %0 = "FHELinalg.add_eint_int"(%e, %i) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func @single_add_eint(%e0: tensor<8x!FHE.eint<2>>, %e1: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  // CHECK: %[[ret:.*]] = "FHELinalg.add_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 2 : ui{{[0-9]+}}} : (tensor<8x!FHE.eint<2>>, tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
  %0 = "FHELinalg.add_eint"(%e0, %e1) : (tensor<8x!FHE.eint<2>>, tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func @single_cst_sub_int_eint(%e: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  %cst = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>

  // CHECK: %[[ret:.*]] = "FHELinalg.sub_int_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (tensor<8xi3>, tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
  %0 = "FHELinalg.sub_int_eint"(%cst, %e) : (tensor<8xi3>, tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func @single_neg_eint(%e: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  // CHECK: %[[ret:.*]] = "FHELinalg.neg_eint"(%[[op0:.*]]) {MANP = 1 : ui{{[0-9]+}}} : (tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
  %0 = "FHELinalg.neg_eint"(%e) : (tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func @single_dyn_sub_int_eint(%e: tensor<8x!FHE.eint<2>>, %i: tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
{
  // CHECK: %[[ret:.*]] = "FHELinalg.sub_int_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 9 : ui{{[0-9]+}}} : (tensor<8xi3>, tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
  %0 = "FHELinalg.sub_int_eint"(%i, %e) : (tensor<8xi3>, tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func @single_cst_mul_eint_int(%e: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  %cst = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>

  // %0 = "FHELinalg.mul_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 3 : ui{{[0-9]+}}} : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  %0 = "FHELinalg.mul_eint_int"(%e, %cst) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func @single_dyn_mul_eint_int(%e: tensor<8x!FHE.eint<2>>, %i: tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
{
  // CHECK: %[[ret:.*]] = "FHELinalg.mul_eint_int"([[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  %0 = "FHELinalg.mul_eint_int"(%e, %i) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func @chain_add_eint_int(%e: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  %cst0 = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>
  %cst1 = arith.constant dense<[0, 7, 2, 5, 6, 2, 1, 7]> : tensor<8xi3>
  %cst2 = arith.constant dense<[0, 1, 2, 0, 1, 2, 0, 1]> : tensor<8xi3>
  %cst3 = arith.constant dense<[0, 1, 1, 0, 0, 1, 0, 1]> : tensor<8xi3>
  // CHECK: %[[ret:.*]] = "FHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  %0 = "FHELinalg.add_eint_int"(%e, %cst0) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  // CHECK-NEXT: %[[ret:.*]] = "FHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  %1 = "FHELinalg.add_eint_int"(%0, %cst1) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  // CHECK-NEXT: %[[ret:.*]] = "FHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  %2 = "FHELinalg.add_eint_int"(%1, %cst2) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  // CHECK-NEXT: %[[ret:.*]] = "FHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  %3 = "FHELinalg.add_eint_int"(%2, %cst3) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  return %3 : tensor<8x!FHE.eint<2>>
}

// -----

func @chain_add_eint_int_neg_eint(%e: tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
{
  %cst0 = arith.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>
  // CHECK: %[[ret:.*]] = "FHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  %0 = "FHELinalg.add_eint_int"(%e, %cst0) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  // CHECK-NEXT: %[[ret:.*]] = "FHELinalg.neg_eint"(%[[op0:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
  %1 = "FHELinalg.neg_eint"(%0) : (tensor<8x!FHE.eint<2>>) -> tensor<8x!FHE.eint<2>>
  return %1 : tensor<8x!FHE.eint<2>>
}

// -----

/////////////////////////////////////////////////
// FHELinalg.apply_multi_lookup_table
/////////////////////////////////////////////////

func @apply_lookup_table(%t: tensor<3x3x!FHE.eint<2>>) -> tensor<3x3x!FHE.eint<3>> {
  %lut = arith.constant dense<[1,3,5,7]> : tensor<4xi64>
  // CHECK: %[[RES:.*]] = "FHELinalg.apply_lookup_table"(%[[T:.*]], %[[LUT:.*]]) {MANP = 1 : ui1} : (tensor<3x3x!FHE.eint<2>>, tensor<4xi64>) -> tensor<3x3x!FHE.eint<3>>
  %res = "FHELinalg.apply_lookup_table"(%t, %lut) : (tensor<3x3x!FHE.eint<2>>, tensor<4xi64>) -> tensor<3x3x!FHE.eint<3>>
  return %res : tensor<3x3x!FHE.eint<3>>
}

// -----

func @apply_lookup_table_after_op(%t: tensor<8x!FHE.eint<2>>, %i: tensor<8xi3>) -> tensor<8x!FHE.eint<3>> {
  %lut = arith.constant dense<[1,3,5,7]> : tensor<4xi64>
  // CHECK: %[[V0:.*]] = "FHELinalg.mul_eint_int"([[T:.*]], %[[I:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  %0 = "FHELinalg.mul_eint_int"(%t, %i) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  // CHECK-NEXT: %[[RES:.*]] = "FHELinalg.apply_lookup_table"(%[[V0]], %[[LUT:.*]]) {MANP = 1 : ui1} : (tensor<8x!FHE.eint<2>>, tensor<4xi64>) -> tensor<8x!FHE.eint<3>>
  %res = "FHELinalg.apply_lookup_table"(%0, %lut) : (tensor<8x!FHE.eint<2>>, tensor<4xi64>) -> tensor<8x!FHE.eint<3>>
  return %res : tensor<8x!FHE.eint<3>>
}

// -----


func @apply_multi_lookup_table(%t: tensor<3x3x!FHE.eint<2>>, %luts: tensor<3x3x4xi64>) -> tensor<3x3x!FHE.eint<3>> {
  // CHECK: %[[RES:.*]] = "FHELinalg.apply_multi_lookup_table"(%[[T:.*]], %[[LUT:.*]]) {MANP = 1 : ui1} : (tensor<3x3x!FHE.eint<2>>, tensor<3x3x4xi64>) -> tensor<3x3x!FHE.eint<3>>
  %res = "FHELinalg.apply_multi_lookup_table"(%t, %luts) : (tensor<3x3x!FHE.eint<2>>, tensor<3x3x4xi64>) -> tensor<3x3x!FHE.eint<3>>
  return %res : tensor<3x3x!FHE.eint<3>>
}

// -----

func @apply_multi_lookup_table_after_op(%t: tensor<8x!FHE.eint<2>>, %i: tensor<8xi3>, %luts: tensor<8x4xi64>) -> tensor<8x!FHE.eint<3>> {
  // CHECK: %[[V0:.*]] = "FHELinalg.mul_eint_int"([[T:.*]], %[[I:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  %0 = "FHELinalg.mul_eint_int"(%t, %i) : (tensor<8x!FHE.eint<2>>, tensor<8xi3>) -> tensor<8x!FHE.eint<2>>
  // CHECK-NEXT: %[[RES:.*]] = "FHELinalg.apply_multi_lookup_table"(%[[V0]], %[[LUT:.*]]) {MANP = 1 : ui1} : (tensor<8x!FHE.eint<2>>, tensor<8x4xi64>) -> tensor<8x!FHE.eint<3>>
  %res = "FHELinalg.apply_multi_lookup_table"(%0, %luts) : (tensor<8x!FHE.eint<2>>, tensor<8x4xi64>) -> tensor<8x!FHE.eint<3>>
  return %res : tensor<8x!FHE.eint<3>>
}

// -----

/////////////////////////////////////////////////
// FHELinalg.dot_eint_int
/////////////////////////////////////////////////

func @single_cst_dot(%t: tensor<4x!FHE.eint<2>>) -> !FHE.eint<2>
{
  %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi3>
  // sqrt(1^2*1 + 2^2*1 + 3^2*1 + 4^2*1) = 5.477225575
  // CHECK: %[[V0:.*]] = "FHELinalg.dot_eint_int"(%[[T:.*]], %[[CST:.*]]) {MANP = 6 : ui{{[[0-9]+}}} : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> !FHE.eint<2>
  %0 = "FHELinalg.dot_eint_int"(%t, %cst) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> !FHE.eint<2>
  return %0 : !FHE.eint<2>
}

// -----

func @single_dyn_dot(%t: tensor<4x!FHE.eint<2>>, %dyn: tensor<4xi3>) -> !FHE.eint<2>
{
  // sqrt(1*(2^3-1)^2*4) = 14
  // CHECK: %[[V0:.*]] = "FHELinalg.dot_eint_int"([[T:.*]], %[[DYN:.*]]) {MANP = 14 : ui{{[[0-9]+}}} : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> !FHE.eint<2>
  %0 = "FHELinalg.dot_eint_int"(%t, %dyn) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> !FHE.eint<2>

  return %0 : !FHE.eint<2>
}

// -----

func @single_cst_dot_after_op(%t: tensor<4x!FHE.eint<2>>, %i: tensor<4xi3>) -> !FHE.eint<2>
{
  // sqrt((2^3)^2*1) = sqrt(64) = 8
  // CHECK: %[[V0:.*]] = "FHELinalg.mul_eint_int"([[T:.*]], %[[I:.*]]) {MANP = 8 : ui{{[0-9]+}}}
  %0 = "FHELinalg.mul_eint_int"(%t, %i) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> tensor<4x!FHE.eint<2>>

  %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi3>
  // sqrt(1^2*64 + 2^2*64 + 3^2*64 + 4^2*64) = sqrt(1920) = 43.8178046
  // CHECK: %[[V1:.*]] = "FHELinalg.dot_eint_int"(%[[V0]], %[[CST:.*]]) {MANP = 44 : ui{{[[0-9]+}}}
  %1 = "FHELinalg.dot_eint_int"(%0, %cst) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> !FHE.eint<2>

  return %1 : !FHE.eint<2>
}

// -----

func @single_dyn_dot_after_op(%t: tensor<4x!FHE.eint<2>>, %i: tensor<4xi3>) -> !FHE.eint<2>
{
  // sqrt((2^3)^2*1) = sqrt(64) = 8
  // CHECK: %[[V0:.*]] = "FHELinalg.mul_eint_int"([[T:.*]], %[[I:.*]]) {MANP = 8 : ui{{[0-9]+}}}
  %0 = "FHELinalg.mul_eint_int"(%t, %i) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> tensor<4x!FHE.eint<2>>

  // sqrt(4*(2^3-1)^2*64) = sqrt(12544) = 112
  // CHECK: %[[V1:.*]] = "FHELinalg.dot_eint_int"(%[[V0]], %[[I]]) {MANP = 112 : ui{{[[0-9]+}}}
  %1 = "FHELinalg.dot_eint_int"(%0, %i) : (tensor<4x!FHE.eint<2>>, tensor<4xi3>) -> !FHE.eint<2>

  return %1 : !FHE.eint<2>
}

// -----

/////////////////////////////////////////////////
// FHELinalg.matmul_ent_int
/////////////////////////////////////////////////

func @matmul_eint_int_dyn_p_1(%arg0: tensor<3x1x!FHE.eint<2>>, %arg1: tensor<1x2xi3>) -> tensor<3x2x!FHE.eint<2>> {
  // p = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, i3) = 1 * (2^3)^2 = 64
  // manp(add_eint(mul, acc)) = 64 + 1 = 65
  // ceil(sqrt(65)) = 9
  // CHECK: %[[V1:.*]] = "FHELinalg.matmul_eint_int"(%[[A0:.*]], %[[A1:.*]]) {MANP = 9 : ui{{[0-9]+}}}
  %1 = "FHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<3x1x!FHE.eint<2>>, tensor<1x2xi3>) -> tensor<3x2x!FHE.eint<2>>
  return %1 : tensor<3x2x!FHE.eint<2>>
}

// -----

func @matmul_eint_int_dyn_p_2(%arg0: tensor<3x2x!FHE.eint<2>>, %arg1: tensor<2x2xi3>) -> tensor<3x2x!FHE.eint<2>> {
  // p = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, i3) = 1 * (2^3)^2 = 64
  // manp(add_eint(mul, acc)) = 64 + 1 = 65
  // p = 1
  // manp(mul_eint_int(eint<2>, i3) = 1 * (2^3)^2 = 64
  // manp(add_eint(mul, acc)) = 64 + 65 = 129
  // ceil(sqrt(129)) = 12
  // CHECK: %[[V1:.*]] = "FHELinalg.matmul_eint_int"(%[[A0:.*]], %[[A1:.*]]) {MANP = 12 : ui{{[0-9]+}}}
  %1 = "FHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<3x2x!FHE.eint<2>>, tensor<2x2xi3>) -> tensor<3x2x!FHE.eint<2>>
  return %1 : tensor<3x2x!FHE.eint<2>>
}

// -----

func @matmul_eint_int_cst_p_1(%arg0: tensor<3x1x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<2>> {
  %0 = arith.constant dense<[[3, 1]]> : tensor<1x2xi3>
  // c(m,n) = a(m,p) * b(p,n) the max cst is used for n = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, 3) = 1 * 3^2 = 9
  // manp(add_eint(mul, acc)) = 9 + 1 = 10
  // ceil(sqrt(10)) = 4
  // CHECK: %[[V1:.*]] = "FHELinalg.matmul_eint_int"(%[[A0:.*]], %[[A1:.*]]) {MANP = 4 : ui{{[0-9]+}}}
  %1 = "FHELinalg.matmul_eint_int"(%arg0, %0): (tensor<3x1x!FHE.eint<2>>, tensor<1x2xi3>) -> tensor<3x2x!FHE.eint<2>>
  return %1 : tensor<3x2x!FHE.eint<2>>
}

// -----

func @matmul_eint_int_cst_p_2_n_0(%arg0: tensor<3x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<2>> {
  %0 = arith.constant dense<[[4, 1],[3, 1]]> : tensor<2x2xi3>
  // c(m,n) = a(m,p) * b(p,n) the max csts [4,3] are used for n = 0
  // p = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, 3) = 1 * 3^2 = 9
  // manp(add_eint(mul, acc)) = 9 + 1 = 10
  // p = 1
  // mul = manp(mul_eint_int(eint<2>, 4) = 1 * 4^2 = 17
  // manp(add_eint(mul, acc)) = 17 + 9 = 26
  // ceil(sqrt(26)) = 6
  // CHECK: %[[V1:.*]] = "FHELinalg.matmul_eint_int"(%[[A0:.*]], %[[A1:.*]]) {MANP = 6 : ui{{[0-9]+}}}
  %1 = "FHELinalg.matmul_eint_int"(%arg0, %0): (tensor<3x2x!FHE.eint<2>>, tensor<2x2xi3>) -> tensor<3x2x!FHE.eint<2>>
  return %1 : tensor<3x2x!FHE.eint<2>>
}

// -----

func @matmul_eint_int_cst_p_2_n_1(%arg0: tensor<3x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<2>> {
  %0 = arith.constant dense<[[1, 4],[3, 1]]> : tensor<2x2xi3>
  // c(m,n) = a(m,p) * b(p,n) the max csts [4,1] are used for n = 1
  // p = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, 4) = 1 * 4^2 = 16
  // manp(add_eint(mul, acc)) = 16 + 1 = 17
  // p = 1
  // mul = manp(mul_eint_int(eint<2>, 1) = 1 * 1^2 = 1
  // manp(add_eint(mul, acc)) = 1 + 17 = 18
  // ceil(sqrt(18)) = 5
  // CHECK: %[[V1:.*]] = "FHELinalg.matmul_eint_int"(%[[A0:.*]], %[[A1:.*]]) {MANP = 5 : ui{{[0-9]+}}}
  %1 = "FHELinalg.matmul_eint_int"(%arg0, %0): (tensor<3x2x!FHE.eint<2>>, tensor<2x2xi3>) -> tensor<3x2x!FHE.eint<2>>
  return %1 : tensor<3x2x!FHE.eint<2>>
}

/////////////////////////////////////////////////
// FHELinalg.matmul_int_eint
/////////////////////////////////////////////////

// -----

func @matmul_int_eint_dyn_p_1(%arg0: tensor<3x1xi3>, %arg1: tensor<1x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<2>> {
  // p = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, i3) = 1 * (2^3)^2 = 64
  // manp(add_eint(mul, acc)) = 64 + 1 = 65
  // ceil(sqrt(65)) = 9
  // CHECK: %[[V1:.*]] = "FHELinalg.matmul_int_eint"(%[[A0:.*]], %[[A1:.*]]) {MANP = 9 : ui{{[0-9]+}}}
  %1 = "FHELinalg.matmul_int_eint"(%arg0, %arg1): (tensor<3x1xi3>, tensor<1x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<2>>
  return %1 : tensor<3x2x!FHE.eint<2>>
}

// -----

func @matmul_int_eint_dyn_p_2(%arg0: tensor<3x2xi3>, %arg1: tensor<2x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<2>> {
  // p = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, i3) = 1 * (2^3)^2 = 64
  // manp(add_eint(mul, acc)) = 64 + 1 = 65
  // p = 1
  // manp(mul_eint_int(eint<2>, i3) = 1 * (2^3)^2 = 64
  // manp(add_eint(mul, acc)) = 64 + 65 = 129
  // ceil(sqrt(129)) = 12
  // CHECK: %[[V1:.*]] = "FHELinalg.matmul_int_eint"(%[[A0:.*]], %[[A1:.*]]) {MANP = 12 : ui{{[0-9]+}}}
  %1 = "FHELinalg.matmul_int_eint"(%arg0, %arg1): (tensor<3x2xi3>, tensor<2x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<2>>
  return %1 : tensor<3x2x!FHE.eint<2>>
}

// -----

func @matmul_int_eint_cst_p_1(%arg0: tensor<1x3x!FHE.eint<2>>) -> tensor<2x3x!FHE.eint<2>> {
  %0 = arith.constant dense<[[3], [1]]> : tensor<2x1xi3>
  // c(m,n) = a(m,p) * b(p,n) the max cst is used for m = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, 3) = 1 * 3^2 = 9
  // manp(add_eint(mul, acc)) = 9 + 1 = 10
  // ceil(sqrt(10)) = 4
  // CHECK: %[[V1:.*]] = "FHELinalg.matmul_int_eint"(%[[A0:.*]], %[[A1:.*]]) {MANP = 4 : ui{{[0-9]+}}}
  %1 = "FHELinalg.matmul_int_eint"(%0, %arg0): (tensor<2x1xi3>, tensor<1x3x!FHE.eint<2>>) -> tensor<2x3x!FHE.eint<2>>
  return %1 : tensor<2x3x!FHE.eint<2>>
}

// -----

func @matmul_int_eint_cst_p_2_n_0(%arg0: tensor<2x3x!FHE.eint<2>>) -> tensor<2x3x!FHE.eint<2>> {
  %0 = arith.constant dense<[[3, 4],[1, 1]]> : tensor<2x2xi3>
  // c(m,n) = a(m,p) * b(p,n) the max csts [4,3] are used for m = 0
  // p = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, 3) = 1 * 3^2 = 9
  // manp(add_eint(mul, acc)) = 9 + 1 = 10
  // p = 1
  // mul = manp(mul_eint_int(eint<2>, 4) = 1 * 4^2 = 17
  // manp(add_eint(mul, acc)) = 17 + 9 = 26
  // ceil(sqrt(26)) = 6
  // CHECK: %[[V1:.*]] = "FHELinalg.matmul_int_eint"(%[[A0:.*]], %[[A1:.*]]) {MANP = 6 : ui{{[0-9]+}}}
  %1 = "FHELinalg.matmul_int_eint"(%0, %arg0): (tensor<2x2xi3>, tensor<2x3x!FHE.eint<2>>) -> tensor<2x3x!FHE.eint<2>>
  return %1 : tensor<2x3x!FHE.eint<2>>
}

// -----

func @matmul_int_eint_cst_p_2_n_1(%arg0: tensor<2x3x!FHE.eint<2>>) -> tensor<2x3x!FHE.eint<2>> {
  %0 = arith.constant dense<[[4, 1],[3, 1]]> : tensor<2x2xi3>
  // c(m,n) = a(m,p) * b(p,n) the max csts [4,1] are used for m = 1
  // p = 0
  // acc = manp(0) = 1
  // mul = manp(mul_eint_int(eint<2>, 4) = 1 * 4^2 = 16
  // manp(add_eint(mul, acc)) = 16 + 1 = 17
  // p = 1
  // mul = manp(mul_eint_int(eint<2>, 1) = 1 * 1^2 = 1
  // manp(add_eint(mul, acc)) = 1 + 17 = 18
  // ceil(sqrt(18)) = 5
  // CHECK: %[[V1:.*]] = "FHELinalg.matmul_int_eint"(%[[A0:.*]], %[[A1:.*]]) {MANP = 5 : ui{{[0-9]+}}}
  %1 = "FHELinalg.matmul_int_eint"(%0, %arg0): (tensor<2x2xi3>, tensor<2x3x!FHE.eint<2>>) -> tensor<2x3x!FHE.eint<2>>
  return %1 : tensor<2x3x!FHE.eint<2>>
}

// -----

func @zero() -> tensor<8x!FHE.eint<2>>
{
  // CHECK: %[[ret:.*]] = "FHELinalg.zero"() {MANP = 1 : ui{{[0-9]+}}} : () -> tensor<8x!FHE.eint<2>>
  %0 = "FHELinalg.zero"() : () -> tensor<8x!FHE.eint<2>>

  return %0 : tensor<8x!FHE.eint<2>>
}

// -----

func @sum() -> !FHE.eint<7> {
  %0 = "FHELinalg.zero"() : () -> tensor<5x3x4x2x!FHE.eint<7>>

  // CHECK: MANP = 11 : ui{{[0-9]+}}
  %1 = "FHELinalg.sum"(%0) : (tensor<5x3x4x2x!FHE.eint<7>>) -> !FHE.eint<7>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %2 = "FHELinalg.sum"(%0) { axes = [0] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<3x4x2x!FHE.eint<7>>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %3 = "FHELinalg.sum"(%0) { axes = [1] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x4x2x!FHE.eint<7>>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %4 = "FHELinalg.sum"(%0) { axes = [2] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x3x2x!FHE.eint<7>>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %5 = "FHELinalg.sum"(%0) { axes = [3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x3x4x!FHE.eint<7>>

  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %6 = "FHELinalg.sum"(%0) { axes = [0, 1] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<4x2x!FHE.eint<7>>

  // CHECK: MANP = 5 : ui{{[0-9]+}}
  %7 = "FHELinalg.sum"(%0) { axes = [0, 2] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>>

  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %8 = "FHELinalg.sum"(%0) { axes = [0, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<3x4x!FHE.eint<7>>

  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %9 = "FHELinalg.sum"(%0) { axes = [1, 2] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x2x!FHE.eint<7>>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %10 = "FHELinalg.sum"(%0) { axes = [1, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x4x!FHE.eint<7>>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %11 = "FHELinalg.sum"(%0) { axes = [2, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x3x!FHE.eint<7>>

  // CHECK: MANP = 8 : ui{{[0-9]+}}
  %12 = "FHELinalg.sum"(%0) { axes = [0, 1, 2] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<2x!FHE.eint<7>>

  // CHECK: MANP = 6 : ui{{[0-9]+}}
  %13 = "FHELinalg.sum"(%0) { axes = [0, 1, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>>

  // CHECK: MANP = 7 : ui{{[0-9]+}}
  %14 = "FHELinalg.sum"(%0) { axes = [0, 2, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>>

  // CHECK: MANP = 5 : ui{{[0-9]+}}
  %15 = "FHELinalg.sum"(%0) { axes = [1, 2, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x!FHE.eint<7>>

  // CHECK: MANP = 11 : ui{{[0-9]+}}
  %16 = "FHELinalg.sum"(%0) { axes = [0, 1, 2, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> !FHE.eint<7>

  // CHECK: MANP = 11 : ui{{[0-9]+}}
  %17 = "FHELinalg.sum"(%0) { keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x1x!FHE.eint<7>>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %18 = "FHELinalg.sum"(%0) { axes = [0], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x3x4x2x!FHE.eint<7>>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %19 = "FHELinalg.sum"(%0) { axes = [1], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x1x4x2x!FHE.eint<7>>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %20 = "FHELinalg.sum"(%0) { axes = [2], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x3x1x2x!FHE.eint<7>>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %21 = "FHELinalg.sum"(%0) { axes = [3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x3x4x1x!FHE.eint<7>>

  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %22 = "FHELinalg.sum"(%0) { axes = [0, 1], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x1x4x2x!FHE.eint<7>>

  // CHECK: MANP = 5 : ui{{[0-9]+}}
  %23 = "FHELinalg.sum"(%0) { axes = [0, 2], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x3x1x2x!FHE.eint<7>>

  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %24 = "FHELinalg.sum"(%0) { axes = [0, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x3x4x1x!FHE.eint<7>>

  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %25 = "FHELinalg.sum"(%0) { axes = [1, 2], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x1x1x2x!FHE.eint<7>>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %26 = "FHELinalg.sum"(%0) { axes = [1, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x1x4x1x!FHE.eint<7>>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %27 = "FHELinalg.sum"(%0) { axes = [2, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x3x1x1x!FHE.eint<7>>

  // CHECK: MANP = 8 : ui{{[0-9]+}}
  %28 = "FHELinalg.sum"(%0) { axes = [0, 1, 2], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x2x!FHE.eint<7>>

  // CHECK: MANP = 6 : ui{{[0-9]+}}
  %29 = "FHELinalg.sum"(%0) { axes = [0, 1, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x1x4x1x!FHE.eint<7>>

  // CHECK: MANP = 7 : ui{{[0-9]+}}
  %30 = "FHELinalg.sum"(%0) { axes = [0, 2, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x3x1x1x!FHE.eint<7>>

  // CHECK: MANP = 5 : ui{{[0-9]+}}
  %31 = "FHELinalg.sum"(%0) { axes = [1, 2, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x1x1x1x!FHE.eint<7>>

  // CHECK: MANP = 11 : ui{{[0-9]+}}
  %32 = "FHELinalg.sum"(%0) { axes = [0, 1, 2, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x1x!FHE.eint<7>>

  // ===============================

  %35 = "FHELinalg.zero"() : () -> tensor<2x0x3x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %36 = "FHELinalg.sum"(%35) : (tensor<2x0x3x!FHE.eint<7>>) -> !FHE.eint<7>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %37 = "FHELinalg.sum"(%35) { axes = [0] } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<0x3x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %38 = "FHELinalg.sum"(%35) { axes = [1] } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<2x3x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %39 = "FHELinalg.sum"(%35) { axes = [2] } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<2x0x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %40 = "FHELinalg.sum"(%35) { axes = [0, 1] } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %41 = "FHELinalg.sum"(%35) { axes = [0, 2] } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<0x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %42 = "FHELinalg.sum"(%35) { axes = [1, 2] } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<2x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %43 = "FHELinalg.sum"(%35) { axes = [0, 1 ,2] } : (tensor<2x0x3x!FHE.eint<7>>) -> !FHE.eint<7>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %44 = "FHELinalg.sum"(%35) { keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %45 = "FHELinalg.sum"(%35) { axes = [0], keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<1x0x3x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %46 = "FHELinalg.sum"(%35) { axes = [1], keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<2x1x3x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %47 = "FHELinalg.sum"(%35) { axes = [2], keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<2x0x1x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %48 = "FHELinalg.sum"(%35) { axes = [0, 1], keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<1x1x3x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %49 = "FHELinalg.sum"(%35) { axes = [0, 2], keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<1x0x1x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %50 = "FHELinalg.sum"(%35) { axes = [1, 2], keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<2x1x1x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %51 = "FHELinalg.sum"(%35) { axes = [0, 1 ,2], keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>>

  return %1 : !FHE.eint<7>
}

// -----

func @concat() -> tensor<3x!FHE.eint<7>> {
  %0 = "FHELinalg.zero"() : () -> tensor<4x!FHE.eint<7>>
  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %1 = "FHELinalg.sum"(%0) { keep_dims = true } : (tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>>

  %2 = "FHELinalg.zero"() : () -> tensor<5x!FHE.eint<7>>
  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %3 = "FHELinalg.sum"(%2) { keep_dims = true } : (tensor<5x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>>

  %4 = "FHELinalg.zero"() : () -> tensor<10x!FHE.eint<7>>
  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %5 = "FHELinalg.sum"(%4) { keep_dims = true } : (tensor<10x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %6 = "FHELinalg.concat"(%1, %3) : (tensor<1x!FHE.eint<7>>, tensor<1x!FHE.eint<7>>) ->  tensor<2x!FHE.eint<7>>
  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %7 = "FHELinalg.concat"(%1, %5) : (tensor<1x!FHE.eint<7>>, tensor<1x!FHE.eint<7>>) ->  tensor<2x!FHE.eint<7>>
  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %8 = "FHELinalg.concat"(%3, %5) : (tensor<1x!FHE.eint<7>>, tensor<1x!FHE.eint<7>>) ->  tensor<2x!FHE.eint<7>>
  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %9 = "FHELinalg.concat"(%1, %3, %5) : (tensor<1x!FHE.eint<7>>, tensor<1x!FHE.eint<7>>, tensor<1x!FHE.eint<7>>) ->  tensor<3x!FHE.eint<7>>

  return %9 : tensor<3x!FHE.eint<7>>
}
