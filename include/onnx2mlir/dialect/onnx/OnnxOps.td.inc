/********************************************************
 *   ONNX version [1.19.0]                              *
 *   Generated at [2025-05-06 22:45:53.874995]          *
 ********************************************************/

/// Constant
def Onnx_ConstantOp : Onnx_Op<"Constant", []> {
  let summary = "ONNX Constant operation";
  let description = [{
      This operator produces a constant tensor.
  }];

  let arguments = (ins OptionalAttr<AnyAttr>: $value);
  //let results = (outs AnyTypeOf<[TensorOf<[UI8]>]>:$output);
  //let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>, TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>]>: $output);
  let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[StringType]>]>:$output);
}

/// Add
def Onnx_AddOp : Onnx_Op<"Add", []> {
  let summary = "ONNX Add operation";
  let description = [{
      Performs element-wise binary addition (with Numpy-style broadcasting support).
  }];

  //let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>: $A, AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>: $B);
  //let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>: $C);
  let arguments = (ins AnyTensor : $A, AnyTensor : $B);
  let results = (outs AnyTensor : $result);
}
