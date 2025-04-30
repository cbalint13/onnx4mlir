/******************************************************************************
 *
 * Copyright (c) 2021,2025
 *
 *     Cristian Balint <cristian dot balint at gmail dot com>
 *
 * ONNX2MLIR (ONNX dialect mappings for composable optimizations)
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 *****************************************************************************/

/*!
 * \file OnnxOps.td
 *
 */

#ifndef ONNX2MLIR_FRONTEND_ONNXOPS_TD_
#define ONNX2MLIR_FRONTEND_ONNXOPS_TD_

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "onnx2mlir/dialect/onnx/OnnxDialect.td"

def InferShapedType : DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                                                ["inferReturnTypeComponents"]>;
/// Onnx operator
class Onnx_Op<string mnemonic, list<Trait> traits>
    : Op<Onnx_Dialect, mnemonic, traits #[NoMemoryEffect, InferShapedType]>;

/// Constant
def Onnx_ConstantOp : Onnx_Op<"Constant", [ConstantLike]> {
  let summary = "ONNX Constant operation";
  let description = [{
    This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
    or value_* must be specified.
  }];

  let arguments = (ins ElementsAttr : $value);
  let results = (outs AnyStaticShapeTensor : $result);
}
// nn.bias_add
def Onnx_AddOp : Onnx_Op<"Add", [SameOperandsElementType]> {
  let summary = "ONNX Add operation";
  let description = [{
    Performs element-wise binary addition (with Numpy-style broadcasting support).
    This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
  }];

  let arguments = (ins AnyTensor : $A, AnyTensor : $B);
  let results = (outs AnyTensor : $result);
}

/* OLD TVM RELAY DIALECT
/// Unary-elementwise operators
class Onnx_UnaryElementwiseOp<string mnemonic, list<Trait> traits,
                               Type ArgType, Type ResultType = ArgType>
    : Onnx_Op<mnemonic, traits #[Elementwise]> {
  let arguments = (ins ArgType : $data);
  let results = (outs ResultType : $result);
}

// nn.relu
def Onnx_ReLUOp : Onnx_UnaryElementwiseOp<"nn.relu", [], AnyTensor> {
  let summary = "ReLU operator.";
  let description = [{Rectified linear unit.}];
}

// nn.dense
def Onnx_DenseOp : Onnx_Op<"nn.dense", [SameOperandsElementType]> {
  let summary = "Dense operator.";
  let description = [{Applies a linear transformation Y = X * W ^ T.}];

  let arguments = (ins AnyTensor : $data, AnyTensor : $weight);
  let results = (outs AnyTensor : $result);
}
*/

#endif // ONNX2MLIR_FRONTEND_ONNXOPS_TD_
