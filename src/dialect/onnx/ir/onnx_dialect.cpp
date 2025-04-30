#include "onnx2mlir/dialect/onnx/OnnxDialect.hpp"
#include "onnx2mlir/dialect/onnx/OnnxOps.hpp"

#include "onnx2mlir/dialect/onnx/OnnxOpsDialect.cpp.inc"

namespace onnx2mlir::dialect::onnx {

void OnnxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "onnx2mlir/dialect/onnx/OnnxOps.cpp.inc"
      >();
}

} // namespace relay
