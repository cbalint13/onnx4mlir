
#include "onnx2mlir/dialect/onnx/OnnxDialect.hpp"
#include "onnx2mlir/dialect/onnx/OnnxOps.hpp"
#include "onnx2mlir/dialect/onnx/OnnxTypes.hpp"

#define GET_OP_CLASSES
#include "onnx2mlir/dialect/onnx/Onnx.cpp.inc"

namespace onnx2mlir::dialect::onnx {

void OnnxDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "onnx2mlir/dialect/onnx/Onnx.cpp.inc"
      >();
}

} // namespace onnx2mlir::dialect::onnx
