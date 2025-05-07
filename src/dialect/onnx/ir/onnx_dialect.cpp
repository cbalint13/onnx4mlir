
#include "onnx2mlir/dialect/onnx/OnnxDialect.hpp"

#include "onnx2mlir/dialect/onnx/OnnxDialect.cpp.inc"

namespace onnx2mlir::dialect::onnx {

void OnnxDialect::initialize() {
  registerTypes();
  registerOps();
}

} // namespace onnx2mlir::dialect::onnx
