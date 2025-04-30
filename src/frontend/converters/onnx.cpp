
#include <onnx/common/version.h>
#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>
#include <onnx/version_converter/convert.h>

#include <fstream>

#include "onnx2mlir/frontend/onnx.hpp"

namespace onnx2mlir::frontend {

void ONNXConverter::convert(mlir::ModuleOp &module) {}

} // end namespace onnx2mlir::frontend
