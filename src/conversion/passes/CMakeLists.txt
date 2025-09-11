add_definitions(-DONNX_NAMESPACE=onnx -DONNX_ML=${ONNX_USE_ML})

add_mlir_conversion_library(onnx2mlir_onnx_linalg_pass
  OBJECT
    onnx_to_linalg.cpp
    onnx_to_linalg/constant.cpp
  DEPENDS
    onnx2mlir_onnx_dialect
    MLIRConversionPassIncGen
  LINK_LIBS
    PUBLIC MLIR
)
