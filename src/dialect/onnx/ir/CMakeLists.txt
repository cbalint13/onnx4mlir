add_mlir_dialect_library(onnx2mlir_onnx_dialect
  OBJECT
    onnx_dialect.cpp
    onnx_ops.cpp
  DEPENDS
    MLIROnnxOpsIncGen
  LINK_LIBS PUBLIC
    MLIR
)
