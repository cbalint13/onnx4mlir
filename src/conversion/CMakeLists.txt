# Onnx tablegen passes gen
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name conversion)
add_public_tablegen_target(MLIRConversionPassesIncGen)

# Onnx docs
add_mlir_doc(Passes ConversionPasses conversion/ -gen-pass-doc)

add_subdirectory(passes)
