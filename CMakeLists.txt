cmake_minimum_required(VERSION 3.15)
project(onnx2mlir LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(ONNX REQUIRED 1.17)
if(ONNX_FOUND)
  message(STATUS "Found ONNX version: ${ONNX_VERSION}")
endif()

set(LLVM_CONFIG "llvm-config-21" CACHE STRING "Build with LLVM set to specific llvm-config path")

# use llvm config
if (DEFINED LLVM_CONFIG)
  message(STATUS "Using ${LLVM_CONFIG}")
  separate_arguments(LLVM_CONFIG)
  execute_process(COMMAND ${LLVM_CONFIG} --libdir
    RESULT_VARIABLE __llvm_exit_code
    OUTPUT_VARIABLE __llvm_cmakedir_space
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT "${__llvm_exit_code}" STREQUAL "0")
    message(FATAL_ERROR "Fatal error executing: ${LLVM_CONFIG} --cmakedir")
  else()
    list(APPEND CMAKE_PREFIX_PATH "${__llvm_cmakedir_space}/cmake")
  endif()
else()
  message(FATAL_ERROR "Please use -DLLVM_CONFIG=<path-to-llvm-config> for a valid llvm setup.")
endif()

find_package(MLIR REQUIRED CONFIG)
if(MLIR_FOUND)
  message(STATUS "Found LLVM/MLIR version: ${LLVM_VERSION}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-but-set-variable")

#set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
#set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/src)
#set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
#include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
#link_directories(${LLVM_BUILD_LIBRARY_DIR})
#link_directories(${PROJECT_BINARY_DIR}/src)
add_definitions(${LLVM_DEFINITIONS})


add_subdirectory(include/onnx2mlir/dialect/onnx)
add_subdirectory(src/dialect/onnx/ir)


set(SOURCE_IMPORTER_FRONTENDS
  src/frontend/importers/onnx.cpp
)
set(SOURCE_CONVERTER_FRONTENDS
  src/frontend/converters/onnx.cpp
)
add_library(frontends OBJECT
  ${SOURCE_IMPORTER_FRONTENDS}
  ${SOURCE_CONVERTER_FRONTENDS}
)
target_link_libraries(frontends
  PRIVATE
    onnx2mlir_onnx_dialect
  PUBLIC
    ONNX::onnx
)

add_executable(onnx2mlir src/main.cpp)
target_link_libraries(onnx2mlir
  PRIVATE
    frontends
  PUBLIC
    LLVM
    MLIR
)
