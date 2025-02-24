cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(deploy_onnx_rt)

set(USE_CUDNN OFF)
set(CAFFE2_USE_CUDNN OFF)
set(USE_CUSPARSELT OFF)

## Find OpenCV installation
find_package(OpenCV REQUIRED HINTS ${OPENCV_INSTALL_ROOT}/build/ QUIET)
find_package(Torch REQUIRED HINTS ${LIBTORCH_INSTALL_ROOT}/share/cmake/Torch/ QUIET)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")

## ONNX Runtime Providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(LIBPNG_ROOTDIR "libpng root dir")
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")
set(CMAKE_CXX_STANDARD 17)

if(NOT ONNXRUNTIME_ROOTDIR)
    message(STATUS "You must provide an onnx runtime root directory!")
endif()

# The ORT package has a different include directory structure to a local install via cmake.
# We added the path for the pre-built package above. Add the path for a local install to support either usage.
# TODO: If we want to support additional EPs being loadable from a local install we also need to add EP specific
# directories under /include/onnxruntime/core/providers
include_directories("${ONNXRUNTIME_ROOTDIR}/include"                            # Pre-built package
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"                # Linux local install to /usr/local
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session"   # Windows local install
                    ${OpenCV_INCLUDE_DIRS}                                      # OpenCV
                    ${TORCH_INCLUDE_DIRS})

link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

if(onnxruntime_USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()

# Add Jason's tvm-tools as a dependency. This is used to compile the ONNX networks to TVM.
include(FetchContent)
FetchContent_Declare(
  tvm-tools
  GIT_REPOSITORY git@github.imgtec.org:jason-rogers/tvm_tools.git
  GIT_TAG        support_imgnn
)
FetchContent_MakeAvailable(tvm-tools)

function(compile_onnx_network input_file output_name extra_args)
    add_custom_command(
        OUTPUT ${output_name}.so
        OUTPUT ${output_name}.json
        OUTPUT ${output_name}.params
        COMMAND python3 ${tvm-tools_SOURCE_DIR}/compile.py -i ${input_file} -o ${CMAKE_BINARY_DIR}/${output_name} -f opencl ${extra_args} -b 35.4.1632.23
        COMMENT "Compiling: ${input_file}"
    )
endfunction()

# Compile the ONNX networks
#compile_onnx_network(${CMAKE_SOURCE_DIR}/../iter_140215_epoch_4_step_15999.onnx iter_140215_epoch_4_step_15999 "")

# Define custom targets that depend on the generated files
#add_custom_target(iter_140215_epoch_4_step_15999_compile ALL
#    DEPENDS ${CMAKE_BINARY_DIR}/iter_140215_epoch_4_step_15999.so ${CMAKE_BINARY_DIR}/iter_140215_epoch_4_step_15999.json ${CMAKE_BINARY_DIR}/iter_140215_epoch_4_step_15999.params
#)

## Build & Link
add_executable(deploy_onnx_rt main.cpp)
#add_dependencies(deploy_onnx_rt iter_140215_epoch_4_step_15999_compile)

target_link_directories(deploy_onnx_rt PRIVATE ${TVM_BUILD_DIR})
target_link_libraries(deploy_onnx_rt PRIVATE onnxruntime ${TORCH_LIBRARIES} ${OpenCV_LIBS} opencv_imgcodecs tvm tvm_runtime)


target_include_directories(deploy_onnx_rt PRIVATE
    ${TVM_SOURCE_ROOT}/include
    ${TVM_SOURCE_ROOT}/3rdparty/dlpack/include
    ${TVM_SOURCE_ROOT}/3rdparty/dmlc-core/include)
