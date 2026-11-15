cmake_minimum_required(VERSION 3.20)
project(triton-perlane-retire LANGUAGES CXX)

# Out-of-tree Triton plugin pass. Two ways to say which Triton to build against.
#
# 1. An installed wheel (Triton >= 3.9, built with TRITON_EXT_ENABLED; this is
#    what `pip install triton` ships) plus the LLVM/MLIR distribution it was
#    built with. Headers and libtriton.so come from the wheel; LLVM is only
#    used for headers and mlir-tblgen, its symbols are re-exported by libtriton:
#      -DTRITON_WHEEL_DIR=<site-packages>/triton
#      -DTRITON_LLVM_DIR=<llvm dir>   (python ci/download_llvm.py in triton-ext
#                                      fetches the artifact Triton pins)
#
# 2. A Triton source/build tree (Triton main, in-source builds have both in one
#    place):
#      -DTRITON_SRC_DIR=/path/to/triton
#      -DTRITON_BUILD_DIR=/path/to/triton-build   (defaults to TRITON_SRC_DIR)
#      -DTRITON_LLVM_DIR=/path/to/llvm-dist       (the ~/.triton/llvm/... dir)
set(TRITON_WHEEL_DIR "" CACHE PATH "Installed triton package directory (wheel mode)")
set(TRITON_SRC_DIR "" CACHE PATH "Triton source tree (source mode)")
set(TRITON_BUILD_DIR "${TRITON_SRC_DIR}" CACHE PATH "Triton build tree (source mode)")
set(TRITON_LLVM_DIR "" CACHE PATH "LLVM/MLIR distribution used to build Triton")

if(NOT TRITON_LLVM_DIR OR (NOT TRITON_WHEEL_DIR AND NOT TRITON_SRC_DIR))
  message(FATAL_ERROR
    "Set -DTRITON_LLVM_DIR and either -DTRITON_WHEEL_DIR (installed wheel) "
    "or -DTRITON_SRC_DIR (source tree)")
endif()

# Match Triton and LLVM (triton-ext builds extensions as C++20).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_PREFIX_PATH "${TRITON_LLVM_DIR}")
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# libtriton is an asserts build (ABI breaking checks on); a plugin compiled
# against headers with the other setting fails at dlopen with "undefined
# symbol" on ilist iterators. Check it now instead of at load time.
file(READ "${TRITON_LLVM_DIR}/include/llvm/Config/abi-breaking.h" _abi)
string(REGEX MATCH "#define[ \t]+LLVM_ENABLE_ABI_BREAKING_CHECKS[ \t]+([0-9]+)"
       _abi_match "${_abi}")
if(NOT _abi_match OR NOT CMAKE_MATCH_1 EQUAL 1)
  message(FATAL_ERROR
    "LLVM_ENABLE_ABI_BREAKING_CHECKS must be 1 in ${TRITON_LLVM_DIR} "
    "(found '${CMAKE_MATCH_1}'); use the LLVM artifact Triton pins")
endif()

if(TRITON_WHEEL_DIR)
  get_filename_component(TRITON_WHEEL_DIR "${TRITON_WHEEL_DIR}" ABSOLUTE)
  set(_triton_includes "${TRITON_WHEEL_DIR}/include")
  set(TRITON_LIB "${TRITON_WHEEL_DIR}/_C/libtriton.so"
      CACHE FILEPATH "Path to libtriton.so (must be built with TRITON_EXT_ENABLED)")
  if(NOT EXISTS "${_triton_includes}/triton/Tools/PluginUtils.h")
    message(FATAL_ERROR
      "${TRITON_WHEEL_DIR} has no plugin headers; the wheel must be a "
      "Triton >= 3.9 build with TRITON_EXT_ENABLED")
  endif()
else()
  set(_triton_includes
    ${TRITON_SRC_DIR}/include
    ${TRITON_SRC_DIR}            # for nvidia/... third_party include forms
    ${TRITON_BUILD_DIR}/include  # generated headers (triton/Version.h)
  )
  set(TRITON_LIB "${TRITON_BUILD_DIR}/python/triton/_C/libtriton.so"
      CACHE FILEPATH "Path to libtriton.so (must be built with TRITON_EXT_ENABLED)")
endif()

include_directories(
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${_triton_includes}
  ${CMAKE_CURRENT_BINARY_DIR}  # local Passes.h.inc
)
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name PerLaneRetire)
add_public_tablegen_target(PerLaneRetireIncGen)

add_library(PerLaneLoopRetirement SHARED PerLaneLoopRetirement.cpp)
add_dependencies(PerLaneLoopRetirement PerLaneRetireIncGen)
target_compile_options(PerLaneLoopRetirement PRIVATE
  -fvisibility=hidden -fno-exceptions -fno-rtti -Wno-deprecated-declarations)

# Link against libtriton.so so the plugin carries a DT_NEEDED edge and its
# MLIR/LLVM + plugin-glue symbols resolve at dlopen (the same library is
# already loaded in the host process). --no-as-needed keeps the edge although
# nothing is referenced at link time.
target_link_libraries(PerLaneLoopRetirement PRIVATE
  "$<$<PLATFORM_ID:Linux>:-Wl,--no-as-needed>"
  "${TRITON_LIB}"
  "$<$<PLATFORM_ID:Linux>:-Wl,--as-needed>")
