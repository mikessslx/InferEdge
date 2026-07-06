#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_VERSION="1.7.1"
export PYTHON_VERSION="cp39"

# Set necessary paths
export LD_LIBRARY_PATH=/root/libtorch/lib:${LD_LIBRARY_PATH:-}
export Torch_DIR=/root/libtorch/share/cmake/Torch
export CMAKE_PREFIX_PATH=/root/libtorch:${CMAKE_PREFIX_PATH:-}

# Build WasmEdge with the PyTorch plugin
cd /root/wasmedge
rm -rf build
cmake -GNinja -Bbuild \
    -DCMAKE_BUILD_TYPE=Debug \
    -DTorch_DIR="$Torch_DIR" \
    -DWASMEDGE_FORCE_DISABLE_LTO=ON \
    -DWASMEDGE_USE_LLVM=ON \
    -DWASMEDGE_PLUGIN_WASI_NN_BACKEND="PyTorch" \
    -DCMAKE_C_FLAGS_DEBUG="-O0 -g0" \
    -DCMAKE_CXX_FLAGS_DEBUG="-O0 -g0" \
    -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations"
cmake --build build -j1
cmake --install build --prefix ~/wasmedge-install
