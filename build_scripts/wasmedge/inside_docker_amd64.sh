#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_VERSION="1.7.1"
export PYTORCH_ABI="libtorch-cxx11-abi"

# Set necessary paths
export LD_LIBRARY_PATH=/root/libtorch/lib:${LD_LIBRARY_PATH:-}
export Torch_DIR=/root/libtorch

# Build WasmEdge with the PyTorch plugin
cd /root/wasmedge
rm -rf build
cmake -GNinja -Bbuild \
    -DCMAKE_BUILD_TYPE=Release \
    -DWASMEDGE_USE_LLVM=ON \
    -DWASMEDGE_PLUGIN_WASI_NN_BACKEND="PyTorch" \
    -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations"
cmake --build build
cmake --install build --prefix ~/wasmedge-install
