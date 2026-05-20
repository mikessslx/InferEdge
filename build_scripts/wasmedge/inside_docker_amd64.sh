export PYTORCH_VERSION="1.7.1"
export PYTORCH_ABI="libtorch-cxx11-abi"

# Set necessary paths
export LD_LIBRARY_PATH=$(pwd)/libtorch/lib:${LD_LIBRARY_PATH}
export Torch_DIR=$(pwd)/libtorch

# Build WasmEdge with the PyTorch plugin
cd /root/wasmedge
cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release -DWASMEDGE_PLUGIN_WASI_NN_BACKEND="PyTorch"
cmake --build build
cmake --install build --prefix ~/wasmedge-install