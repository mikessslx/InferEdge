Includes image classification source code in Rust that will be compiled to native and WebAssembly binaries
for all the deployment mechanisms. The code for WebAssembly is taken directly from
WasmEdge's example for image classification using WASI-NN, available at: 
https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/pytorch-mobilenet-image. The code
for native is a modification of it using the tch crate for LibTorch which attempts to mirror the original
as closely as possible.

You may modify the source code if you would like to do performance characterization on a different image
classification workflow, though take care to ensure both versions are as close as possible. The current
workflow loads in an image, preprocesses it, performs inference, and lists out the top 5 most probable
image classes.