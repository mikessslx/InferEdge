# InferEdge: Characterizing Edge Inference Performance
Official implementation of InferEdge: a software suite automating performance characterization of edge inference.

## About the Research
Data processing on relatively hardware-limited resources closer to the end-user is the key premise of edge computing. Machine learning (ML) inference is a key workload that supports this processing, facilitated by deployment mechanisms such as Docker containers or WebAssembly. However, the performance characteristics of the deployment mechanisms for edge inference workloads across different target hardware platforms remain unknown. Therefore, we develop InferEdge, a software suite that automates edge inference performance characterization for a range of edge inference models on heterogeneous processors. 

Experimental results obtained using this suite provide valuable insights. Interpreted WebAssembly executes the slowest due to long input preparation times and is sensitive to the input used. Meanwhile, ahead-of-time compiled WebAssembly closely matches native execution speeds for smaller inference workloads, outperforming Docker due to lower cold start times. However, this comes at the expense of increased memory consumption. Moreover, on some platforms, WebAssembly’s advantage in execution speed becomes increasingly slim, and its disadvantage in memory usage increasingly large, as workload size increases. Consequently, which deployment mechanism performs better depends on the workload and target platform, validating the need for a suite that can automate performance characterization and offer insights.

Further details regarding the design, implementation, and experimental results of InferEdge will be available in our paper, which has been accepted to the 18th IEEE/ACM International Conference on Utility and Cloud Computing (UCC2025) and will be included in its proceedings.

## Code Structure
A quick summary of each subdirectory's contents is as follows:

* build_scripts: shell scripts for building WasmEdge
* cadvisor: an editable cAdvisor config file; will also contain the cAdvisor binary when built by the suite
* data_scripts: Python scripts for collecting data through experiments and analyzing the results
* docker: a Docker installation script; will also contain the Docker image for image classification when built by the suite
* Dockerfiles: Dockerfiles for building various Docker images involved in the suite
* host_scripts: scripts to be executed on the host machine, including model generation scripts
* inputs: inputs for ML inference that will be transferred to the target device
* libtorch: will contain an appropriate LibTorch library when downloaded by the suite
* models: will contain models for ML inference that will be transferred to the target device, including those that the suite will generate
* native: will contain native binaries for ML inference when built by the suite
* prometheus: an editable Prometheus config file; will also contain the Prometheus binary when downloaded by the suite
* python: requirements.txt files for setting up virtual environments on the host machine and target device
* results: will contain the experimental results retrieved from the target device, alongside the results of analyzing them
* rust: ML inference source code
* target_scripts: scripts to be executed on the target machine, including the setup script
* wasm: will contain WebAssembly binaries for ML inference when built by the suite
* wasmedge: will contain the WasmEdge binary and related files when built by the suite

A more detailed description of each subdirectory's contents can be found in the subdirectory's README file. 

## Usage
To run the suite, grant execute permissions to the script as follows:

```chmod u+x suite.sh```

Then run the following command on the shell:

```./suite.sh```

Note that elements of this script will require sudo permissions, for example when installing required packages. Additionally, note that an explanation of the suite's functionality can also be found upon launching the suite and selecting the explanation option.
