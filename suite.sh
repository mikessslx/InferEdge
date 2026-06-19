#!/bin/bash
# This is the main script of the suite, providing functionality to automate the performance characterization
# of edge ML inference.

export PYTORCH_VERSION="1.7.1"
export PYTHON_VERSION="cp39"
export PYTORCH_ABI="libtorch-cxx11-abi"
export PROMETHEUS_VERSION="3.2.1"
export SUITE_NAME="CS4099Suite"
export WASMEDGE_REF="${WASMEDGE_REF:-0.14.1}"
export WASMEDGE_BUILD_MAX_ATTEMPTS="${WASMEDGE_BUILD_MAX_ATTEMPTS:-1}"

function main() {
    prompt_user_for_action
}

function prompt_user_for_target_details_if_not_set() {
    # Prompt the user for the login details of the target machine if they are not set
    if [ -z "$target_address" ] || [ -z "$target_username" ] || [ -z "$target_password" ]; then
        prompt_user_for_target_details
    fi
}

function prompt_user_for_target_details() {
    # Prompt the user for the login details of the target machine
    read -p "Enter the target machine's address: " target_address
    read -p "Enter the username to log into the target machine (this username must have admin permissions): " target_username
    read -s -p "Enter the password for the user with the given username: " target_password
    echo
}

function prompt_user_for_architecture_if_not_set() {
    # Prompt the user for the architecture of the target machine if it is not set
    if [ -z "$arch" ]; then
        prompt_user_for_architecture
    fi
}

function prompt_user_for_architecture() {
    # Prompt the user for the architecture of the target machine
    while true; do
        echo "Which of the following architectures does the target machine use?"
            echo "1. x86_64"
            echo "2. aarch64"
        local arch_input
        read -p "Enter the number identifying the architecture: " arch_input
        case $arch_input in
            1) arch="amd64"; break ;;
            2) arch="arm64"; break ;;
            *) echo "Invalid option." ;;
        esac
    done

    echo "Architecture set to $arch."
    prompt_user_for_mac_if_not_set
}

function prompt_user_for_mac_if_not_set() {
    # Prompt the user to clarify whether the target machine is a Mac if this has not been set
    if [ -z "$is_mac" ]; then
        prompt_user_for_mac
    fi
}

function prompt_user_for_mac() {
    # Prompt the user to clarify whether the target machine is a Mac
    while true; do
        echo "Is the target machine a Mac? Answer 'yes' if you are running a VM on a Mac as well."
            echo "1. Yes"
            echo "2. No"
        local is_mac_input
        read -p "Enter the number identifying the correct option: " is_mac_input
        case $is_mac_input in
            1) is_mac=1; break ;;
            2) is_mac=0; break ;;
            *) echo "Invalid option." ;;
        esac
    done
}

function prompt_user_for_action() {
    # Prompt the user for the action they would like to perform
    while true; do
        echo "What would you like to do?"
            echo "1. Run the entire suite from start to finish"
            echo "2. Select specific steps to run"
            echo "3. Set/change target machine details"
            echo "4. Read an explanation of the suite"
            echo "5. Exit"
        local action
        read -p "Enter the number identifying the action you would like to perform: " action
        case $action in
            1) run_suite ;;
            2) prompt_user_for_specific_steps ;;
            3) prompt_user_for_target_details ;;
            4) explain_suite ;;
            5) exit 0 ;;
            *) echo "Invalid option." ;;
        esac
    done
}

function run_suite() {
    # Run the entire suite
    prompt_user_for_target_details_if_not_set
    acquire_files
    transfer_files
    setup_target_machine
    run_data_collection
    retrieve_data_collection_results
    run_data_analysis
}

function prompt_user_for_specific_steps() {
    # Prompt the user for specific actions they would like to perform
    while true; do
        echo "What would you like to do? Note that you should not run a step unless the previous one has been run in the past."
            echo "1. Acquire files to transfer to target machine"
            echo "2. Transfer files to target machine"
            echo "3. Setup target machine"
            echo "4. Run data collection on target machine"
            echo "5. Retrieve data collection results from target machine"
            echo "6. Run data analysis on host machine"
            echo "7. Back to main menu"
        local action
        read -p "Enter the number identifying the action you would like to perform: " action
        case $action in
            1) acquire_files ;;
            2) transfer_files ;;
            3) setup_target_machine ;;
            4) run_data_collection ;;
            5) retrieve_data_collection_results ;;
            6) run_data_analysis ;;
            7) prompt_user_for_action ;;
            *) echo "Invalid option." ;;
        esac
    done
}

function acquire_files() {
    # Acquire the files needed for experiments to be performed on the target machine
    echo "Acquiring files for experiments..."

    echo "Would you like to delete the architecture-specific files from previous runs?"
    echo "You must do this if you are acquiring files for a target device with an architecture different from the previous target devices used."
        echo "1. Yes"
        echo "2. No"
    while true; do
        local delete_files_input
        read -p "Enter the number identifying your choice: " delete_files_input
        case $delete_files_input in
            1) sudo rm -rf WasmEdge native/torch_image_classification libtorch/bin libtorch/include libtorch/lib libtorch/share \
                cadvisor/cadvisor prometheus/prometheus docker/*.tar wasmedge/bin wasmedge/include wasmedge/lib64 wasmedge/plugin; break ;;
            2) break ;;
            *) echo "Invalid option." ;;
        esac
    done

    prompt_user_for_architecture_if_not_set
    setup_qemu_if_required

    install_rust
    install_python_and_dependencies 
    
    generate_model_files

    case $arch in 
        "arm64") 
            download_libtorch_arm64
            wasmedge_image="wasmedge/wasmedge:manylinux_2_28_aarch64-plugins-deps"
            ;;
        "amd64") 
            download_libtorch_amd64
            wasmedge_image="wasmedge/wasmedge:manylinux_2_28_x86_64-plugins-deps"
            ;;
    esac

    build_wasmedge
    build_cadvisor
    download_prometheus 
    build_docker_and_native 
    compile_wasm 

    echo "Finished acquiring files for experiments!"
    echo "Note that if you want to add more models and inputs not included by the suite, you can do so by adding them to the models and inputs directories, respectively."
}

function setup_qemu_if_required() {
    # Setup QEMU if the host architecture is different from the target architecture
    host_arch=$(uname -m)
    if [ "$host_arch" = "x86_64" ]; then
        host_arch="amd64"
    elif [ "$host_arch" = "aarch64" ]; then
        host_arch="arm64"
    fi

    if [ "$arch" != "$host_arch" ]; then
        setup_qemu
    fi
}

function setup_qemu() {
    # Setup QEMU for cross-compilation
    echo "Setting up QEMU for cross-compilation..."
    docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
    echo "QEMU setup complete!"
}

function install_rust() {
    # Check if Rust is installed, if not, provide the option to install it
    if ! command -v rustc &> /dev/null; then
        echo "Rust is not installed. Would you like to install Rust through the script?"
            echo "1. Yes"
            echo "2. No"
        read -p "Enter the number identifying your choice: " choice
        if [ "$choice" = "1" ]; then
            echo "Installing Rust..."
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
            source $HOME/.cargo/env
            echo "Rust installed successfully!"
        else
            echo "Please install Rust manually and re-run this script."
            exit 1
        fi
    fi
}

function install_python_and_dependencies() {
    # Check if Python is installed, if not, provide the option to install it
    if ! command -v python3 &> /dev/null; then
        echo "Python3 is not installed. Would you like to install Python3 through the script?"
            echo "1. Yes"
            echo "2. No"
        read -p "Enter the number identifying your choice: " choice
        if [ "$choice" = "1" ]; then
            echo "Installing Python3..."
            if [ "$(uname -s)" != "Darwin" ]; then
                sudo apt install python3 python3-pip python3-venv
            else 
                # For Macs
                brew install python3
            fi
            echo "Python3 installed successfully!"
        else
            echo "Please install Python3 manually and re-run this script."
            exit 1
        fi        
    fi

    # Check if virtualenv already activated

    echo "Loading Python3 virtual environment..."
    python3 -m venv myenv
    source myenv/bin/activate

    pip install -r python/host/requirements.txt
    echo "Python3 virtual environment loaded successfully!"
}

function generate_model_files() {
    # Generate the model files used in the experiments
    echo "Generating model files..."

    mkdir -p models/models

    generate_mobilenet_model
    generate_efficientnet_models
    generate_resnet_models

    echo "Finished generating model files!"
}

function generate_efficientnet_models() {
    # Generate EfficientNet models
    echo "Which EfficientNet models would you like to generate?"
        echo "1. EfficientNetB0"
        echo "2. EfficientNetB1"
        echo "3. EfficientNetB2"
        echo "4. EfficientNetB3"
        echo "5. EfficientNetB4"
        echo "6. EfficientNetB5"
        echo "7. EfficientNetB6"
        echo "8. EfficientNetB7"
    local efficientnet_input
    read -p "Enter the numbers identifying the EfficientNet models you would like to generate (comma-separated): " efficientnet_input
    
    IFS="," read -r -a efficientnet_models_idx <<< "$efficientnet_input"
    local efficientnet_models=()

    for efficientnet_model_idx in "${efficientnet_models_idx[@]}"; do
        case $efficientnet_model_idx in
            1) efficientnet_models+=("--b0") ;;
            2) efficientnet_models+=("--b1") ;;
            3) efficientnet_models+=("--b2") ;;
            4) efficientnet_models+=("--b3") ;;
            5) efficientnet_models+=("--b4") ;;
            6) efficientnet_models+=("--b5") ;;
            7) efficientnet_models+=("--b6") ;;
            8) efficientnet_models+=("--b7") ;;
        esac
    done

    if [ ${#efficientnet_models[@]} -eq 0 ]; then
        echo "No EfficientNet models selected. Skipping generation."
        return
    fi

    cd models/models
    echo "Generating EfficientNet models..."
    python3 ../../host_scripts/model_generation/gen_efficientnet_models.py "${efficientnet_models[@]}"
    echo "Finished generating EfficientNet models!"
    cd -
}

function generate_resnet_models() {
    # Generate ResNet models
    echo "Which ResNet models would you like to generate?"
        echo "1. ResNet18"
        echo "2. ResNet34"
        echo "3. ResNet50"
        echo "4. ResNet101"
        echo "5. ResNet152"
    local resnet_input
    read -p "Enter the numbers identifying the ResNet models you would like to generate (comma-separated): " resnet_input

    IFS="," read -r -a resnet_models_idx <<< "$resnet_input"
    local resnet_models=()

    for resnet_model_idx in "${resnet_models_idx[@]}"; do
        case $resnet_model_idx in
            1) resnet_models+=("--resnet18") ;;
            2) resnet_models+=("--resnet34") ;;
            3) resnet_models+=("--resnet50") ;;
            4) resnet_models+=("--resnet101") ;;
            5) resnet_models+=("--resnet152") ;;
        esac
    done

    if [ ${#resnet_models[@]} -eq 0 ]; then
        echo "No ResNet models selected. Skipping generation."
        return
    fi

    cd models/models
    echo "Generating ResNet models..."
    python3 ../../host_scripts/model_generation/gen_resnet_models.py "${resnet_models[@]}"
    echo "Finished generating ResNet models!"
    cd -
}

function generate_mobilenet_model() {
    # Generate MobileNet models
    echo "Which MobileNet models would you like to generate?"
        echo "1. MobileNetV3-Small"
        echo "2. MobileNetV3-Large"
    local mobilenet_input
    read -p "Enter the numbers identifying the MobileNet models you would like to generate (comma-separated): " mobilenet_input
    
    IFS="," read -r -a mobilenet_models_idx <<< "$mobilenet_input"
    local mobilenet_models=()

    for mobilenet_model_idx in "${mobilenet_models_idx[@]}"; do
        case $mobilenet_model_idx in
            1) mobilenet_models+=("--mobilenetv3_small") ;;
            2) mobilenet_models+=("--mobilenetv3_large") ;;
        esac
    done

    if [ ${#mobilenet_models[@]} -eq 0 ]; then
        echo "No MobileNet models selected. Skipping generation."
        return
    fi

    cd models/models
    echo "Generating MobileNet models..."
    python3 ../../host_scripts/model_generation/gen_mobilenet_models.py "${mobilenet_models[@]}"
    echo "Finished generating MobileNet models!"
    cd -
}

function build_wasmedge() {
    # Build WasmEdge with the PyTorch plugin

    echo "Building WasmEdge ref $WASMEDGE_REF..."

    local build_dir="wasmedge"

    # Remove stale source and install output before switching WasmEdge refs.
    sudo rm -rf WasmEdge "$build_dir"/bin "$build_dir"/include "$build_dir"/lib64 "$build_dir"/plugin
    mkdir -p "$build_dir"

    # Get the WasmEdge source code
    git clone https://github.com/WasmEdge/WasmEdge.git
    cd WasmEdge
    git checkout "$WASMEDGE_REF"
    cd -

    # Give the script to run inside the container the necessary permissions
    chmod +x build_scripts/wasmedge/inside_docker_"$arch".sh

    local platform="linux/$arch"

    # The build can fail transiently, but deterministic compiler errors should
    # stop instead of retrying forever.
    docker pull --platform "$platform" "$wasmedge_image"
    local build_attempt=1
    while true; do
        if sudo docker run --platform "$platform" --rm \
            --entrypoint /bin/bash \
            -v "$(pwd)/$build_dir":/root/wasmedge-install \
            -v "$(pwd)/WasmEdge":/root/wasmedge \
            -v "$(pwd)/libtorch":/root/libtorch \
            -v "$(pwd)/build_scripts/wasmedge/inside_docker_${arch}.sh":/root/inside_docker.sh \
            "$wasmedge_image" -c "git config --global --add safe.directory /root/wasmedge && /root/inside_docker.sh"
        then
            break
        fi

        if [ "$build_attempt" -ge "$WASMEDGE_BUILD_MAX_ATTEMPTS" ]; then
            echo "WasmEdge build failed after $build_attempt attempt(s)."
            exit 1
        fi

        build_attempt=$((build_attempt + 1))
        echo "WasmEdge build failed. Retrying attempt $build_attempt/$WASMEDGE_BUILD_MAX_ATTEMPTS..."
    done

    # Keep the WASI-NN plugin in the location expected by target setup.
    mkdir -p "$build_dir"/plugin

    # This file may be created by another user when the Docker container is run,
    # so use sudo when moving it.
    local wasi_nn_plugin
    wasi_nn_plugin="$(find "$build_dir" -name libwasmedgePluginWasiNN.so -print -quit)"
    if [ -z "$wasi_nn_plugin" ]; then
        echo "Failed to find libwasmedgePluginWasiNN.so after WasmEdge build."
        exit 1
    fi
    if [ "$wasi_nn_plugin" != "$build_dir/plugin/libwasmedgePluginWasiNN.so" ]; then
        sudo mv -f "$wasi_nn_plugin" "$build_dir"/plugin/
    fi

    # Clean up; again, we need to use sudo because some of the files inside this directory
    # were technically created by another user when the Docker container was run
    sudo rm -rf WasmEdge

    echo "Finished building WasmEdge!"
}

function download_libtorch_arm64() {
    # Download libtorch for arm64 architecture
    echo "Downloading libtorch..."

    local build_dir="libtorch"

    # Download and extract libtorch
    curl -s -L -o torch.whl https://download.pytorch.org/whl/cpu/torch-${PYTORCH_VERSION}-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_aarch64.whl
    temp_dir=$(mktemp -d)
    unzip -q torch.whl -d "$temp_dir"
    rm -f torch.whl

    mkdir -p "$build_dir"
    mv "$temp_dir"/torch/lib "$temp_dir"/torch/bin "$temp_dir"/torch/include \
        "$temp_dir"/torch/share "$build_dir"
    rm -rf "$temp_dir"

    echo "Finished downloading libtorch!"
}

function download_libtorch_amd64() {
    echo "Downloading libtorch..."

    local build_dir="libtorch"

    # Download and extract libtorch
    export TORCH_LINK="https://download.pytorch.org/libtorch/cpu/${PYTORCH_ABI}-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip" && \
    curl -s -L -o torch.zip $TORCH_LINK
    temp_dir=$(mktemp -d)
    unzip -q torch.zip -d "$temp_dir"
    rm -f torch.zip

    mkdir -p "$build_dir" 
    mv "$temp_dir"/libtorch/* "$build_dir"

    echo "Finished downloading libtorch!"
}

function build_cadvisor() {
    # Build cAdvisor for the target architecture
    echo "Building cAdvisor..."

    local image_name="cadvisor-build:$arch"
    local build_dir="cadvisor"
    mkdir -p "$build_dir"

    # Build the container that will build cAdvisor
    if ! docker_build_image "$image_name" Dockerfiles/cadvisor_build/Dockerfile; then
        echo "Failed to build cAdvisor image."
        exit 1
    fi

    # Create a temporary container so we can extract the 
    # build results
    local container_id
    if ! container_id=$(docker create "$image_name"); then
        echo "Failed to create cAdvisor build container."
        exit 1
    fi
    if ! docker cp "$container_id":/output/. "$build_dir"; then
        docker rm "$container_id" >/dev/null 2>&1 || true
        echo "Failed to copy cAdvisor binary from build container."
        exit 1
    fi

    # Clean up the container
    docker rm "$container_id"

    echo "Finished building cAdvisor!"
}

function docker_build_image() {
    local image_name="$1"
    local dockerfile="$2"
    local cache_mode="${3:-cache}"

    local build_args=(
        --platform "linux/$arch"
        --build-arg "TARGETARCH=$arch"
        --build-arg "BUILDPLATFORM=linux/$arch"
        -t "$image_name"
        -f "$dockerfile"
    )

    if [ "$cache_mode" = "no-cache" ]; then
        build_args=(--no-cache "${build_args[@]}")
    fi

    if docker buildx version >/dev/null 2>&1; then
        docker buildx build "${build_args[@]}" --load .
    else
        echo "docker buildx is not available; falling back to docker build."
        docker build "${build_args[@]}" .
    fi
}

function download_prometheus() {
    # Download Prometheus for the target architecture
    echo "Downloading Prometheus..."

    local url="https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-${arch}.tar.gz"
    local output_file="prometheus.tar.gz"
    local extract_dir="prometheus-${PROMETHEUS_VERSION}.linux-${arch}"
    local temp_dir="prometheus_temp"
    
    curl -L -o "$output_file" "$url"
    mkdir -p "$temp_dir"
    tar -xzf "$output_file" -C "$temp_dir"
    mv "$temp_dir"/"$extract_dir"/prometheus prometheus

    rm -rf "$temp_dir"
    rm "$output_file"

    echo "Finished downloading Prometheus!"
}

function build_docker_and_native() {
    # Build the Docker image and native binary for the target architecture
    echo "Building the Docker image and native binary..."
    local image_name="image-classification:$arch"

    # Build the Docker container
    if ! docker_build_image "$image_name" Dockerfiles/image_classification/Dockerfile; then
        echo "Failed to build Docker image and native binary."
        exit 1
    fi

    # Save the Docker container into a tar file
    if ! docker save -o docker/image-classification-${arch}.tar "$image_name"; then
        echo "Failed to save Docker image."
        exit 1
    fi

    # Extract the binary compiled for the container so it can also be run for the 
    # native deployment mechanism
    mkdir -p native
    local container_id
    if ! container_id=$(docker create "$image_name"); then
        echo "Failed to create native extraction container."
        exit 1
    fi
    if ! docker cp "$container_id":/torch_image_classification native; then
        docker rm "$container_id" >/dev/null 2>&1 || true
        echo "Failed to copy native binary from image."
        exit 1
    fi

    # Clean up the container
    docker rm "$container_id"

    echo "Finished building the Docker image and native binary!"
}

function compile_wasm() {
    # Compile the WebAssembly binary for the target architecture
    echo "Compiling the WebAssembly binary..."

    cd rust/wasm
    rustup target add wasm32-wasip1
    cargo build --target=wasm32-wasip1 --release
    cd - 

    # Move the compiled Wasm binary to the wasm directory
    mkdir -p wasm
    local wasm_src="rust/wasm/target/wasm32-wasip1/release/interpreted.wasm"
    local wasm_dst="wasm/interpreted.wasm"
    if [ -e "$wasm_dst" ] && [ "$wasm_src" -ef "$wasm_dst" ]; then
        echo "WebAssembly binary already exists at $wasm_dst."
    else
        mv -f "$wasm_src" "$wasm_dst"
    fi

    echo "Finished compiling the WebAssembly binary!"
}

function transfer_files() {
    # Transfer the files required to run the experiments to the target machine
    prompt_user_for_target_details_if_not_set

    # Transfer the suite files to the target machine, including 
    # models, inputs, native & wasm binaries, libtorch, Docker tar file, the
    # data collection Python file, and the Cadvisor and Prometheus binaries
    echo "Transferring suite files to target device..."
    transfer_suite_files
    echo "Finished transferring suite files!"

    # Transfer the results of the WasmEdge build, which will be located
    # in a separate location to the rest of the suite
    echo "Transferring WasmEdge files to target device..."
    transfer_wasmedge_files
    echo "Finished transferring WasmEdge files!"
}   

function transfer_suite_files() {
    # Transfer the suite files to the target machine

    # Create a directory to store the suite files in the target machine
    sshpass -p "$target_password" ssh "$target_username"@"$target_address" "mkdir -p /home/$target_username/Desktop/$SUITE_NAME"

    # Transfer the suite files to the target machine
    sshpass -p "$target_password" scp -r models/models inputs/inputs native wasm libtorch cadvisor prometheus python docker target_scripts data_scripts/collect_data.py \
        "$target_username"@"$target_address":/home/"$target_username"/Desktop/"$SUITE_NAME"

    # Create a directory in the suite directory to store results 
    sshpass -p "$target_password" ssh "$target_username"@"$target_address" "mkdir -p /home/$target_username/Desktop/$SUITE_NAME/results"
}   

function transfer_wasmedge_files() {
    # Transfer the WasmEdge files to the target machine

    # Create a directory to store the WasmEdge files in the target machine
    sshpass -p "$target_password" ssh "$target_username"@"$target_address" "mkdir -p /home/$target_username/.wasmedge"

    # Transfer the WasmEdge files to the target machine
    sshpass -p "$target_password" scp -r wasmedge/* "$target_username"@"$target_address":/home/"$target_username"/.wasmedge
}

function setup_target_machine() {
    # Setup the target machine through the setup script
    prompt_user_for_target_details_if_not_set

    # Ask the user if they want to run the script directly, in case the target
    # machine has an Internet connection, or if they want to simply transfer it in
    # case the target machine cannot access the Internet while connected to the host
    echo "How would you like to setup the target machine?"
    echo "Note that running the script requires that the target machine has an Internet connection." 
    echo "Also note that it requires sudo permissions."
        echo "1. Run the setup script directly on the target machine from this machine"
        echo "2. Run the setup script manually on the target machine"

    while true; do
        local setup_option
        read -p "Enter the number identifying the setup option: " setup_option
        case $setup_option in
            1) run_setup_script_remotely; break ;;
            2) run_setup_script_manually_prompt; break ;;
            *) echo "Invalid option. Please try again." 
        esac
    done
}

function run_setup_script_remotely() {
    # Run the setup script on the target machine remotely
    prompt_user_for_mac_if_not_set

    echo "Running setup script on target device..."
    
    if [ "$is_mac" = 1 ]; then
        sshpass -p "$target_password" ssh -t "$target_username@$target_address" "chmod +x /home/$target_username/Desktop/$SUITE_NAME/target_scripts/setup.sh && sudo /home/$target_username/Desktop/$SUITE_NAME/target_scripts/setup.sh -m"
    else
        sshpass -p "$target_password" ssh -t "$target_username@$target_address" "chmod +x /home/$target_username/Desktop/$SUITE_NAME/target_scripts/setup.sh && sudo /home/$target_username/Desktop/$SUITE_NAME/target_scripts/setup.sh"
    fi

    echo "Finished running setup script on target device!"
}

function run_setup_script_manually_prompt() {
    # Prompt the user to run the setup script manually on the target machine
    echo "Please run the script on the target as follows: sudo /home/$target_username/Desktop/$SUITE_NAME/target_scripts/setup.sh"
    echo "If the target is on a Mac, please run the script on the target as follows instead: sudo /home/$target_username/Desktop/$SUITE_NAME/target_scripts/setup.sh -m"

    echo "You may need to disconnect the target machine's Ethernet connection to allow it to connect to the Internet."
    echo "Press enter to continue once you have run the script on the target machine."
    read -r 
}

function run_data_collection() {
    # Run the data collection script on the target machine
    prompt_user_for_target_details_if_not_set
    prompt_user_for_mac_if_not_set

    echo "Running data collection on target device..."

    local set_name
    read -p "Enter a name to identify this set of experiments: " set_name

    while true; do
        local trials
        read -p "Enter the number of trials to run for each experiment (at least 2): " trials
        if [ "$trials" -ge 2 ]; then
            break
        else
            echo "Invalid number of trials. Please enter a number greater than or equal to 2."
        fi
    done

    prompt_user_for_mechanisms

    echo "Would you like to allow experiment trials to have missing data on some metrics (e.g. instructions retired)?"
    echo "If you have doubts that the target machine can access perf metrics, you should answer 'yes'."
    echo "Additionally, if you are running the experiments on a virtual machine, you are advised to answer 'yes' as other metrics may be unavailable."
        echo "1. Yes"
        echo "2. No"

    while true; do
        local allow_missing_metrics_input
        read -p "Enter the number identifying your choice: " allow_missing_metrics_input
        case $allow_missing_metrics_input in
            1) allow_missing_metrics=1; break ;;
            2) allow_missing_metrics=0; break ;;
            *) echo "Invalid option. Please try again." ;;
        esac
    done

    options=""
    if [ "$is_mac" = 1 ]; then
        options="$options -m"
    fi
    if [ "$allow_missing_metrics" = 1 ]; then
        options="$options -a"
    fi

    sshpass -p "$target_password" ssh -t "$target_username@$target_address" "/home/$target_username/Desktop/$SUITE_NAME/target_scripts/collect_data.sh $options $trials "$set_name" $mechanisms"

    echo "Finished running data collection on target device!"
}

function retrieve_data_collection_results() {
    # Retrieve the results of the data collection from the target machine
    prompt_user_for_target_details_if_not_set

    echo "Retrieving results from target device..."

    local set_name
    read -p "Enter the name of the set of experiments to retrieve results from: " set_name

    sshpass -p "$target_password" scp -r "$target_username@$target_address:/home/$target_username/Desktop/$SUITE_NAME/results/$set_name" results
    
    echo "Finished retrieving results from target device!"
}

function run_data_analysis() {
    # Run the data analysis scripts on the host machine
    install_python_and_dependencies

    echo "Analyzing results of experiments..."

    local set_name
    read -p "Enter the name of the set of experiments to analyze: " set_name

    local analyzed_results_dir 
    read -p "Enter the name of the directory to create within results/$set_name where the analyzed results will be stored (default: analyzed_results): " analyzed_results_dir
    if [ -z "$analyzed_results_dir" ]; then
        analyzed_results_dir="analyzed_results"
    fi

    run_per_experiment_data_analysis "$set_name" "$analyzed_results_dir"

    echo "Would you like to perform aggregate analysis on the entire set of experiments, using the results of the previous analyses?"
        echo "1. Yes"
        echo "2. No"

    while true; do
        local aggregate_analysis
        read -p "Enter the number identifying your choice: " aggregate_analysis
        case $aggregate_analysis in
            1) run_aggregate_data_analysis "$set_name" "$analyzed_results_dir"; break ;;
            2) break ;;
            *) echo "Invalid option. Please try again." ;;
        esac
    done

    echo "Finished analyzing results of experiments!"
}

function run_per_experiment_data_analysis() { 
    # Run data analysis for each experiment in a specified set
    set_name="$1"
    analyzed_results_dir="$2"

    echo "Running data analysis for each experiment in the set..."

    local significance_level
    read -p "Enter the significance level to be used for the analysis (default: 0.05): " significance_level
    if [ -z "$significance_level" ]; then
        significance_level=0.05
    fi

    options=""
    
    echo "Would you like to view the output for each experiment? If you answer 'no', you can still choose to save the outputs to files."
        echo "1. Yes"
        echo "2. No"

    while true; do
        local view_output
        read -p "Enter the number identifying your choice: " view_output
        case $view_output in
            1) options="$options --view-output"; break ;;
            2) break ;;
            *) echo "Invalid option. Please try again." ;;
        esac
    done

    echo "Would you like to save the output for each experiment to a file?"
        echo "1. Yes"
        echo "2. No"

    while true; do
        local save_output
        read -p "Enter the number identifying your choice: " save_output
        case $save_output in
            1) options="$options --save-output"; break ;;
            2) break ;;
            *) echo "Invalid option. Please try again." ;;
        esac
    done

    prompt_user_for_mechanisms
    prompt_user_for_metrics "$set_name"

    echo "Which view of the Docker deployment mechanism's overhead would you like to use?"
        echo "1. Include only the Docker container's overhead"
        echo "2. Include the Docker container's overhead and the Docker daemon's full overhead"
        echo "3. Include the Docker container's overhead and the Docker daemon's estimated additional overhead due to the container"
    
    while true; do
        local docker_overhead_input
        read -p "Enter the number identifying your choice: " docker_overhead_input
        case $docker_overhead_input in
            1) docker_overhead=0; break ;;
            2) docker_overhead=1; break ;;
            3) docker_overhead=2; break ;;
            *) echo "Invalid option. Please try again." ;;
        esac
    done

    if [ "$view_output" = 1 ]; then
        echo "Would you like to print statistically insignificant output during the analysis?"
            echo "1. Yes"
            echo "2. No"

        while true; do
            local include_insig_output
            read -p "Enter the number identifying your choice: " include_insig_output
            case $include_insig_output in
                1) options="$options --include-insignificant-output"; break; ;;
                2) break ;;
                *) echo "Invalid option. Please try again." ;;
            esac
        done
    fi
    
    # For each combination of model and input, there's a perf results file and a time results file
    # so only need to iterate over one of them
    for results_file in $(ls results/"$set_name"/*time_results.csv); do
        model=$(basename "$results_file" | cut -d '&' -f 1)
        input=$(basename "$results_file" | cut -d '&' -f 2)
        echo "Analyzing data for $model and $input..."
        python3 data_scripts/analyze_data.py \
            --experiment-set "$set_name" \
            --model "$model" \
            --input "$input" \
            --significance-level "$significance_level" \
            --docker-overhead-view "$docker_overhead" \
            --mechanisms "$mechanisms" \
            --metrics "$metrics" \
            --analyzed-results-dir "$analyzed_results_dir" \
            $options 
        echo "Analysis complete."
        echo "Press Enter to continue to the next experiment..."
        read -r
    done

    echo "Finished running data analysis for each experiment in the set!"
    echo "The results of the analysis are stored in the results/$set_name/analyzed_results directory."
}

function prompt_user_for_mechanisms() {
    # Prompt the user for the deployment mechanisms they would like to include in the analysis
    echo "Which deployment mechanisms would you like to include? Note that when analyzing data, you can only include deployment mechanisms that were included in the data collection."
        echo "1. Native"
        echo "2. Docker"
        echo "3. WebAssembly interpreted"
        echo "4. WebAssembly JIT-compiled"
        echo "5. WebAssembly ahead of time (AoT)-compiled"
    local mechanisms_input
    read -p "Enter the numbers identifying the deployment mechanisms you would like to include (comma-separated): " mechanisms_input

    # Convert the mechanisms input into a comma-separated string eg. "native,docker"
    IFS=',' read -ra mechanisms_array <<< "$mechanisms_input"
    mechanisms=()

    for mechanism in "${mechanisms_array[@]}"; do
        case $mechanism in
            1) mechanisms+=("native") ;;
            2) mechanisms+=("docker") ;;
            3) mechanisms+=("wasm_interpreted") ;;
            4) mechanisms+=("wasm_jit") ;;
            5) mechanisms+=("wasm_aot") ;;
        esac
    done

    mechanisms=$(IFS=,; echo "${mechanisms[*]}")
}

function prompt_user_for_metrics() {
    # Prompt the user for the metrics they would like to analyze
    set_name="$1" # 

    echo "Based on the data collected, the following metrics are available for analysis:"
        # Read one time file and one perf file to get the available metrics
        time_file=$(ls results/"$set_name"/*time_results.csv | head -n 1)
        perf_file=$(ls results/"$set_name"/*perf_results.csv | head -n 1)

        # The first 3 columns are the same for all result files; they only include non-metric columns
        # such as deployment mechanism, trial number and start time
        time_metrics=$(head -n 1 "$time_file" | tr -d '\r' | cut -d ',' -f 4-)
        perf_metrics=$(head -n 1 "$perf_file" | tr -d '\r' | cut -d ',' -f 4-)

        # Include derived metrics; time_metrics will always include all metrics required to calculate these
        time_metrics="$time_metrics,model-load-time-(seconds),input-preparation-time-(seconds),inference-time-(seconds),input-resize-time-(seconds),input-load-time-(seconds)"

        # If perf_metrics includes instructions and cycles (since some experiments might not have them), then we 
        # must additionally consider the instructions-per-cycle and cycles-per-instruction metrics calculated in the analysis
        if [[ "$perf_metrics" == *"instructions"* && "$perf_metrics" == *"cycles"* ]]; then
            perf_metrics="$perf_metrics,instructions-per-cycle,cycles-per-instruction"
        fi

        echo "$time_metrics,$perf_metrics"
    read -p "Enter the metrics you would like to analyze (comma-separated). If nothing is entered, all metrics will be used: " metrics
    if [ -z "$metrics" ]; then
        metrics="$time_metrics,$perf_metrics"
    fi
}

function list_models_and_inputs() {
    # List the models and inputs available for analysis based on the data collected
    set_name="$1"
    analyzed_results_dir="$2"

    echo "Based on the data collected, the following models and inputs are available for analysis:"
        models=$(list_models "$set_name" "$analyzed_results_dir")        
        inputs=$(list_inputs "$set_name" "$analyzed_results_dir")

        echo "models: $models"
        echo "inputs: $inputs"
}

function list_inputs() {
    # List the inputs available for analysis based on the data collected
    set_name="$1"
    analyzed_results_dir="$2"

    # Read the aggregate results file's unique values for the second column, which
    # corresponds to the input 
    inputs=$(cut -d ',' -f 2 results/"$set_name"/"$analyzed_results_dir"/aggregate_results.csv | tail -n +2 | sort -u| paste -sd, -)

    echo "$inputs"
}

function list_models() {
    # List the models available for analysis based on the data collected
    set_name="$1"
    analyzed_results_dir="$2"

    # Read the aggregate results file's unique values for the first column, which
    # corresponds to the model 
    models=$(cut -d ',' -f 1 results/"$set_name"/"$analyzed_results_dir"/aggregate_results.csv | tail -n +2 | sort -u| paste -sd, -)

    echo "$models"
}

function run_aggregate_data_analysis() {
    # Run aggregate data analysis on the results of the analysis of a set of experiments
    set_name="$1"
    analyzed_results_dir="$2"

    echo "Running aggregate data analysis for the set of experiments..."

    prompt_user_for_metrics "$set_name"

    local options=""
    echo "Would you like to view all of the plots that will be produced? If you answer 'no', you can still choose to save the outputs to files."
        echo "1. Yes"
        echo "2. No"

    while true; do
        local view_output
        read -p "Enter the number identifying your choice: " view_output
        case $view_output in
            1) options="$options --view-output"; break ;;
            2) break ;;
            *) echo "Invalid option. Please try again." ;;
        esac
    done

    echo "Would you like to save all of the plots to files?"
        echo "1. Yes"
        echo "2. No"

    while true; do
        local save_output
        read -p "Enter the number identifying your choice: " save_output
        case $save_output in
            1) options="$options --save-output"; break ;;
            2) break ;;
            *) echo "Invalid option. Please try again." ;;
        esac
    done

    constant_options="$options"

    while true; do
        echo "What would you like to do?"
            echo "1. Compare across models for specific inputs"
            echo "2. Compare across inputs for specific models"
            echo "3. Compare across models for all inputs"
            echo "4. Compare across inputs for all models"
            echo "5. Finish aggregate analysis"
        local action
        read -p "Enter the number identifying the action you would like to perform: " action
        case $action in
            1) compare_across_models ;;
            2) compare_across_inputs ;;
            3) compare_across_models_all_inputs ;;
            4) compare_across_inputs_all_models ;;
            5) prompt_user_for_action ;;
            *) echo "Invalid option." ;;
        esac
        options="$constant_options"
    done

    echo "Finished running aggregate data analysis for the set of experiments!"
    echo "The results of the analysis are stored in the results/$set_name/analyzed_results directory."
}

function compare_across_models() {
    list_models_and_inputs "$set_name" "$analyzed_results_dir"
    read -p "Enter the models to compare (comma-separated): " models_to_compare
    read -p "Enter the input to use in comparing models: " input
    options="$options --compare-across-models --models-to-compare $models_to_compare --input $input"
    prompt_user_for_chart_type

    python3 data_scripts/analyze_aggregate_data.py \
        --experiment-set "$set_name" \
        --metrics "$metrics" \
        --analyzed-results-dir "$analyzed_results_dir" \
        $options
}

function compare_across_models_all_inputs() {
    list_models_and_inputs "$set_name" "$analyzed_results_dir"
    read -p "Enter the models to compare (comma-separated): " models_to_compare
    options="$options --compare-across-models --models-to-compare $models_to_compare"
    prompt_user_for_chart_type

    # For each input, run the analysis
    inputs=$(list_inputs "$set_name" "$analyzed_results_dir")
    for input in $(echo "$inputs" | tr ',' ' '); do
        echo "Comparing across models for input: $input"
        options="$options --input $input"
        
        python3 data_scripts/analyze_aggregate_data.py \
            --experiment-set "$set_name" \
            --metrics "$metrics" \
            --analyzed-results-dir "$analyzed_results_dir" \
            $options
    done
}

function compare_across_inputs() {
    list_models_and_inputs "$set_name" "$analyzed_results_dir"
    read -p "Enter the inputs to compare (comma-separated): " inputs_to_compare
    read -p "Enter the model to use in comparing inputs: " model
    options="$options --compare-across-inputs --inputs-to-compare $inputs_to_compare --model $model"
    prompt_user_for_chart_type

    python3 data_scripts/analyze_aggregate_data.py \
        --experiment-set "$set_name" \
        --metrics "$metrics" \
        --analyzed-results-dir "$analyzed_results_dir" \
        $options
}

function compare_across_inputs_all_models() {
    list_models_and_inputs "$set_name" "$analyzed_results_dir"
    read -p "Enter the inputs to compare (comma-separated): " inputs_to_compare
    options="$options --compare-across-inputs --inputs-to-compare $inputs_to_compare"
    prompt_user_for_chart_type

    # For each model, run the analysis
    models=$(list_models "$set_name" "$analyzed_results_dir")
    for model in $(echo "$models" | tr ',' ' '); do
        echo "Comparing across inputs for model: $model"
        options="$options --model $model"
        
        python3 data_scripts/analyze_aggregate_data.py \
            --experiment-set "$set_name" \
            --metrics "$metrics" \
            --analyzed-results-dir "$analyzed_results_dir" \
            $options
    done
}

function prompt_user_for_chart_type() {
    echo "Which type of chart would you like to generate?"
        echo "1. Bar chart"
        echo "2. Line plot"

    while true; do
        local chart_type_input
        read -p "Enter the number identifying your choice: " chart_type_input
        case $chart_type_input in
            1) options="$options --chart-type bar"; break ;;
            2) options="$options --chart-type lineplot"; break ;;
            *) echo "Invalid option. Please try again." ;;
        esac
    done
}

function explain_suite() {
    # Explain how the suite works

    cat << EOF
This suite is designed to automate the process of characterizing the performance of edge ML inference
on a target machine using different deployment mechanisms, namely Docker, interpreted WebAssembly, ahead-of-time
(AOT)-compiled WebAssembly, and native. Currently, it only supports conducting performance characterization for
image classification tasks.

This includes the following steps:
1. Acquiring the files necessary for experiments to be run on the target machine, including models, the ML library,
   the Docker image, binaries containing ML inference code, and applications for performance monitoring.
2. Transferring to the target machine all the files it requires to run the experiments.
3. Setting up the target machine to enable it to run the experiments, including installing dependencies and performing
   ahead-of-time compilation of the WebAssembly binary.
4. Running a named set of experiments on the target machine to collect data. In essence, for each combination of models and inputs
   transferred over, as stored in the models and inputs directories, an experiment will be run consisting of 20 trials for
   each deployment mechanism. Each trial involves executing ML inference on the given input using the given model through
   the deployment mechanism (e.g. for Docker, by executing a container that performs this inference). Data on performance
   metrics associated with this inference, such as wall time, memory usage, and instructions retired will be collected and stored
   in result files.
5. Retrieving the results of a set of experiments from the target machine to the host machine you are running this script from.
6. Analyzing the results of a set of experiments. This includes determining if there are statistically significant differences
   between the performance of different deployment mechanisms; generating plots to chart the performance of different mechanisms
   for each metric; and generating plots to chart how each mechanism's performance on a given metric varies across different models or inputs.

Note that if the models and inputs provided by the suite are insufficient, you may add your own to the models and inputs folders.
Additionally, if you wish to include more lower-level performance metrics to measure, you may modify the perf_config.json file in the
cadvisor directory.
EOF
}

main
