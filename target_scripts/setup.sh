#!/bin/bash
# This script is meant to be run on the target machine to perform one-time setup of the environment 
# in preparation for the experiments.

export USERNAME=${SUDO_USER:-$(whoami)}
export SUITE_PATH="/home/$USERNAME/Desktop/CS4099Suite"

function main() {
    if [ "$(uname -m)" = "aarch64" ]; then
        arch="arm64"
    else
        arch="amd64"
    fi

    apt-get update

    setup_wasmedge

    if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
        # cgroup v2 is active, must enable memory controller
        enable_memory_controller
    fi

    install_time
    setup_docker
    load_docker_image
    setup_cadvisor
    setup_prometheus
    setup_python
    setup_collect_data

    # Setup LD_LIBRARY_PATH in preparation for AoT compilation
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$SUITE_PATH/libtorch/lib
    if [ "$mac" = 1 ]; then
        aot_compile_wasm_mac
    else
        aot_compile_wasm_non_mac
    fi
}

function install_time() {
    # Install the GNU time command used by the data collection script
    apt-get install -y time
}

function setup_wasmedge() {
    # Grant execute permissions to the WasmEdge binary and sets up appropriate paths and links
    chmod u+x "/home/$USERNAME/.wasmedge/bin/wasmedge"

    # Add wasmedge to the PATH for the remainder of this script
    # since it will be used in various parts
    export PATH=${PATH}:/home/$USERNAME/.wasmedge/bin

    # Create symbolic links expected by WasmEdge in case they are missing
    if [ ! -e "/home/$USERNAME/.wasmedge/lib64/libwasmedge.so" ]; then
        ln -s "/home/$USERNAME/.wasmedge/lib64/libwasmedge.so.0.1.0" "/home/$USERNAME/.wasmedge/lib64/libwasmedge.so"
    fi

    if [ ! -e "/home/$USERNAME/.wasmedge/lib64/libwasmedge.so.0" ]; then
        ln -s "/home/$USERNAME/.wasmedge/lib64/libwasmedge.so.0.1.0" "/home/$USERNAME/.wasmedge/lib64/libwasmedge.so.0"
    fi
}

function enable_memory_controller() {
    # Enable the memory controller for cgroup v2 devices so that memory usage metrics
    # can be collected
    local cmdline_file="/boot/firmware/cmdline.txt"
    local cgroup_enable_param="cgroup_enable=memory"
    local current_cmdline_contents="$(cat "$cmdline_file")"

    if ! grep -q "$cgroup_enable_param" <<< "$current_cmdline_contents"
    then 
        cp "$cmdline_file" "$cmdline_file.bak"
        local new_cmdline_contents="${current_cmdline_contents} ${cgroup_enable_param}"
        sh -c "echo '$new_cmdline_contents' > $cmdline_file"
    fi
}

function setup_docker() {
    # Check if Docker is installed, if not, provide the option to install it
    if ! command -v docker &> /dev/null; then
        echo "Docker is not installed. Would you like to install Docker through the script?"
            echo "1. Yes"
            echo "2. No"
        read -p "Enter the number identifying your choice: " choice
        if [ "$choice" = "1" ]; then
            chmod u+x "$SUITE_PATH/docker/install-docker.sh"
            "$SUITE_PATH"/docker/install-docker.sh
        else
            echo "Please install Docker manually and re-run this script."
            exit 1
        fi
    fi
}

function load_docker_image() {
    # Load the Docker image for the image classification model
    docker load -i "$SUITE_PATH/docker/image-classification-$arch.tar"
}

function setup_cadvisor() {
    # Grant execute permissions to the cAdvisor binary and install required packages
    chmod u+x "$SUITE_PATH/cadvisor/cadvisor"
    apt install libpfm4 linux-perf
}

function setup_prometheus() {
    # Grant execute permissions to the Prometheus binary
    chmod u+x "$SUITE_PATH/prometheus/prometheus"
}

function setup_python() {
    # Check if Python is installed, if not, provide the option to install it
    if ! command -v python3 &> /dev/null
    then
        echo "Python3 is not installed. Would you like to install Python3 through the script?"
            echo "1. Yes"
            echo "2. No"
        read -p "Enter the number identifying your choice: " choice
        if [ "$choice" = "1" ]; then
            apt install -y python3 python3-venv python3-pip
        else
            echo "Please install Python3 manually and re-run this script."
            exit 1
        fi
    fi

    # Install cgroup tools to allow management of custom cgroups for data collection
    apt install -y cgroup-tools

    # Create a Python virtual environment and install packages required by the data collection script
    python3 -m venv "$SUITE_PATH/myenv" 
    source "$SUITE_PATH/myenv/bin/activate" && pip install -r "$SUITE_PATH/python/target/requirements.txt"
}

function setup_collect_data() {
    # Grant execute permissions to the collect_data.sh script
    chmod u+x "$SUITE_PATH/target_scripts/collect_data.sh"
}

function aot_compile_wasm_non_mac() {
    # AoT compile the WebAssembly code (if not on Mac)
    wasmedge compile "$SUITE_PATH/wasm/interpreted.wasm" "$SUITE_PATH/wasm/aot.wasm"
}

function aot_compile_wasm_mac() {
    # AoT compile the WebAssembly code (if on Mac)
    wasmedge compile "$SUITE_PATH/wasm/interpreted.wasm" "$SUITE_PATH/wasm/aot.so"
}

# Check if user passed in an optional argument specifying the target machine
# is on a Mac
while getopts "m" opt; do
    case $opt in
        m) mac=1 ;;
        \?) echo "Invalid option: -$OPTARG" >&2
            exit 1 ;;
    esac
done

main