#!/bin/bash
# This script is meant to be run on the Raspberry Pi to perform the data collection experiments for a number
# of different models and inputs.

export USERNAME=$(whoami)
export SUITE_PATH="/home/$USERNAME/Desktop/CS4099Suite"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/$USERNAME/.wasmedge/lib64:$SUITE_PATH/libtorch/lib
export PATH=${PATH}:/home/$USERNAME/.wasmedge/bin

function main() {
    # Check for the required arguments
    if [ "$#" -ne 3 ]; then
        echo "Usage: $0 <trials> <experiments set name> <mechanisms>"
        exit 1
    fi

    if [ "$(uname -m)" = "aarch64" ]; then
        arch="arm64"
    else
        arch="amd64"
    fi

    trials=$1
    set_name=$2
    mechanisms=$3

    # Create the results directory 
    cd "$SUITE_PATH"
    mkdir -p "results/$set_name"

    # Activate the Python environment with the necessary dependencies to run the data 
    # collection script
    source myenv/bin/activate

    run_data_collection
}

function run_data_collection() {
    # Iterate over each model file in the models folder and each input file in the inputs folder,
    # and run the collect_data.py script on them with the specified options
    for model in models/*; do
        for input in inputs/*; do
            if [ -f "$model" ] && [ -f "$input" ]; then
                basename_model=$(basename "$model")
                basename_input=$(basename "$input")
                echo "Running collect_data.py with model: $basename_model and input: $basename_input"

                options=""
                if [ "$is_mac" = 1 ]; then
                    options="$options --is_mac"
                fi
                if [ "$allow_missing_metrics" = 1 ]; then
                    options="$options --allow_missing_metrics"
                fi

                python collect_data.py --model "$basename_model" --input "$basename_input" \
                    --trials $trials --set_name $set_name --mechanisms "$mechanisms" \
                    --arch $arch $options
            fi
        done
    done
}

# Check for optional arguments: -a for allowing missing perf events and -m for Mac
while getopts "am" opt; do
    case $opt in
        a)
            allow_missing_metrics=1
            ;;
        m)
            is_mac=1
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# Remove the processed optional arguments
shift $((OPTIND - 1))

main $1 $2 $3