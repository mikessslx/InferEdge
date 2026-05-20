"""This script runs all the experiments and collects the relevant data, storing it in the 
   specified files.
"""
import requests
import json
import subprocess
import csv
import random
import os
import argparse
import time
from datetime import datetime, timezone
from sys import platform

# The root of the suite directory where this script is in
SUITE_DIR = os.path.abspath(os.path.dirname(__file__))

# The absolute path of the directory storing results
RESULTS_DIR = os.path.join(SUITE_DIR, "results")

# Path to the WebAssembly binary
WASM_BINARY_PATH = os.path.expanduser("~/.wasmedge/bin/wasmedge")

# Path to the WebAssembly files
INTERPRETED_WASM_FILE_PATH = f"{SUITE_DIR}/wasm/interpreted.wasm"
AOT_WASM_FILE_PATH_TEMPLATE = f"{SUITE_DIR}/wasm/aot.{{extension}}"

# Path to cAdvisor perf events config file
CADVISOR_PERF_CONFIG_PATH = f"{SUITE_DIR}/cadvisor/perf_config.json"

# The list of perf events to measure
def read_perf_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        return config.get("core").get("events")

PERF_EVENTS = read_perf_config(CADVISOR_PERF_CONFIG_PATH)

# Path to the cAdvisor, Prometheus binaries
CADVISOR_BINARY_PATH = f"{SUITE_DIR}/cadvisor/cadvisor"
PROMETHEUS_BINARY_PATH = f"{SUITE_DIR}/prometheus/prometheus"

# Path to the native-compiled code
NATIVE_BINARY_NAME = "torch_image_classification"
NATIVE_BINARY_PATH = f"{SUITE_DIR}/native/{NATIVE_BINARY_NAME}"

# Path to the models and inputs directories
MODELS_PATH = f"{SUITE_DIR}/models"
INPUTS_PATH = f"{SUITE_DIR}/inputs"

# Container and image names
CONTAINER_NAME="benchmarked-container"
IMG_NAME_TEMPLATE="image-classification:{arch}"         

# Commands to start, stop, remove, inspect container
CONTAINER_START_CMD_TEMPLATE = f"sudo docker run --privileged --name {CONTAINER_NAME} -v {MODELS_PATH}:/models -v {INPUTS_PATH}:/inputs {{img_name}}"
CONTAINER_STOP_CMD = "sudo docker stop {container_name}"
CONTAINER_REMOVE_CMD = "sudo docker rm {container_name}"
CONTAINER_INSPECT_ID_CMD = "sudo docker inspect -f '{{{{.Id}}}}' {container_name}"

# Commands to start and stop Prometheus, cAdvisor
PROMETHEUS_START_CMD = f"sudo {PROMETHEUS_BINARY_PATH} --config.file={SUITE_DIR}/prometheus/prometheus.yml --web.enable-admin-api" 
PROMETHEUS_STOP_CMD = f"sudo pkill -f {PROMETHEUS_BINARY_PATH}"
CADVISOR_START_CMD = f"sudo {CADVISOR_BINARY_PATH} -perf_events_config={CADVISOR_PERF_CONFIG_PATH}"
CADVISOR_STOP_CMD = f"sudo pkill -f {CADVISOR_BINARY_PATH}"

# Values for the PATH and LD_LIBRARY_PATH environment variables
LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH")
PATH = os.environ.get("PATH")

# The command used to measure wall time
TIME_CMD_PREFIX="/usr/bin/time -v"

# A list of tuples containing the names of metrics from running time to measure, alongside the 
# new name of the metric as it will be written in the results file
TIME_METRICS = [("Elapsed (wall clock) time", "wall-time-seconds"), 
    ("Time until model loaded in seconds", "until-model-loaded-time-seconds"),
    ("Time until input loaded in seconds", "until-input-loaded-time-seconds"),
    ("Time until input resized in seconds", "until-input-resized-time-seconds"),
    ("Time until input ready in seconds", "until-input-ready-time-seconds"),
    ("Time until inference executed in seconds", "until-inference-executed-time-seconds"),
    ("Total workload duration in seconds", "workload-time-seconds")]

# A list of short names for the time metrics, used in the results file; also includes "overhead-time-seconds", which is
# not part of any command's output, but is rather derived from wall time and inference time
TIME_METRICS_SHORT_NAMES = [time_metric[1] for time_metric in TIME_METRICS] + ["overhead-time-seconds"]

# The endpoint that Prometheus is listening on
PROMETHEUS_URL="http://localhost:9090"

# The suffixes of the filenames to store results in
PERF_RESULTS_FILENAME_SUFFIX = "&perf_results.csv"
TIME_RESULTS_FILENAME_SUFFIX = "&time_results.csv"

# Basic field names to include in every CSV file storing experiment results
CSV_BASIC_FIELD_NAMES = ["deployment-mechanism", "trial-number", "start-time"] 

# Field names for memory metrics
MEMORY_FIELD_NAMES = ["average-memory-over-time-in-bytes", "maximum-memory-over-time-in-bytes"]

# Field names for CPU metrics
CPU_FIELD_NAMES = ["CPU-total-utilization-percentage", "CPU-user-utilization-percentage", "CPU-system-utilization-percentage"]

# Field names for events that might be missing/not available for cAdvisor and Prometheus
# depending on the system
POSSIBLE_MISSING_METRICS = PERF_EVENTS + CPU_FIELD_NAMES

# Number of CPU cores 
NUM_CORES = os.cpu_count()

# Name of custom cgroup we will execute non-container processes in, so cAdvisor and Prometheus can track
# their metrics
CUSTOM_CGROUP_NAME = "custom"

# How long we measure the Docker daemon's metrics for, as a baseline, before the 
# container experiment is started
DAEMON_MEASUREMENT_TIME = 10

# How long we wait after cAdvisor & Prometheus are started before starting an experiment
CADVISOR_PROMETHEUS_WAIT_TIME = 20

## Prometheus queries
PROMETHEUS_QUERIES_LABELS = [None] + MEMORY_FIELD_NAMES + CPU_FIELD_NAMES

# The daemon's ID as expected by Prometheus
DAEMON_ID = "/system.slice/docker.service"

PROMETHEUS_PERF_AND_MEMORY_QUERIES = [
    "sum by (event) (container_perf_events_total{{id='{name_or_id}'}})",
    "avg_over_time(container_memory_usage_bytes{{id='{name_or_id}'}}[{container_duration_ms}ms] @ {end_container_timestamp:.2f})",
    "container_memory_max_usage_bytes{{id='{name_or_id}'}}",
    "100 * rate(container_cpu_usage_seconds_total{{id='{name_or_id}'}}[{container_duration_ms}ms] @ {end_container_timestamp:.2f})" + f" / {NUM_CORES}",
    "100 * rate(container_cpu_user_seconds_total{{id='{name_or_id}'}}[{container_duration_ms}ms] @ {end_container_timestamp:.2f})" + f" / {NUM_CORES}",
    "100 * rate(container_cpu_system_seconds_total{{id='{name_or_id}'}}[{container_duration_ms}ms] @ {end_container_timestamp:.2f})" + f" / {NUM_CORES}"
]

# When trying to measure a baseline for the Docker daemon's overhead, we need to get a time-independent measure for perf events, so 
# use rate, since we cannot guarantee that the measurement time for the baseline will be the same as the container
# execution time
PROMETHEUS_PERF_QUERIES_RATE = """sum by (event) (rate(container_perf_events_total{{id='{name_or_id}'}}[{container_duration_ms}ms] 
    @ {end_container_timestamp:.2f}))"""

# Similarly when measuring the total increase in the Docker daemon's overhead, we use increase since the daemon will have been running
# from before the experiment started, so it would contain values that do not directly correspond to the experiment's
PROMETHEUS_PERF_QUERIES_INCREASE = """sum by (event) (increase(container_perf_events_total{{id='{name_or_id}'}}[{container_duration_ms}ms]
    @ {end_container_timestamp:.2f}))"""

# Queries for the Docker daemon's overhead when measuring its baseline state
PROMETHEUS_PERF_AND_MEMORY_QUERIES_DAEMON_BASELINE = [PROMETHEUS_PERF_QUERIES_RATE.replace("{name_or_id}", DAEMON_ID)]
for query in PROMETHEUS_PERF_AND_MEMORY_QUERIES[1:]:
    PROMETHEUS_PERF_AND_MEMORY_QUERIES_DAEMON_BASELINE.append(query.replace("{name_or_id}", DAEMON_ID))

# Queries for the Docker daemon's overhead when measuring it during the container experiment
PROMETHEUS_PERF_AND_MEMORY_QUERIES_DAEMON_DURING_CONTAINER = [PROMETHEUS_PERF_QUERIES_INCREASE.replace("{name_or_id}", DAEMON_ID)]
for query in PROMETHEUS_PERF_AND_MEMORY_QUERIES[1:]:
    PROMETHEUS_PERF_AND_MEMORY_QUERIES_DAEMON_DURING_CONTAINER.append(query.replace("{name_or_id}", DAEMON_ID))

# The commands used to start a custom cgroup and execute a command in it, and to delete it
CREATE_CGROUP_CMD=f"sudo cgcreate -g cpuacct,memory,perf_event:{CUSTOM_CGROUP_NAME}"
EXEC_IN_CGROUP_CMD_PREFIX=f"sudo LD_LIBRARY_PATH={LD_LIBRARY_PATH} PATH={PATH} cgexec -g cpuacct,memory,perf_event:{CUSTOM_CGROUP_NAME}"
DELETE_CGROUP_CMD=f"sudo cgdelete -g cpuacct,memory,perf_event:{CUSTOM_CGROUP_NAME}"

# The number of times to retry an experiment before giving up
MAX_RETRIES = 15

# Variable tracking whether cAdvisor and Prometheus are currently running or not
cadvisor_and_prometheus_running = False

def is_cgroup_v2():
    """Checks if the system is using cgroup v2
    
    Returns:
        bool: True if the system is using cgroup v2, False otherwise
    """
    return os.path.isfile("/sys/fs/cgroup/cgroup.controllers")

def collect_time_data(n, results_filename, container_exec_cmd, container_start_cmd, wasm_interpreted_cmd, wasm_aot_cmd, native_cmd,
    deployment_mechanisms):
    """Runs the time experiments and collects the relevant data from the output, storing it in the specified file.

    Args:
        n: The number of trials to run for each deployment mechanism
        results_filename: The name of the file to store the results in
        container_exec_cmd: The command to execute the workload in the container, for the Docker mechanism
        container_start_cmd: The command to start the container, for the Docker mechanism
        wasm_interpreted_cmd: The command to run the workload for the WebAssembly interpreted mechanism
        wasm_aot_cmd: The command to run the workload for the WebAssembly AOT mechanism
        native_cmd: The command to run the workload as a standalone native binary, for the native mechanism
        deployment_mechanisms: The list of deployment mechanisms to use
    """
    # Randomly intersperse experiments of each type
    experiments = []
    if "docker" in deployment_mechanisms:
        experiments += ["docker"] * n
    if "wasm_interpreted" in deployment_mechanisms:
        experiments += ["wasm_interpreted"] * n
    if "wasm_aot" in deployment_mechanisms:
        experiments += ["wasm_aot"] * n
    if "native" in deployment_mechanisms:
        experiments += ["native"] * n
    random.shuffle(experiments)

    # Keep track of trial number for each deployment mechanism
    docker_trial = 1
    wasm_interpreted_trial = 1
    wasm_aot_trial = 1
    native_trial = 1

    metrics = []

    # Noting that experiments contains deployment mechanisms in the order they will be run
    for deployment_mechanism in experiments:
        print(f"Starting {deployment_mechanism} experiment")
        start_time = datetime.now(timezone.utc)
        trial_metrics = {}

        if deployment_mechanism == "docker":
            trial = docker_trial
            print(f"Trial {trial}")
            docker_trial += 1
            for attempt in range(MAX_RETRIES):
                try:
                    trial_metrics = run_time_experiment(container_start_cmd + " " + container_exec_cmd)
                    remove_container(CONTAINER_NAME)
                    trial_metrics_rows = prepare_trial_data_as_csv_rows(deployment_mechanism, trial, start_time, trial_metrics, TIME_METRICS_SHORT_NAMES)
                    metrics.extend(trial_metrics_rows)
                    break
                except Exception as e:
                    print(f"Error during docker trial {trial}, attempt {attempt + 1}: {e}")
                    remove_container(CONTAINER_NAME)
                    if attempt == MAX_RETRIES - 1:
                        break
        elif deployment_mechanism == "wasm_interpreted":
            trial = wasm_interpreted_trial
            print(f"Trial {trial}")
            wasm_interpreted_trial += 1
            for attempt in range(MAX_RETRIES):
                try:
                    trial_metrics = run_time_experiment(wasm_interpreted_cmd)
                    trial_metrics_rows = prepare_trial_data_as_csv_rows(deployment_mechanism, trial, start_time, trial_metrics, TIME_METRICS_SHORT_NAMES)
                    metrics.extend(trial_metrics_rows)
                    break
                except Exception as e:
                    print(f"Error during wasm_interpreted trial {trial}, attempt {attempt + 1}: {e}")
                    if attempt == MAX_RETRIES - 1:
                        break
        elif deployment_mechanism == "wasm_aot":
            trial = wasm_aot_trial
            print(f"Trial {trial}")
            wasm_aot_trial += 1
            for attempt in range(MAX_RETRIES):
                try:
                    trial_metrics = run_time_experiment(wasm_aot_cmd)
                    trial_metrics_rows = prepare_trial_data_as_csv_rows(deployment_mechanism, trial, start_time, trial_metrics, TIME_METRICS_SHORT_NAMES)
                    metrics.extend(trial_metrics_rows)
                    break
                except Exception as e:
                    print(f"Error during wasm_aot trial {trial}, attempt {attempt + 1}: {e}")
                    if attempt == MAX_RETRIES - 1:
                        break
        elif deployment_mechanism == "native":
            trial = native_trial
            print(f"Trial {trial}")
            native_trial += 1
            for attempt in range(MAX_RETRIES):
                try:
                    trial_metrics = run_time_experiment(native_cmd)
                    trial_metrics_rows = prepare_trial_data_as_csv_rows(deployment_mechanism, trial, start_time, trial_metrics, TIME_METRICS_SHORT_NAMES)
                    metrics.extend(trial_metrics_rows)
                    break
                except Exception as e:
                    print(f"Error during native trial {trial}, attempt {attempt + 1}: {e}")
                    if attempt == MAX_RETRIES - 1:
                        break
    
    # Write the results into a CSV
    field_names = CSV_BASIC_FIELD_NAMES + TIME_METRICS_SHORT_NAMES
    write_metrics_to_csv(results_filename, field_names, metrics)

def prepare_trial_data_as_csv_rows(deployment_mechanism, trial, start_time, trial_metrics_sets, metric_names, allow_missing_metrics=False):
    """Prepares the data of a trial, formatting it in a way allowing it to be written as a CSV row later.

    Args:
        deployment_mechanism: The deployment mechanism used for the trial
        trial: The trial number
        start_time: The start time of the trial
        trial_metrics_sets: A list of tuples in format ("special_identifier", trial_metrics_set)
            where trial_metrics_set is a list consisting of the trial metrics themselves. This format
            is used so we can store different types of metrics for the same experiment type, e.g. for
            container perf experiment we want to store one set of metrics for the container and another
            for the Docker overhead
        metric_names: The names of the metrics to include in the CSV row
        allow_missing_metrics: Whether to allow missing metrics or not
    Returns:
        A list of dictionaries, each dictionary representing a row in the CSV file
    """
    trial_metrics_rows = []

    for trial_metrics_set in trial_metrics_sets:
        identifier = trial_metrics_set[0]
        trial_metrics = trial_metrics_set[1]
    
        trial_metrics_row = {
            "deployment-mechanism": deployment_mechanism + identifier,
            "trial-number": trial,
            "start-time": start_time.isoformat(),
        }

        for metric_name in metric_names:
            try:
                trial_metrics_row[metric_name] = trial_metrics[metric_name]
            except KeyError:
                if not (metric_name in POSSIBLE_MISSING_METRICS and allow_missing_metrics):
                    raise

        trial_metrics_rows.append(trial_metrics_row)

    return trial_metrics_rows

def write_metrics_to_csv(results_filename, field_names, metrics):
    """Writes the metrics collected from the experiments to a CSV file.

    Args:
        results_filename: The name of the file to store the results in
        field_names: The names of the fields to include in the CSV file
        metrics: The metrics to write to the CSV file
    """
    with open(results_filename, "w", newline="") as csv_file:
        print(f"Writing results to {results_filename}")
        writer = csv.DictWriter(csv_file, fieldnames = field_names)
        writer.writeheader()
        writer.writerows(metrics)

def run_time_experiment(cmd):
    """Runs the time command on a given command and collects the time metrics from the output.
    
    Args:
        cmd: The command to run
    Returns:
        A list of tuples in format ("special_identifier", trial_metrics_set), where trial_metrics_set is a dictionary
            containing the trial metrics themselves. This format is used and expected by other functions so we can store different types 
            of metrics for the same experiment type, e.g. for container perf experiment we want to store one set of metrics for the 
            container and another for the Docker overhead. However, in this case, the Docker mechanism's execution time is the same 
            regardless of whether we consider the Docker overhead or not, so we will only have one set of metrics for each mechanism 
            and no special identifier differentiating them
    """
    stop_cadvisor_and_prometheus_if_running()
    cmd = TIME_CMD_PREFIX.split() + cmd.split()
    cmd_output, time_output = run_shell_cmd_and_get_stdout_and_stderr(cmd)

    time_metrics = parse_time_output(time_output)
    time_metrics.update(parse_time_output(cmd_output))
    time_metrics["overhead-time-seconds"] = round((time_metrics["wall-time-seconds"] - time_metrics["workload-time-seconds"]), 2)

    return [("", time_metrics)]

def parse_time_output(output):
    """Parses an output of the time experiments and collects the time metrics from it.

    Args:
        output: An output of the time experiments (which can be either from standard error, representing the time command's output,
         or from the workload being executed, which contains inference time statistics)
    Returns:
        A dictionary containing the relevant time metrics
    """
    metrics = {}
    for line in output.split("\n"):
        for metric in TIME_METRICS:
            metric_actual_name = metric[0]
            metric_new_name = metric[1]

            if metric_actual_name in line:
                words = line.strip().split()
                raw_value = words[-1]
                
                if metric_new_name == "wall-time-seconds":
                    time_parts = raw_value.split(":")
                    float_time_parts = [float(time_part) for time_part in time_parts]
                    if len(time_parts) == 2: # m:ss format
                        mins, seconds = float_time_parts
                        value = mins * 60 + seconds
                    elif len(time_parts) == 3: # h:mm:ss format
                        hours, mins, seconds = float_time_parts
                        value = hours * 3600 + mins * 60 + seconds
                # Output will already be in e.g. format 4.02 for 4.02 seconds, no further processing needed
                else:
                    value = float(raw_value)

                metrics[metric_new_name] = value

    return metrics

def collect_perf_data(n, results_filename, container_exec_cmd, container_start_cmd, wasm_interpreted_cmd, wasm_aot_cmd, native_cmd, 
    allow_missing_metrics, deployment_mechanisms):
    """Runs the performance experiments (measuring performance metrics besides time) and collects the relevant data from Prometheus, 
    storing it in the specified file.

    Args:
        n: The number of trials to run for each deployment mechanism
        results_filename: The name of the file to store the results in
        container_exec_cmd: The command to execute the workload in the container, for the Docker mechanism
        container_start_cmd: The command to start the container, for the Docker mechanism
        wasm_interpreted_cmd: The command to run the workload for the WebAssembly interpreted mechanism
        wasm_aot_cmd: The command to run the workload for the WebAssembly AOT mechanism
        native_cmd: The command to run the workload as a standalone native binary, for the native mechanism
        allow_missing_metrics: Whether to allow missing metrics or not
        deployment_mechanisms: The list of deployment mechanisms to use
    """

    # Randomly intersperse experiments of each type
    experiments = []
    if "docker" in deployment_mechanisms:
        experiments += ["docker"] * n
    if "wasm_interpreted" in deployment_mechanisms:
        experiments += ["wasm_interpreted"] * n
    if "wasm_aot" in deployment_mechanisms:
        experiments += ["wasm_aot"] * n
    if "native" in deployment_mechanisms:
        experiments += ["native"] * n
    random.shuffle(experiments)

    # Keep track of trial number for each deployment mechanism
    docker_trial = 1
    wasm_interpreted_trial = 1
    wasm_aot_trial = 1
    native_trial = 1

    metrics = []

    for experiment in experiments:
        print(f"Starting {experiment} experiment")
        start_time = datetime.now(timezone.utc)
        trial_metrics = {}
        
        if experiment == "docker":
            trial = docker_trial
            print(f"Trial {trial}")
            docker_trial += 1
            for attempt in range(MAX_RETRIES):
                try:
                    trial_metrics = run_container_perf_experiment(container_exec_cmd, container_start_cmd)
                    remove_container_and_its_prometheus_data(CONTAINER_NAME)
                    trial_metrics_row = prepare_trial_data_as_csv_rows(experiment, trial, start_time, trial_metrics, 
                        PERF_EVENTS + MEMORY_FIELD_NAMES + CPU_FIELD_NAMES, allow_missing_metrics)
                    metrics.extend(trial_metrics_row)
                    break
                except Exception as e:
                    print(f"Error during docker trial {trial}, attempt {attempt + 1}: {e}")
                    remove_container(CONTAINER_NAME)
                    if attempt == MAX_RETRIES - 1:
                        raise
        elif experiment == "wasm_interpreted":
            trial = wasm_interpreted_trial
            print(f"Trial {trial}")
            wasm_interpreted_trial += 1
            for attempt in range(MAX_RETRIES):
                try:
                    trial_metrics = run_non_container_perf_experiment(wasm_interpreted_cmd)
                    trial_metrics_row = prepare_trial_data_as_csv_rows(experiment, trial, start_time, trial_metrics, 
                        PERF_EVENTS + MEMORY_FIELD_NAMES + CPU_FIELD_NAMES, allow_missing_metrics)
                    metrics.extend(trial_metrics_row)
                    break
                except Exception as e:
                    print(f"Error during wasm_interpreted trial {trial}, attempt {attempt + 1}: {e}")
                    delete_prometheus_series_given_id(CUSTOM_CGROUP_NAME)
                    if attempt == MAX_RETRIES - 1:
                        raise
        elif experiment == "wasm_aot":
            trial = wasm_aot_trial
            print(f"Trial {trial}")
            wasm_aot_trial += 1
            for attempt in range(MAX_RETRIES):
                try:
                    trial_metrics = run_non_container_perf_experiment(wasm_aot_cmd)
                    trial_metrics_row = prepare_trial_data_as_csv_rows(experiment, trial, start_time, trial_metrics, 
                        PERF_EVENTS + MEMORY_FIELD_NAMES + CPU_FIELD_NAMES, allow_missing_metrics)
                    metrics.extend(trial_metrics_row)
                    break
                except Exception as e:
                    print(f"Error during wasm_aot trial {trial}, attempt {attempt + 1}: {e}")
                    delete_prometheus_series_given_id(CUSTOM_CGROUP_NAME)
                    if attempt == MAX_RETRIES - 1:
                        raise
        elif experiment == "native":
            trial = native_trial
            print(f"Trial {trial}")
            native_trial += 1
            for attempt in range(MAX_RETRIES):
                try:
                    trial_metrics = run_non_container_perf_experiment(native_cmd)
                    trial_metrics_row = prepare_trial_data_as_csv_rows(experiment, trial, start_time, trial_metrics, 
                        PERF_EVENTS + MEMORY_FIELD_NAMES + CPU_FIELD_NAMES, allow_missing_metrics)
                    metrics.extend(trial_metrics_row)
                    break
                except Exception as e:
                    print(f"Error during native trial {trial}, attempt {attempt + 1}: {e}")
                    delete_prometheus_series_given_id(CUSTOM_CGROUP_NAME)
                    if attempt == MAX_RETRIES - 1:
                        raise
    
    # Write the results into a CSV
    field_names = CSV_BASIC_FIELD_NAMES + PERF_EVENTS + MEMORY_FIELD_NAMES + CPU_FIELD_NAMES
    write_metrics_to_csv(results_filename, field_names, metrics)

    stop_cadvisor_and_prometheus_if_running()

def start_cadvisor_and_prometheus():
    """Starts cAdvisor and Prometheus in the background."""
    start_cadvisor()
    start_prometheus()
    global cadvisor_and_prometheus_running
    cadvisor_and_prometheus_running = True

def start_prometheus():
    """Starts Prometheus in the background."""
    run_shell_cmd_in_background(PROMETHEUS_START_CMD.split())

def start_cadvisor():
    """Starts cAdvisor in the background."""
    run_shell_cmd_in_background(CADVISOR_START_CMD.split())

def start_cadvisor_and_prometheus_if_not_running():
    """Starts cAdvisor and Prometheus if they are not already running."""
    global cadvisor_and_prometheus_running
    if not cadvisor_and_prometheus_running:
        start_cadvisor_and_prometheus()

    # Give cAdvisor and Prometheus time to start up
    time.sleep(CADVISOR_PROMETHEUS_WAIT_TIME)

def stop_cadvisor_and_prometheus():
    """Stops cAdvisor and Prometheus."""
    stop_cadvisor()
    stop_prometheus()   
    global cadvisor_and_prometheus_running
    cadvisor_and_prometheus_running = False

def stop_prometheus():
    """Stops Prometheus."""
    run_shell_cmd(PROMETHEUS_STOP_CMD.split())

def stop_cadvisor():
    """Stops cAdvisor."""
    run_shell_cmd(CADVISOR_STOP_CMD.split())

def stop_cadvisor_and_prometheus_if_running():
    """Stops cAdvisor and Prometheus if they are running."""
    global cadvisor_and_prometheus_running
    if cadvisor_and_prometheus_running:
        stop_cadvisor_and_prometheus()

def run_non_container_perf_experiment(cmd): 
    """Run a performance experiment for a non-container deployment mechanism, such as WebAssembly or native, 
    and collect the relevant data from Prometheus.

    Args:
        cmd: The command to run for the experiment
    Returns:
        A list of tuples in format ("special_identifier", trial_metrics_set), where trial_metrics_set is a dictionary
            containing the trial metrics themselves. This format is used and expected by other functions so we can store different types 
            of metrics for the same experiment type, e.g. for container perf experiment we want to store one set of metrics for the 
            container and another for the Docker overhead. However, in this case, non-container mechanisms
            don't have different types of metrics, so we will only have one set of metrics for each mechanism
    """
    start_cadvisor_and_prometheus_if_not_running()

    # Create the cgroup that the process will be assigned to
    run_shell_cmd(CREATE_CGROUP_CMD.split())

    start_time = datetime.now(timezone.utc)
    start_timestamp = start_time.timestamp()

    run_in_cgroup_cmd = EXEC_IN_CGROUP_CMD_PREFIX.split() + cmd.split()
    run_shell_cmd(run_in_cgroup_cmd)

    end_time = datetime.now(timezone.utc)
    end_timestamp = end_time.timestamp()

    execution_duration_ms = round((end_timestamp - start_timestamp) * 1000)

    metrics = {}

    for query, label in zip(PROMETHEUS_PERF_AND_MEMORY_QUERIES, PROMETHEUS_QUERIES_LABELS):
        formatted_query = query.format(name_or_id=f"/{CUSTOM_CGROUP_NAME}", 
            container_duration_ms=execution_duration_ms, end_container_timestamp=end_timestamp)
        metrics.update(get_parsed_prometheus_query_results(formatted_query, label))

    delete_prometheus_series_given_id(CUSTOM_CGROUP_NAME)

    return [("", metrics)]

def run_container_perf_experiment(container_exec_cmd, container_start_cmd):
    """Run a performance experiment for the Docker deployment mechanism,
    and collect the relevant data from Prometheus.

    Args:
        container_exec_cmd: The command to execute the workload in the container
        container_start_cmd: The command to start the container
    Returns:
        A list of tuples in format ("special_identifier", trial_metrics_set), where trial_metrics_set is a dictionary
            containing the trial metrics themselves. This format is used and expected by other functions so we can store different types 
            of metrics for the same experiment type, e.g. for container perf experiment we want to store one set of metrics for the 
            container and another for the Docker overhead.
    """
    start_cadvisor_and_prometheus_if_not_running()

    # Clear the daemon's cgroup first, so maximum memory usage is not affected by memory
    # usage that occured before the experiment
    cleanup_daemon_cgroup()

    # Get the daemon's baseline metrics
    daemon_metrics_baseline = {}
    time.sleep(DAEMON_MEASUREMENT_TIME)

    curr_time = datetime.now(timezone.utc)
    curr_timestamp = curr_time.timestamp()

    for query, label in zip(PROMETHEUS_PERF_AND_MEMORY_QUERIES_DAEMON_BASELINE, PROMETHEUS_QUERIES_LABELS):
        formatted_query = query.format(container_duration_ms=DAEMON_MEASUREMENT_TIME * 1000, 
            end_container_timestamp=curr_timestamp)
        daemon_metrics_baseline.update(get_parsed_prometheus_query_results(formatted_query, label))

    # Run the container and time the execution
    start_container_time = datetime.now(timezone.utc)
    start_container_timestamp = start_container_time.timestamp()

    container_cmd = container_start_cmd.split() + container_exec_cmd.split()
    run_shell_cmd(container_cmd)

    end_container_time = datetime.now(timezone.utc)
    end_container_timestamp = end_container_time.timestamp()

    container_duration_ms = round((end_container_timestamp - start_container_timestamp) * 1000)
    
    # Get the container's metrics during the execution time
    container_cgroup_id = get_cgroup_id_for_container(CONTAINER_NAME)
    container_metrics = {}

    for query, label in zip(PROMETHEUS_PERF_AND_MEMORY_QUERIES, PROMETHEUS_QUERIES_LABELS):
        formatted_query = query.format(name_or_id=container_cgroup_id, container_duration_ms=container_duration_ms, 
            end_container_timestamp=end_container_timestamp)
        container_metrics.update(get_parsed_prometheus_query_results(formatted_query, label))

    # Get the daemon's metrics during that same time
    daemon_metrics_during_container = {}

    for query, label in zip(PROMETHEUS_PERF_AND_MEMORY_QUERIES_DAEMON_DURING_CONTAINER, PROMETHEUS_QUERIES_LABELS):
        formatted_query = query.format(container_duration_ms=container_duration_ms, 
            end_container_timestamp=end_container_timestamp)
        daemon_metrics_during_container.update(get_parsed_prometheus_query_results(formatted_query, label))

    # Sum the metrics for the container and the daemon during the container's execution
    container_and_daemon_metrics = {key: container_metrics[key] + daemon_metrics_during_container.get(key, 0) 
        for key in container_metrics}

    # Multiply the perf events part of the daemon's baseline metrics by the container's execution time in
    # seconds, since the former was obtained using rate
    for perf_event in PERF_EVENTS:
        if perf_event in daemon_metrics_baseline:
            daemon_metrics_baseline[perf_event] = round(daemon_metrics_baseline[perf_event] * (container_duration_ms / 1000))

    # Subtract the daemon's baseline metrics from the daemon's metrics during the container's execution
    daemon_extra_overhead_metrics = {key: max(0, daemon_metrics_during_container[key] - daemon_metrics_baseline.get(key, 0))
        for key in daemon_metrics_during_container}

    # Sum the metrics for the container and only the extra overhead for the daemon during the container's execution
    container_and_daemon_extra_overhead_metrics = {key: container_metrics[key] + daemon_extra_overhead_metrics.get(key, 0)
        for key in container_metrics}

    return [("_container", container_metrics), ("_container_and_daemon", container_and_daemon_metrics),
        ("_container_and_daemon_extra_overhead", container_and_daemon_extra_overhead_metrics)]

def stop_container(container_name):
    """Stops a container with the given name.

    Args:
        container_name: The name of the container to stop
    """
    cmd = CONTAINER_STOP_CMD.format(container_name=container_name).split()
    run_shell_cmd(cmd)

def remove_container(container_name):
    """Removes a container with the given name.

    Args:
        container_name: The name of the container to remove
    """
    cmd = CONTAINER_REMOVE_CMD.format(container_name=container_name).split()

    try:
        run_shell_cmd(cmd)
    except subprocess.CalledProcessError as e:
        # The following error happens if the container was not successfully
        # started in the first place; this does not necessarily indicate a 
        # major failure so we can ignore it
        if "No such container" not in e.stderr:
            raise

def remove_container_and_its_prometheus_data(container_name):
    """Removes a container with the given name, and deletes its Prometheus data.

    Args:
        container_name: The name of the container to remove
    """
    remove_container(container_name)
    delete_prometheus_series_given_name(container_name)

def get_cgroup_id_for_container(container_name):
    """Gets the cgroup ID for a container with the given name.

    Args:
        container_name: The name of the container
    Returns:
        The cgroup ID for the container
    """
    cmd = CONTAINER_INSPECT_ID_CMD.format(container_name=container_name).split()
    # We first strip whitespace from the command, then strip the single quotes from the output
    # Otherwise the last single quote will not be caught
    container_id = run_shell_cmd_and_get_stdout(cmd).strip().strip("'")

    return f"/system.slice/docker-{container_id}.scope"

def run_shell_cmd_and_get_stdout_and_stderr(cmd):
    """Runs a shell command and returns its output and stderr.

    Args:
        cmd: The command to run
    Returns:
        The output of the command
    """
    try:
        result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        raise


def run_shell_cmd_and_get_stdout(cmd):
    """Runs a shell command and returns its output.

    Args:
        cmd: The command to run
    Returns:
        The output of the command
    """
    try:
        result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        raise

def run_shell_cmd_and_get_stderr(cmd):
    """Runs a shell command and returns its output to stderr.

    Args:
        cmd: The command to run
    Returns:
        The output of the command
    """
    # Use stdout=subprocess.DEVNULL to suppress the output to stdout
    try:
        result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        raise

def run_shell_cmd(cmd):
    """Runs a shell command.

    Args:
        cmd: The command to run
    """
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        raise

def run_shell_cmd_in_background(cmd):
    """Runs a shell command in the background.

    Args:
        cmd: The command to run
    """
    try:
        # Use stdout=subprocess.DEVNULL and stderr=subprocess.DEVNULL to suppress the output
        # to stdout and stderr
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        raise

def query_prometheus(query):
    """Queries Prometheus with a given query string.

    Args:
        query: The query string 
    Returns:
        The result of the query
    """
    params = {"query": query}
    return query_prometheus_with_params(params)

def query_prometheus_with_params(params):
    """Queries Prometheus with given parameters.
    Args:
        params: The parameters to include in the query
    Returns:
        The result of the query
    """
    response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", 
        params)
    data = response.json()
    if data["status"] != "success":
        raise Exception("Error: Prometheus query failed")
    return data["data"]["result"]

def delete_prometheus_series_given_name(name):
    """Deletes Prometheus data for series with a given name.
    
    Args:
        name: The name of the series to delete
    """
    match = f"{{name='{name}'}}"
    delete_prometheus_series(match)

def cleanup_custom_cgroup():
    """Cleans up the custom cgroup created for non-Docker experiments."""
    if cgroup_exists(CUSTOM_CGROUP_NAME):
        run_shell_cmd(DELETE_CGROUP_CMD.split())
    delete_prometheus_series_given_id(CUSTOM_CGROUP_NAME)

def cgroup_exists(cgroup_name):
    """Checks if a cgroup with the given name exists.

    Args:
        cgroup_name: The name of the cgroup to check
    Returns:
        True if the cgroup exists, False otherwise
    """
    # For cgroup v1, check in /sys/fs/cgroup/memory/cgroup_name
    path_v1 = f"/sys/fs/cgroup/memory/{cgroup_name}"
    # For cgroup v2, typically the unified hierarchy is mounted at /sys/fs/cgroup
    path_v2 = f"/sys/fs/cgroup/{cgroup_name}"
    
    return os.path.exists(path_v1) or os.path.exists(path_v2)

def cleanup_daemon_cgroup():
    """Cleans up the Docker daemon's cgroup.""" 
    
    delete_prometheus_series_given_id(DAEMON_ID)

def delete_prometheus_series_given_id(id):
    """Deletes Prometheus data for series with a given ID.
    
    Args:
        id: The ID of the series to delete
    """
    match = f"{{id='{id}'}}"
    delete_prometheus_series(match)

def delete_prometheus_series(match):
    """Deletes Prometheus data for series matching a given match string.

    Args:
        match: The match string 
    """
    params = {"match[]": match}
    response = requests.post(f"{PROMETHEUS_URL}/api/v1/admin/tsdb/delete_series", params=params)
    if response.status_code != 204:
        raise Exception("Error: Prometheus series deletion failed")

def get_parsed_prometheus_query_results(query, label=None):
    """Queries Prometheus with a given query string and parses the output.

    Args:
        query: The query string 
        label: The label to use for the metric being queried (only if the query string
            is for a single metric)
    Returns:
        A dictionary containing the parsed metrics
    """
    data = query_prometheus(query)
    return parse_prometheus_output(data, label)

def parse_prometheus_output(output, label=None):
    """Parses the output of a Prometheus query and collects the metrics from it.

    Args:
        output: The output of the Prometheus query
        label: The label to use for the metric being queried (only if the query string
            is for a single metric)
    Returns:
        A dictionary containing the parsed metrics
    """
    metrics = {}
    for entry in output:
        metric = entry["metric"]
        if label is not None:
            key = label
        elif "event" in metric:
            key = metric["event"]
        elif "__name__" in metric:
            key = metric["__name__"]
        else:
            key = "unknown_metric"

        value = round(float(entry["value"][1]), 2)
        metrics[key] = value
    return metrics

def main():
    # Parse the command line arguments to determine which model and input to use
    parser = argparse.ArgumentParser(description="Benchmark the performance of different edge ML deployment mechanisms")
    parser.add_argument("--model", type=str, required=True, help="The ML model to use")
    parser.add_argument("--input", type=str, required=True, help="The input file to run ML inference on")
    parser.add_argument("--trials", type=int, required=True, help="The number of trials to run for each experiment type")
    parser.add_argument("--mechanisms", type=str, default="docker,wasm_interpreted,wasm_aot,native",
                        help="Comma-separated list of mechanisms to include (choose from docker, wasm_interpreted, wasm_aot, native)")
    parser.add_argument("--arch", type=str, required=True, help="The architecture of the target device this is being run on")
    parser.add_argument("--set_name", type=str, required=True, help="The name of the set of experiments being run")
    parser.add_argument("--allow_missing_metrics", action="store_true", help="Allow missing events in the results")
    parser.add_argument("--is_mac", action="store_true", help="Set to true if running on MacOS as the underlying hardware")

    args = parser.parse_args()
    model = args.model
    input_file = args.input
    trials = args.trials
    mechanisms = set(m.strip().lower() for m in args.mechanisms.split(","))
    arch = args.arch
    set_name = args.set_name
    allow_missing_metrics = args.allow_missing_metrics

    # Path to the model and input
    model_path = f"models/{model}"
    input_path = f"inputs/{input_file}"

    # The name of the Docker image to use
    img_name = IMG_NAME_TEMPLATE.format(arch=arch)

    # The command to execute the workload inside the container
    container_start_cmd = CONTAINER_START_CMD_TEMPLATE.format(img_name=img_name)
    container_exec_cmd = f"./{NATIVE_BINARY_NAME} /{model_path} /{input_path}"
    
    # For Macs, the AoT Wasm file must have the .so extension
    if args.is_mac:
        aot_wasm_file_path = AOT_WASM_FILE_PATH_TEMPLATE.format(extension="so")
    else:
        aot_wasm_file_path = AOT_WASM_FILE_PATH_TEMPLATE.format(extension="wasm")

    # The commands to execute for the WebAssembly deployment mechanisms
    wasm_interpreted_cmd =f"{WASM_BINARY_PATH} --dir .:. {INTERPRETED_WASM_FILE_PATH} {model_path} {input_path}"
    wasm_aot_cmd = f"{WASM_BINARY_PATH} --dir .:. {aot_wasm_file_path} {model_path} {input_path}"

    # The command to execute for the native deployment mechanism
    native_cmd = f"{NATIVE_BINARY_PATH} {model_path} {input_path}"

    # The name of the file to store the results in
    results_filename_prefix = f"{model}&{input_file}"
    results_filename_prefix_with_path = os.path.join(RESULTS_DIR, set_name, results_filename_prefix)

    try:
        collect_perf_data(trials, results_filename_prefix_with_path + PERF_RESULTS_FILENAME_SUFFIX, container_exec_cmd, container_start_cmd, wasm_interpreted_cmd, wasm_aot_cmd, native_cmd, allow_missing_metrics, mechanisms)
        collect_time_data(trials, results_filename_prefix_with_path + TIME_RESULTS_FILENAME_SUFFIX, container_exec_cmd, container_start_cmd, wasm_interpreted_cmd, wasm_aot_cmd, native_cmd, mechanisms)
    finally:
        stop_cadvisor_and_prometheus_if_running()
        cleanup_custom_cgroup()
        

if __name__ == "__main__":
    main()