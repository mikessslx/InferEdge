"""This script analyzes the experimental data, determining if there is a statistically significant difference
   in the performance of different deployment mechanisms and quantifying the extent of that difference.
"""
import pandas as pd
import statsmodels.stats.weightstats as smw
from itertools import combinations
import matplotlib.pyplot as plt
import argparse
import os
import csv

# The names of columns that are not metrics and must hence always be included in the dataframes
NON_METRIC_COLUMNS = ["index", "deployment-mechanism", "trial-number"]

# The names of extra columns computed from values in the result files 
COMPUTED_COLUMNS = ["instructions-per-cycle", "cycles-per-instruction", 
    "model-load-time-(seconds)", "input-ready-time-(seconds)", "inference-time-(seconds)", 
    "input-resize-time-(seconds)", "input-load-time-(seconds)"]

# The absolute path of the "data_scripts" directory where this script is in
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))

# The absolute path of the parent directory, which is the root of the benchmark suite
BENCHMARK_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))

# The absolute path of the "results" directory where the results of the experiments are stored
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")

# The name of the CSV file where the aggregate results from all the experiments within
# an experiment set are stored 
AGGREGATE_CSV_FILENAME = "aggregate_results.csv"

# Numbers representing the different views of the Docker overhead
DOCKER_OVERHEAD_EXCLUDE_DAEMON = 0
DOCKER_OVERHEAD_INCLUDE_FULL_DAEMON = 1
DOCKER_OVERHEAD_INCLUDE_ADDITIONAL_DAEMON = 2

def welch_t_test_with_confidence_interval(arr_x, arr_y, alpha=0.05):
    """Perform Welch's t-test on two samples and calculate the confidence interval of the difference of the means.

    Args:
        arr_x: First sample.
        arr_y: Second sample.
        alpha: Significance level for the confidence interval.
    Returns:
        tuple: Mean of arr_x, mean of arr_y, mean difference, the confidence interval's lower bound,
               the confidence interval's upper bound, the half-width of the confidence interval,
               whether the difference is statistically significant, and the confidence intervals for arr_x and arr_y.
    """
    # Calculate the mean of the data and compare them
    descr_stats_x = smw.DescrStatsW(arr_x)
    descr_stats_y = smw.DescrStatsW(arr_y)
    compare_means = smw.CompareMeans(descr_stats_x, descr_stats_y)

    # Calculate the confidence interval of the difference of the means;
    # use Welch's t-test which does not assume equal variances
    # between the samples represented by arr_x and arr_y
    ci_lower, ci_upper = compare_means.tconfint_diff(usevar="unequal", alpha=alpha)
    
    # Get the difference of the means
    x_mean = descr_stats_x.mean
    y_mean = descr_stats_y.mean
    mean_diff = abs(y_mean - x_mean)

    # Get the half-width of the confidence interval
    ci_half_width = (ci_upper - ci_lower) / 2

    # Determine statistical significance by checking if the confidence interval
    # contains zero (no difference between the means)
    statistically_significant = not (ci_lower <= 0 <= ci_upper)

    # Get individual confidence intervals for the means
    x_ci = descr_stats_x.tconfint_mean(alpha=alpha)
    y_ci = descr_stats_y.tconfint_mean(alpha=alpha)

    return x_mean, y_mean, mean_diff, ci_lower, ci_upper, ci_half_width, statistically_significant, x_ci, y_ci

def initialize_aggregate_df(metric_cols, deployment_mechanisms, model, input):
    """Initialize the aggregate dataframe storing aggregate results for each deployment mechanism.

    Args:
        metric_cols: List of column names corresponding to the metrics.
        deployment_mechanisms: List of deployment mechanisms.
        model: The name of the model used in the experiments.
        input: The name of the input used in the experiments.
    Returns:
        pd.DataFrame: The initialized aggregate dataframe.
    """
    # We include the model and input in the aggregate dataframe since we will later add
    # the data to a CSV file aggregating results from all experiments within an experiment set
    aggregate_df = pd.DataFrame(columns=["model", "input", "deployment-mechanism"])

    for metric in metric_cols:
        # For each metric, add three columns to the aggregate dataframe: the metric's mean, its lower error bound,
        # and its upper error bound
        aggregate_df[f"{metric}-mean"] = None
        aggregate_df[f"{metric}-error-lower"] = None
        aggregate_df[f"{metric}-error-upper"] = None

    for deployment_mechanism in deployment_mechanisms:
        # For each deployment mechanism, add a row to the aggregate dataframe;
        # initially, only deployment mechanism, model, and inputs have a value
        row = {col: None for col in aggregate_df.columns}
        row["model"] = model
        row["input"] = input
        row["deployment-mechanism"] = deployment_mechanism

        aggregate_df.loc[len(aggregate_df)] = row

    return aggregate_df

def analyze_data_significant_difference(df, significance_level, metrics, model, input, analyzed_results_path, 
    include_insignificant_output, view_output, save_output):
    """Analyze the data to determine if there are statistically significant differences between deployment mechanisms.

    Args:
        df: The dataframe containing the experimental data.
        significance_level: The significance level for statistical tests.
        metrics: List of metrics to analyze.
        model: The name of the model used in the experiments.
        input: The name of the input used in the experiments.
        analyzed_results_path: Path to save analyzed results.
        include_insignificant_output: Whether to include output for insignificant results.
        view_output: Whether to view the output of the analysis.
        save_output: Whether to save the output of the analysis to files.
    Returns:
        pd.DataFrame: An aggregate dataframe containing aggregate results for each deployment mechanism.
    """
    # For each deployment mechanism, group the results for each metric
    grouped_df = df.groupby("deployment-mechanism")[metrics]

    # For each pair of experiment types and each metric, test for statistically
    # significant differences and calculate the effect size confidence intervals
    deployment_mechanisms = df["deployment-mechanism"].unique()

    # This new dataframe will save, for each deployment mechanism, its statistics for each metric, for further analysis
    # in other functions e.g. visualizations
    aggregate_df = initialize_aggregate_df(metrics, deployment_mechanisms, model, input)

    for deployment_mechanism_x, deployment_mechanism_y in combinations(deployment_mechanisms, 2):
        # This new dataframe will save, for this specific comparison, the two mechanisms' values for
        # each metric, whether the difference is statistically significant for each, and the effect size
        # confidence intervals
        comparison_df = pd.DataFrame(columns=["metric", f"{deployment_mechanism_x}-value", f"{deployment_mechanism_y}-value",
            "statistically-significant", "effect-size"])

        for metric in metrics:
            arr_x = grouped_df.get_group(deployment_mechanism_x)[metric]
            arr_y = grouped_df.get_group(deployment_mechanism_y)[metric]

            x_mean, y_mean, mean_diff, ci_lower, ci_upper, ci_half_width, statistically_significant, x_ci, y_ci = \
                welch_t_test_with_confidence_interval(arr_x, arr_y, alpha=significance_level)

            # Calculate the ratio of the means and its confidence interval
            if x_mean < y_mean:
                ratio = y_mean / x_mean
                ratio_ci = ci_half_width / x_mean
                ratio_min = ratio - ratio_ci
                ratio_max = ratio + ratio_ci
                ratio_message = f"{deployment_mechanism_x} is {ratio_min:.2f} to {ratio_max:.2f} times larger than {deployment_mechanism_y} for {metric}"
            else:
                ratio = x_mean / y_mean
                ratio_ci = ci_half_width / y_mean
                ratio_min = ratio - ratio_ci
                ratio_max = ratio + ratio_ci
                ratio_message = f"{deployment_mechanism_y} is {ratio_min:.2f} to {ratio_max:.2f} times larger than {deployment_mechanism_x} for {metric}"

            # Update the relevant values of the appropriate row in the aggregate dataframe
            # In the future, this can be made more efficient as these values are being calculated
            # multiple times currently (e.g. when comparing deployment mechanism x with deployment mechanism y,
            # then comparing deployment mechanism y with deployment mechanism z, y's values are recalculated)
            aggregate_df.loc[aggregate_df["deployment-mechanism"] == deployment_mechanism_x, [f"{metric}-mean", f"{metric}-error-lower", f"{metric}-error-upper"]] = \
                [x_mean, x_mean - x_ci[0], x_ci[1] - x_mean]
            aggregate_df.loc[aggregate_df["deployment-mechanism"] == deployment_mechanism_y, [f"{metric}-mean", f"{metric}-error-lower", f"{metric}-error-upper"]] = \
                [y_mean, y_mean - y_ci[0], y_ci[1] - y_mean]
            
            # Add a new row to the comparison dataframe for this metric
            comparison_df.loc[len(comparison_df)] = {
                "metric": metric,
                f"{deployment_mechanism_x}-value": f"{x_ci[0]:,.2f}-{x_ci[1]:,.2f}",
                f"{deployment_mechanism_y}-value": f"{y_ci[0]:,.2f}-{y_ci[1]:,.2f}",
                "statistically-significant": statistically_significant,
                "effect-size": f"{ratio_min:.2f}x-{ratio_max:.2f}x"
            }

            if statistically_significant:
                # Reporting of results and calculations for ratio based on those used by the Sightglass benchmark,
                # available at https://github.com/bytecodealliance/sightglass/blob/main/crates/analysis/src/effect_size.rs
                # (accessed: 27 Jan. 2025)
                print_if_true(f"Statistically significant difference between {deployment_mechanism_x} and {deployment_mechanism_y} for {metric}", view_output)
            else:
                print_if_true(f"No statistically significant difference between {deployment_mechanism_x} and {deployment_mechanism_y} for {metric}", view_output)
                if not include_insignificant_output:
                    continue

            print_if_true(f"Mean difference: {mean_diff:.2f} ± {ci_half_width:.2f} with confidence level {(1 - significance_level) * 100.0}%", view_output)
            print_if_true(f"{deployment_mechanism_x} average: {x_mean:.2f} (95% CI: {x_ci[0]:.2f} to {x_ci[1]:.2f})", view_output)
            print_if_true(f"{deployment_mechanism_y} average: {y_mean:.2f} (95% CI: {y_ci[0]:.2f} to {y_ci[1]:.2f})", view_output)
            print_if_true(ratio_message, view_output)
            print_if_true("", view_output)

        if save_output:
            # Save the comparison dataframe to a CSV file
            comparison_csv_filename = f"{model}-{input}-{deployment_mechanism_x}-{deployment_mechanism_y}-comparison.csv"
            comparison_csv_path = os.path.join(analyzed_results_path, comparison_csv_filename)

            # Enclose everything in quotes, since otherwise importing them into e.g. Excel might
            # not work properly
            comparison_df.to_csv(comparison_csv_path, index=False, quoting=csv.QUOTE_ALL)
        
    return aggregate_df

def add_thousand_separator(number):
    """Add a thousand separator to a number.

    Args:
        number: The number to format.
    Returns:
        str: The formatted number with a thousand separator.
    """
    return f"{number:,}"

def print_if_true(message, condition):
    """Print a message if the condition is true.
    
    Args:
        message: The message to print.
        condition: The condition to check.
    """
    if condition:
        print(message)

def parse_csv_rows(results_filename, deployment_mechanisms, metrics, docker_overhead_view, is_perf_file=True):
    """Parse the CSV file containing the results of the experiments.

    Args:
        results_filename: The path to the CSV file.
        deployment_mechanisms: List of deployment mechanisms to include in subsequent analyses.
        metrics: List of metrics to include in subsequent analyses.
        docker_overhead_view: The view of the Docker overhead to use.
        is_perf_file: Whether the CSV file contains performance data (besides time data) or time data.
    Returns:
        pd.DataFrame: The parsed dataframe.
    """

    try:
        df = pd.read_csv(results_filename)

        # Drop columns corresponding to metrics that were not specified
        df = df.drop(df.columns.difference(NON_METRIC_COLUMNS + metrics), axis=1)

        if docker_overhead_view == DOCKER_OVERHEAD_EXCLUDE_DAEMON:
            # Rename "docker_container" as an deployment mechanism to just "docker"
            df["deployment-mechanism"] = df["deployment-mechanism"].apply(lambda deployment_mechanism: 
                "docker" if deployment_mechanism == "docker_container" else deployment_mechanism)

            # Remove the other rows whose deployment mechanism starts with "docker_container_and_daemon"
            df = df[~df["deployment-mechanism"].str.startswith("docker_container_and_daemon")]
        elif docker_overhead_view == DOCKER_OVERHEAD_INCLUDE_FULL_DAEMON:
            # Rename "docker_container_and_daemon" as an deployment mechanism to just "docker"
            df["deployment-mechanism"] = df["deployment-mechanism"].apply(lambda deployment_mechanism: 
                "docker" if deployment_mechanism.startswith("docker_container_and_daemon") else deployment_mechanism)

            # Remove the rows whose deployment mechanism is "docker_container" and "docker_container_and_daemon_extra_overhead"
            df = df[~df["deployment-mechanism"].str.startswith("docker_container")]
            df = df[~df["deployment-mechanism"].str.startswith("docker_container_and_daemon_extra_overhead")]
        elif docker_overhead_view == DOCKER_OVERHEAD_INCLUDE_ADDITIONAL_DAEMON:
            # Rename "docker_container_and_daemon_extra_overhead" as an deployment mechanism to just "docker"
            df["deployment-mechanism"] = df["deployment-mechanism"].apply(lambda deployment_mechanism: 
                "docker" if deployment_mechanism.startswith("docker_container_and_daemon_extra_overhead") else deployment_mechanism)

            # Remove the rows whose deployment mechanism is "docker_container" and "docker_container_and_daemon"
            df = df[~df["deployment-mechanism"].str.startswith("docker_container")]
            df = df[~df["deployment-mechanism"].str.startswith("docker_container_and_daemon")]   

        if is_perf_file:
            # Check if the "cpu-cycles" and "instructions" columns are present
            if "cpu-cycles" in df.columns and "instructions" in df.columns:
                # Add new columns for instructions-per-cycle and cycles-per-instruction
                df["instructions-per-cycle"] = df["instructions"] / df["cpu-cycles"]
                df["cycles-per-instruction"] = df["cpu-cycles"] / df["instructions"]
        else:
            model_loaded_col = "until-model-loaded-time-(seconds)" if "until-model-loaded-time-(seconds)" in df.columns else "until-model-loaded-time-seconds"
            input_loaded_col = "until-input-loaded-time-(seconds)" if "until-input-loaded-time-(seconds)" in df.columns else "until-input-loaded-time-seconds"
            input_resized_col = "until-input-resized-time-(seconds)" if "until-input-resized-time-(seconds)" in df.columns else "until-input-resized-time-seconds"
            input_ready_col = "until-input-ready-time-(seconds)" if "until-input-ready-time-(seconds)" in df.columns else "until-input-ready-time-seconds"
            inference_executed_col = "until-inference-executed-time-(seconds)" if "until-inference-executed-time-(seconds)" in df.columns else "until-inference-executed-time-seconds"
            df["overhead-time-seconds"] = df["wall-time-seconds"] - df["workload-time-seconds"]
            df["model-load-time-(seconds)"] = df[model_loaded_col]
            df["input-preparation-time-(seconds)"] = df[input_ready_col] - df[model_loaded_col]
            df["inference-time-(seconds)"] = df[inference_executed_col] - df[input_ready_col]
            df["input-resize-time-(seconds)"] = df[input_resized_col] - df[input_loaded_col]
            df["input-load-time-(seconds)"] = df[input_loaded_col] - df[model_loaded_col]

        # Drop rows corresponding to deployment mechanisms that were not specified
        df = df.drop(df[~df["deployment-mechanism"].isin(deployment_mechanisms)].index)
    except FileNotFoundError:
        print(f"Warning: The file {results_filename} does not exist. Continuing without its data.")
        df = pd.DataFrame(columns=NON_METRIC_COLUMNS + metrics)

    return df

def get_metrics_in_df(df):
    """Get the metrics present in the dataframe.

    Args:
        df: The dataframe containing the experimental data.
    Returns:
        list: List of metrics present in the dataframe.
    """
    return [col for col in df.columns if col not in NON_METRIC_COLUMNS]

def plot_metrics_bar_chart(aggregate_df, metrics, view_output, save_output, plots_path, model, input):
    """Plot the deployment mechanisms' aggregate results for each metric.

    Args:
        aggregate_df: The aggregate dataframe containing the results.
        metrics: List of metrics to plot.
        view_output: Whether to view the plots.
        save_output: Whether to save the plots to files.
        plots_path: Path to save the plots.
        model: The name of the model used in the experiments.
        input: The name of the input used in the experiments.
    """
    deployment_mechanisms = aggregate_df["deployment-mechanism"].unique().tolist()

    # For each metric, plot the mean and confidence interval for each deployment mechanism
    for metric in metrics:
        metric_name_without_hyphen = metric.replace("-", " ")
        metric_with_underscores = metric.replace("-", "_")
        plt.figure(metric)

        # Plot the mean and confidence interval for each deployment mechanism
        means = aggregate_df[f"{metric}-mean"].tolist()
        errors = [aggregate_df[f"{metric}-error-lower"].tolist(), aggregate_df[f"{metric}-error-upper"].tolist()]
        plt.bar(deployment_mechanisms, means, yerr=errors, capsize=5)

        # Set title and labels
        plt.title(f"{metric_name_without_hyphen} by deployment mechanism\nfor model {model} and input {input}") 
        plt.ylabel(metric_name_without_hyphen)
        plt.xlabel("deployment mechanism")

        if save_output:
            plot_filename = f"{model}-{input}-{metric_with_underscores}-bar_chart.png"
            plot_filepath = os.path.join(plots_path, plot_filename)
            plt.savefig(plot_filepath)
        
        if view_output:
            plt.show()

def create_or_update_aggregate_csv(aggregate_df, aggregate_csv_path):
    """Create or update the aggregate results CSV file for this set of experiments, which for each experiment contains each 
    deployment mechanism's aggregate results for each metric.

    Args:
        aggregate_df: The aggregate dataframe containing each deployment mechanism's aggregate results for each metric for
            this experiment.
        aggregate_csv_path: The path to the aggregate CSV file.
    """
    # Create the aggregate results CSV file if it does not exist
    if not os.path.exists(aggregate_csv_path):
        aggregate_df.to_csv(aggregate_csv_path, index=False)
    else:
        # Read in the existing aggregate results CSV file as a dataframe
        existing_aggregate_df = pd.read_csv(aggregate_csv_path)

        # Update the dataframe and subsequently the CSV file
        existing_aggregate_df = pd.concat([existing_aggregate_df, aggregate_df], ignore_index=True)
        existing_aggregate_df.to_csv(aggregate_csv_path, index=False)

def create_directory_if_not_exists(directory):
    """Create a directory if it does not exist.

    Args:
        directory: The path to the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    parser = argparse.ArgumentParser(description="Analyze performance data.")
    parser.add_argument("--experiment-set", type=str, required=True, help="The experiment set that the given experiment to analyze is from.")
    parser.add_argument("--model", type=str, required=True, help="The model used in the experiment to analyze.")
    parser.add_argument("--input", type=str, required=True, help="The input used in the experiment to analyze.")
    parser.add_argument("--significance-level", type=float, default=0.05, help="The significance level to use (e.g., 0.05).")
    parser.add_argument("--docker-overhead-view", type=int, default=2, help="The view of the Docker overhead to use (0: exclude daemon overhead, 1: include full daemon overhead, 2: include only additional docker overhead).")
    parser.add_argument("--include-insignificant-output", action="store_true", help="Include statistical comparisons when they are not statistically significant.")
    parser.add_argument("--mechanisms", type=str, default="docker,wasm_interpreted,wasm_jit,wasm_aot,native",
                    help="Comma-separated list of mechanisms to include (choose from docker, wasm_interpreted, wasm_jit, wasm_aot, wasm_aot_persistent, native, native_persistent)")
    parser.add_argument("--metrics", type=str, required=True, help="Comma-separated list of metrics to include.")
    parser.add_argument("--view-output", action="store_true", help="View the output of the analysis.")
    parser.add_argument("--save-output", action="store_true", 
        help="Save the output of the analysis to files. Note that the aggregate CSV will always be saved, since it is required for aggregate analysis.")
    parser.add_argument("--analyzed-results-dir", type=str, default="analyzed_results",
        help="The name of the directory to save the analyzed results in.")
    args = parser.parse_args()

    model = args.model
    input = args.input
    deployment_mechanisms = [mechanism.strip() for mechanism in args.mechanisms.split(",")]
    metrics = [metric.strip() for metric in args.metrics.split(",")] + COMPUTED_COLUMNS

    perf_filename = f"{model}&{input}&perf_results.csv"
    time_filename = f"{model}&{input}&time_results.csv"
    
    # The paths to the experiment's set directory within the results directory
    # the analyzed results directory within the experiment's set directory
    # the plots directory within the analyzed results directory
    # and the comparisons directory within the analyzed results directory
    experiments_set_path = os.path.join(RESULTS_DIR, args.experiment_set)
    analyzed_results_path = os.path.join(experiments_set_path, args.analyzed_results_dir)
    plots_path = os.path.join(analyzed_results_path, "plots")
    comparisons_path = os.path.join(analyzed_results_path, "comparisons")

    create_directory_if_not_exists(analyzed_results_path)
    create_directory_if_not_exists(plots_path)
    create_directory_if_not_exists(comparisons_path)

    perf_path = os.path.join(experiments_set_path, perf_filename)
    time_path = os.path.join(experiments_set_path, time_filename)

    perf_df = parse_csv_rows(perf_path, deployment_mechanisms, metrics, args.docker_overhead_view)
    time_df = parse_csv_rows(time_path, deployment_mechanisms, metrics, args.docker_overhead_view, is_perf_file=False)

    # Note that merging the dataframes in this way might suggest that trial number 1 of the
    # perf experiments corresponds to trial number 1 of the time experiments; this is not
    # actually true, but it is not important since trial number is not relevant for the analysis,
    # and doing this would produce the exact same results as if we had called the analyze_data_significant_difference
    # function on the two dataframes separately
    if perf_df.empty:
        df = time_df
    elif time_df.empty:
        df = perf_df
    else:
        df = pd.merge(perf_df, time_df, on=["deployment-mechanism", "trial-number"])
        
    metrics = get_metrics_in_df(df)
    aggregate_df = analyze_data_significant_difference(df, args.significance_level, metrics, model,
        input, comparisons_path, args.include_insignificant_output,
        args.view_output, args.save_output)
    aggregate_csv_filepath = os.path.join(analyzed_results_path, AGGREGATE_CSV_FILENAME)
    create_or_update_aggregate_csv(aggregate_df, aggregate_csv_filepath)
    
    if args.view_output or args.save_output:
        plot_metrics_bar_chart(aggregate_df, metrics, args.view_output, args.save_output, plots_path,
            model, input)

if __name__ == "__main__":
    main()
