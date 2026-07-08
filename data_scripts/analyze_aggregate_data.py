"""This script analyzes the aggregate performance data for a set of experiments. It should only be run after
analyze_data.py has been run for all the experiments in the set.
"""
import pandas as pd
import statsmodels.stats.weightstats as smw
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from IPython.display import display
import re


plt.rc('font',   size=12)   # base size: 12 pt  (default is 10)
plt.rc('axes',   titlesize=14)  # axes title:   14 pt  (default ≈ 12)
plt.rc('axes',   labelsize=12)  # axis labels:  12 pt  (default ≈ 10)
plt.rc('xtick',  labelsize=12)  # tick labels:  10 pt  (default ≈ 10)
plt.rc('ytick',  labelsize=12)
plt.rc('legend', fontsize=12)   # legend text:  10 pt  (default ≈ 10)
plt.rc('figure', titlesize=16)  # suptitle:     16 pt  (default ≈ 12)


# The names of columns that are not metrics and must hence always be included in the dataframes
NON_METRIC_COLUMNS = ["model", "input", "deployment-mechanism"]

# The absolute path of the "data_scripts" directory where this script is in
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))

# The absolute path of the parent directory, which is the root of the benchmark suite
BENCHMARK_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))

# The absolute path of the "results" directory where the results of the experiments are stored
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")

# The name of the CSV file where the aggregate results from all the experiments within
# an experiment set are stored 
AGGREGATE_CSV_FILENAME = "aggregate_results.csv"

# Maps deployment mechanisms to colors and line styles for plotting
DEPLOYMENT_MECHANISM_TO_COLOR = {
    "wasm_aot": "tab:red",
    "wasm_aot_persistent": "tab:pink",
    "wasm_jit": "tab:purple",
    "wasm_interpreted": "tab:blue",
    "docker": "tab:green",
    "native": "tab:orange",
    "native_persistent": "tab:brown",
}
DEPLOYMENT_MECHANISM_TO_LINESTYLE = {
    "wasm_aot": "-",
    "wasm_aot_persistent": "--",
    "wasm_jit": "-",
    "wasm_interpreted": "--",
    "docker": "-.",
    "native": ":",
    "native_persistent": "--",
}
DEPLOYMENT_MECHANISM_TO_LABEL = {
    "wasm_aot": "AOT-compiled WebAssembly",
    "wasm_aot_persistent": "AOT-compiled WebAssembly persistent",
    "wasm_jit": "JIT-compiled WebAssembly",
    "wasm_interpreted": "Interpreted WebAssembly",
    "docker": "Docker",
    "native": "Native",
    "native_persistent": "Native persistent",
}

def format_input_label(filename):
    """Convert a raw input filename to a human-friendly x-tick label.
    
    Args: 
        filename: the filename.

    Returns:
        str: The human-friendly x-tick label.
    """
    base = os.path.splitext(filename)[0]
    filename_parts = [part for part in base.split("_") if part.lower() != "test"]

    label = filename_parts[0]
    if label.lower().startswith("cifar"):
        # Only CIFAR gets a dash between letters and trailing digits
        m = re.match(r"([A-Za-z]+)(\d+)$", label)
        formatted_label = f"{m.group(1)}-{m.group(2)}" if m else label
    else:
        formatted_label = label 

    index = filename_parts[-1].lstrip("0") or "0"

    return f"{formatted_label} #{index}"

def chart_compare_across_models_or_inputs(aggregate_df, metrics, across_models, variable_values, constant_value, 
    view_output, save_output, plots_path, chart_type):
    """Produce charts comparing the performance of different deployment mechanisms across different models or inputs.
    
    Args:
        aggregate_df: The dataframe containing the aggregate results.
        metrics: The metrics to analyze.
        across_models: Whether to compare across models or inputs.
        variable_values: The values of the variable (e.g. if comparing across models, then the names of the models) to compare.
        constant_value: The value of the constant (e.g. if comparing across models, then the name of the input) to use in comparing models.
        view_output: Whether to view the output of the analysis.
        save_output: Whether to save the output of the analysis to files.
        plots_path: The path to the directory where the plots should be saved.
        chart_type: The type of chart to produce (currently, only "lineplot" and "bar").
    """
    deployment_mechanisms = aggregate_df["deployment-mechanism"].unique()
    variable_values_str = "_".join(variable_values)

    if across_models:            
        # If comparing across models, then models represent the variable, while the input represents a constant
        variable = "model"
        constant = "input"
        plot_filename_prefix = f"aggregate_models_{variable_values_str}_for_input_{constant_value}"
    else:
        # Otherwise, it is the other way around
        variable = "input"
        constant = "model"
        plot_filename_prefix = f"aggregate_models_{variable_values_str}_for_model_{constant_value}"

    # if we're comparing across inputs, variable_values are filenames
    if not across_models:
        pretty_labels = [format_input_label(f) for f in variable_values]
    else:
        # if across_models, leave model names untouched
        pretty_labels = variable_values

    for metric in metrics:
        # Ensure this metric is in this dataframe (since some metrics are only for the perf dataframes,
        # and others for the time dataframes)
        if f"{metric}-mean" in aggregate_df.columns:
            plt.figure(metric)
            metric_name_without_hyphen = metric.replace("-", " ")
            capitalized_metric_name_without_hyphen = metric_name_without_hyphen.capitalize()

            # If the metric name is e.g. CPU usage, capitalize would make it "Cpu usage",
            # so we must replace occurrences of "Cpu" with "CPU"
            capitalized_metric_name_without_hyphen = capitalized_metric_name_without_hyphen.replace("Cpu", "CPU")

            metric_with_underscores = metric.replace("-", "_")

            # Variables for the bar chart case
            x = np.arange(len(variable_values)) # One position per model / input
            width = 0.15 # Width of a single bar
            offsets = np.linspace( # Shift each mechanism sideways
                -width * (len(deployment_mechanisms) - 1) / 2,
                width * (len(deployment_mechanisms) - 1) / 2,
                len(deployment_mechanisms),
            )

            for off, deployment_mechanism in zip(offsets, deployment_mechanisms):

                # Get only the rows for this deployment mechanism
                deployment_mechanism_metric_df = aggregate_df[aggregate_df["deployment-mechanism"] == deployment_mechanism]

                # Plot the mean and confidence interval for each deployment mechanism
                means = deployment_mechanism_metric_df[f"{metric}-mean"].tolist()
                errors = [deployment_mechanism_metric_df[f"{metric}-error-lower"].tolist(), deployment_mechanism_metric_df[f"{metric}-error-upper"].tolist()]
                
                if chart_type == "lineplot":
                    plt.errorbar(pretty_labels, means, yerr=errors, label=DEPLOYMENT_MECHANISM_TO_LABEL[deployment_mechanism], capsize=5, 
                        color=DEPLOYMENT_MECHANISM_TO_COLOR[deployment_mechanism], linestyle=DEPLOYMENT_MECHANISM_TO_LINESTYLE[deployment_mechanism])
                elif chart_type == "bar":
                    plt.bar(x + off, means, width=width, yerr=errors, capsize=4,
                    label=DEPLOYMENT_MECHANISM_TO_LABEL[deployment_mechanism],
                    color=DEPLOYMENT_MECHANISM_TO_COLOR[deployment_mechanism])
            
            # Set title and labels
            title = f"{capitalized_metric_name_without_hyphen} by {variable} on {constant} {constant_value}\nfor different deployment mechanisms"
            capitalized_variable = variable.capitalize()
            plt.title(title)
            plt.ylabel(capitalized_metric_name_without_hyphen)
            plt.xlabel(capitalized_variable)
            plt.legend()

            # Rotate the x-axis labels for better readability
            if chart_type == "lineplot":
                plt.xticks(rotation=45)
            elif chart_type == "bar":
                plt.xticks(x, pretty_labels, rotation=10)
            plt.tight_layout()

            if save_output:
                plot_filename = f"{plot_filename_prefix}-{metric_with_underscores}-lineplot.png"
                plot_filepath = os.path.join(plots_path, plot_filename)
                plt.savefig(plot_filepath)

            if view_output:
                plt.show()

def compare_across_models_or_inputs(aggregate_df, across_models, variable_values, constant_value, 
    metrics, view_output, save_output, plots_path, chart_type):
    """Compare the performance of different deployment mechanisms across different models or inputs.

    Args:
        aggregate_df: The dataframe containing the aggregate results.
        across_models: Whether to compare across models or inputs.
        variable_values: The values of the variable (e.g. if comparing across models, then the names of the models) to compare.
        constant_value: The value of the constant (e.g. if comparing across models, then the name of the input) to use in comparing models.
        metrics: The metrics to analyze.
        view_output: Whether to view the output of the analysis.
        save_output: Whether to save the output of the analysis to files.
        plots_path: The path to the directory where the plots should be saved.
        chart_type: The type of chart to produce (currently, only "lineplot" and "bar").
    """
    if across_models:
        # If comparing across models, then models represent the variable, while the input represents a constant
        variable = "model"
        constant = "input"
    else:
        # Otherwise, it is the other way around
        variable = "input"
        constant = "model"

    # Filter the dataframes to only include rows with the specified variable values and constant value
    aggregate_df = aggregate_df[aggregate_df[variable].isin(variable_values)]
    aggregate_df = aggregate_df[aggregate_df[constant] == constant_value]

    # For each metric and deployment mechanism, lineplot the mean and confidence intervals
    chart_compare_across_models_or_inputs(aggregate_df, metrics, across_models, variable_values, constant_value, view_output, 
        save_output, plots_path, chart_type)

def compare_across_models(aggregate_df, models_to_compare, input, metrics, view_output, save_output, plots_path, chart_type):
    """Compare the performance of different deployment mechanisms across different models.

    Args:
        aggregate_df: The dataframe containing the aggregate results.
        models_to_compare: The models to compare.
        input: The single input to use in comparing models.
        metrics: The metrics to analyze.
        view_output: Whether to view the output of the analysis.
        save_output: Whether to save the output of the analysis to files.
        plots_path: The path to the directory where the plots should be saved.
        chart_type: The type of chart to produce (currently, only "lineplot" and "bar").
    """
    compare_across_models_or_inputs(aggregate_df, True, models_to_compare, input, metrics, view_output, save_output, plots_path, chart_type)

def compare_across_inputs(aggregate_df, inputs_to_compare, model, metrics, view_output, save_output, plots_path, chart_type):
    """Compare the performance of different deployment mechanisms across different inputs.

    Args:
        aggregate_df: The dataframe containing the aggregate results.
        inputs_to_compare: The inputs to compare.
        model: The single model to use in comparing inputs.
        metrics: The metrics to analyze.
        view_output: Whether to view the output of the analysis.
        save_output: Whether to save the output of the analysis to files.
        plots_path: The path to the directory where the plots should be saved.
        chart_type: The type of chart to produce (currently, only "lineplot" and "bar").
    """
    compare_across_models_or_inputs(aggregate_df, False, inputs_to_compare, model, metrics, view_output, save_output, plots_path, chart_type)

def remove_irrelevant_df_columns(df, metric_cols):
    """Remove columns not relevant to the analysis from the dataframe.
    
    Args:
        df: The dataframe to remove columns from.
        metric_cols: The names of the columns containing metrics.
    """
    # Remove the columns that are not required
    cols_to_keep = [col for col in NON_METRIC_COLUMNS + metric_cols if col in df.columns]
    return df[cols_to_keep]

def main():
    parser = argparse.ArgumentParser(description="Analyze aggregated performance data for a set of experiments")
    parser.add_argument("--experiment-set", type=str, required=True, help="The experiment set to analyze.")
    parser.add_argument("--compare-across-models", action="store_true", help="Compare across models.")
    parser.add_argument("--models-to-compare", type=str, help="The models to compare.")
    parser.add_argument("--input", type=str, help="The single input to use in comparing models.")
    parser.add_argument("--compare-across-inputs", action="store_true", help="Compare across inputs.")
    parser.add_argument("--inputs-to-compare", type=str, help="The inputs to compare.")
    parser.add_argument("--model", type=str, help="The model to use in comparing inputs.")
    parser.add_argument("--metrics", type=str, help="The metrics to analyze.")
    parser.add_argument("--chart-type", type=str, choices=["lineplot", "bar"], default="lineplot",
        help="The type of chart to produce (lineplot or bar).")
    parser.add_argument("--view-output", action="store_true", help="View the output of the analysis.")
    parser.add_argument("--save-output", action="store_true", 
        help="Save the output of the analysis to files.")
    parser.add_argument("--analyzed-results-dir", type=str, default="analyzed_results",
        help="The name of the directory to save the analyzed results in.")

    args = parser.parse_args()

    metrics = [metric.strip() for metric in args.metrics.split(",")]

    # Load the aggregate results
    experiments_set_path = os.path.join(RESULTS_DIR, args.experiment_set)
    analyzed_results_path = os.path.join(experiments_set_path, args.analyzed_results_dir)
    aggregate_csv_path = os.path.join(analyzed_results_path, AGGREGATE_CSV_FILENAME)
    aggregate_df = pd.read_csv(aggregate_csv_path)

    # Get the names of the columns corresponding to the provided metrics
    metric_cols_suffixes = ["-mean", "-error-lower", "-error-upper"]
    metric_cols = [f"{metric}{suffix}" for metric in metrics for suffix in metric_cols_suffixes]

    # Remove irrelevant columns from the dataframe
    aggregate_df = remove_irrelevant_df_columns(aggregate_df, metric_cols)

    # Get the path to the plots directory
    plots_path = os.path.join(analyzed_results_path, "plots")

    if args.compare_across_models:
        if args.models_to_compare is None:
            print("You must provide a list of models to compare.")
            exit(1)
        if args.input is None:
            print("You must provide a single input to use in comparing models.")
            exit(1)
        models_to_compare = [model.strip() for model in args.models_to_compare.split(",")]
        compare_across_models(aggregate_df, models_to_compare, args.input, metrics, args.view_output, args.save_output,
            plots_path, args.chart_type)
    if args.compare_across_inputs:
        if args.inputs_to_compare is None:
            print("You must provide a list of inputs to compare.")
            exit(1)
        if args.model is None:
            print("You must provide a single model to use in comparing inputs.")
            exit(1)
        inputs_to_compare = [input.strip() for input in args.inputs_to_compare.split(",")]
        compare_across_inputs(aggregate_df, inputs_to_compare, args.model, metrics, args.view_output, args.save_output,
            plots_path, args.chart_type)

if __name__ == "__main__":
    main()
