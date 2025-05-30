import json
import os
import re
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set font support
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# Target JSON file names
TARGET_FILENAMES = [
    "DeepSeek-Coder.json",
    "DeepSeek-Coder-V2.json",
    "DeepSeek-Math.json",
    "DeepSeek-MoE.json",
    "DeepSeek-R1.json",
    "DeepSeek-V2.json",
    "DeepSeek-V3.json",
    "meta-codellama.json",
    "meta-llama.json",
    "meta-llama3.json",
    "Qwen.json",
    "Qwen2dot5.json",
    "Qwen2dot5-Coder.json",
    "Qwen2dot5-Math.json",
]

# Required fields
REQUIRED_FIELDS = ["model", "ROOT CAUSE"]


def extract_top_level_root_cause(root_cause_text):
    """
    Extract top-level root cause category from the ROOT CAUSE text
    """
    if not root_cause_text or not isinstance(root_cause_text, str):
        return None

    # Define the 9 updated top-level root cause categories
    target_categories = {
        "Environment Compatibility Issues",
        "Configuration and Parameter Setting Errors",
        "Data and Computation Issues",
        "API Usage and Interface Issues",
        "Distributed and Parallel Computing Issues",
        "Tokenizer and Text Processing Issues",
        "Hardware and Resource Limitations",
        "Algorithm Implementation Defects",
        "Model Corruption Issues",
    }

    # Extract first level classification (first part after splitting by " - ")
    first_level = root_cause_text.split(" - ")[0].strip()

    # Check if it belongs to target categories
    if first_level in target_categories:
        return first_level
    else:
        return None


def load_json_files():
    """
    Load all target JSON files and return data list
    """
    script_dir = Path(__file__).parent
    all_json_data = []

    print(f"Searching for target JSON files in {script_dir} and subdirectories...")

    for target_name in TARGET_FILENAMES:
        for filepath in script_dir.rglob(target_name):
            print(f"  Found and reading file: {filepath}")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    try:
                        data_list = json.loads(content)
                        if isinstance(data_list, list):
                            all_json_data.extend(data_list)
                        elif isinstance(data_list, dict):
                            all_json_data.append(data_list)
                    except json.JSONDecodeError:
                        # Try parsing as JSON Lines format
                        file_items = []
                        for line in content.splitlines():
                            line = line.strip()
                            if line:
                                try:
                                    item = json.loads(line)
                                    file_items.append(item)
                                except json.JSONDecodeError:
                                    continue
                        if file_items:
                            all_json_data.extend(file_items)
            except Exception as e:
                print(f"    Error reading file {filepath}: {e}")

    print(f"\nLoaded {len(all_json_data)} JSON records from files.\n")
    return all_json_data


def analyze_root_cause_statistics(data_list):
    """
    Analyze root cause statistics with top-level categorization
    """
    print("Starting root cause analysis...")

    overall_root_cause_counts = Counter()
    model_root_cause_counts = defaultdict(Counter)
    valid_records = 0

    for item in data_list:
        model = item.get("model")
        root_cause_text = item.get("ROOT CAUSE")

        # Check required fields
        if not model or not root_cause_text:
            continue

        # Extract top-level root cause
        top_level_root_cause = extract_top_level_root_cause(root_cause_text)

        if top_level_root_cause:
            valid_records += 1
            overall_root_cause_counts[top_level_root_cause] += 1
            model_root_cause_counts[model][top_level_root_cause] += 1

    print(f"Valid records: {valid_records}")
    print(f"Root cause categories found: {list(overall_root_cause_counts.keys())}")

    return overall_root_cause_counts, model_root_cause_counts, valid_records


def get_model_series(model_name):
    """
    Determine model series based on model name
    """
    model_lower = model_name.lower()
    if "qwen" in model_lower:
        return "Qwen"
    elif "llama" in model_lower or "codellama" in model_lower:
        return "llama"
    elif "deepseek" in model_lower:
        return "DeepSeek"
    else:
        return "Other"


def create_model_root_cause_stacked_chart(model_root_cause_counts, output_dir):
    """
    Create series-level horizontal stacked bar chart for root causes
    """
    print("Generating series-level root cause horizontal stacked bar chart...")

    if not model_root_cause_counts:
        print("Warning: No model root cause data")
        return

    # Define the 9 professional root cause categories in fixed order
    root_cause_categories = [
        "Environment Compatibility Issues",
        "Configuration and Parameter Setting Errors",
        "Data and Computation Issues",
        "API Usage and Interface Issues",
        "Distributed and Parallel Computing Issues",
        "Tokenizer and Text Processing Issues",
        "Hardware and Resource Limitations",
        "Algorithm Implementation Defects",
        "Model Corruption Issues",
    ]

    # Series-level data aggregation
    print("Performing series-level data aggregation...")
    series_aggregated_data = defaultdict(lambda: defaultdict(int))
    series_model_counts = defaultdict(int)

    # Aggregate data by series
    for model, root_causes in model_root_cause_counts.items():
        series = get_model_series(model)
        series_model_counts[series] += 1

        for root_cause, count in root_causes.items():
            series_aggregated_data[series][root_cause] += count

    # Define series order and display names
    series_order = ["Qwen", "llama", "DeepSeek"]
    series_display_names = ["Qwen\nSeries", "Llama\nSeries", "DeepSeek\nSeries"]

    # Prepare final aggregated data
    final_series_data = []
    final_display_names = []

    for series in series_order:
        if series in series_aggregated_data:
            final_series_data.append(series_aggregated_data[series])
            series_name = {
                "Qwen": "Qwen\nSeries",
                "llama": "Llama\nSeries",
                "DeepSeek": "DeepSeek\nSeries",
            }[series]
            final_display_names.append(series_name)

    if not final_series_data:
        print("Warning: No valid series data found")
        return

    print(f"Successfully aggregated {len(final_series_data)} series")

    # Build data matrix for stacked chart
    data_matrix = []
    for series_data in final_series_data:
        series_row = []
        for root_cause in root_cause_categories:
            count = series_data.get(root_cause, 0)
            series_row.append(count)
        data_matrix.append(series_row)

    data_array = np.array(data_matrix)

    # Professional color scheme for 9 categories
    colors = [
        "#c12746",  # ECI - Deep red
        "#f1695d",  # CPSE - Orange red
        "#f59700",  # DCI - Orange
        "#f9cb40",  # AUII - Yellow
        "#d77efc",  # DPCI - Light purple
        "#9046cf",  # TTPI - Deep purple
        "#0061b6",  # HRL - Blue
        "#21acb8",  # AID - Cyan
        "#8FCBAD",  # MCI - Light green
    ]

    # Create compact series-level stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 5))

    y_positions = np.arange(len(final_display_names))
    left_positions = np.zeros(len(final_display_names))

    # Draw stacked bars for each root cause category
    for root_cause_idx, root_cause_name in enumerate(root_cause_categories):
        root_cause_values = data_array[:, root_cause_idx]

        # Create complete legend labels
        legend_labels = [
            "Environment Compatibility Issues",
            "Configuration and Parameter Setting Errors",
            "Data and Computation Issues",
            "API Usage and Interface Issues",
            "Distributed and Parallel Computing Issues",
            "Tokenizer and Text Processing Issues",
            "Hardware and Resource Limitations",
            "Algorithm Implementation Defects",
            "Model Corruption Issues",
        ]

        bars = ax.barh(
            y_positions,
            root_cause_values,
            left=left_positions,
            height=0.6,
            color=colors[root_cause_idx],
            label=legend_labels[root_cause_idx],
            edgecolor="white",
            linewidth=0.8,
        )

        # Add value labels (only show values ≥ 3)
        for i, (bar, value) in enumerate(zip(bars, root_cause_values)):
            if value >= 3:
                label_x = left_positions[i] + value / 2
                label_y = bar.get_y() + bar.get_height() / 2
                ax.text(
                    label_x,
                    label_y,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white",
                    fontsize=23,
                    fontfamily="Times New Roman",
                )

        left_positions += root_cause_values

    # Set chart properties
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        final_display_names,
        fontweight="bold",
        fontsize=35,
        fontfamily="Times New Roman",
    )

    # X-axis label optimization
    ax.tick_params(axis="x", labelsize=35)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
        label.set_fontfamily("Times New Roman")

    # Add grid lines
    ax.grid(axis="x", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Place legend inside chart (upper right corner)
    ax.legend(
        loc="upper right",
        ncol=1,
        frameon=True,
        fancybox=True,
        shadow=True,
        prop={
            "family": "Times New Roman",
            "weight": "bold",
            "size": 15,
        },
        framealpha=0.95,
        edgecolor="gray",
        facecolor="white",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "RQ2-chart.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Generated series-level aggregated horizontal stacked bar chart")
    print(
        f"Showing {len(final_display_names)} model series aggregated root cause distribution"
    )


def create_output_directory():
    """Create RQ2 output directory"""
    script_dir = Path(__file__).parent
    output_dir = script_dir / "RQ2"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory prepared: {output_dir}")
    return output_dir


def main():
    """
    Main function: Generate RQ2-chart.pdf
    """
    print("🚀 Starting RQ2 chart generation...")
    print("Focusing on AI system failure root cause categories")
    print("=" * 80)

    # 1. Load data
    print("📁 Step 1: Loading JSON data files...")
    all_data = load_json_files()
    if not all_data:
        print("❌ No data found, exiting.")
        return
    print(f"✅ Successfully loaded {len(all_data)} records")

    # 2. Create output directory
    print("📂 Step 2: Preparing output directory...")
    output_dir = create_output_directory()

    # 3. Analyze root causes
    print("🔬 Step 3: Performing root cause analysis...")
    overall_root_cause_counts, model_root_cause_counts, valid_records = (
        analyze_root_cause_statistics(all_data)
    )

    if not overall_root_cause_counts:
        print("❌ No valid root cause data found.")
        return

    # 4. Generate chart
    print("📊 Step 4: Generating RQ2-chart.pdf...")
    create_model_root_cause_stacked_chart(model_root_cause_counts, output_dir)

    # 5. Summary
    print("\n" + "=" * 80)
    print("🎉 RQ2 chart generation completed!")
    print("=" * 80)

    print(f"📊 Results:")
    print(f"   • Valid records: {valid_records}")
    print(f"   • Root cause categories: {list(overall_root_cause_counts.keys())}")
    print(f"   • Models involved: {len(model_root_cause_counts)}")

    print(f"📁 Output file:")
    print(f"   • {output_dir}/RQ2-chart.pdf")

    print("=" * 80)


if __name__ == "__main__":
    main()
