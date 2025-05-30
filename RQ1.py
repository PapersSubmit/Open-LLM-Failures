import json
import os
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Target JSON filenames
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


def extract_top_level_symptom(symptom_text):
    """Extract top-level symptom category"""
    if not symptom_text or not isinstance(symptom_text, str):
        return None

    # Define 5 main symptom categories
    target_categories = {
        "Crash",
        "Incorrect Functionality",
        "Loading Failure",
        "Hang",
        "Poor Performance",
    }

    # Extract first level category
    first_level = symptom_text.split(" - ")[0].strip()

    if first_level in target_categories:
        return first_level
    else:
        return None


def load_json_files():
    """Load all target JSON files and return data list"""
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


def analyze_symptom_data(data_list):
    """Analyze symptom statistics focusing on 5 main categories"""
    print("Analyzing top-level symptom data...")

    model_symptom_counts = defaultdict(Counter)
    valid_records = 0

    for item in data_list:
        model = item.get("model")
        symptom_text = item.get("SYMPTOM")

        if not model or not symptom_text:
            continue

        top_level_symptom = extract_top_level_symptom(symptom_text)

        if top_level_symptom:
            valid_records += 1
            model_symptom_counts[model][top_level_symptom] += 1

    print(f"Analysis complete. Processed {valid_records} valid records.")
    print(f"Found {len(model_symptom_counts)} different models.")

    return model_symptom_counts


def get_model_series(model_name):
    """Determine model series based on model name"""
    model_lower = model_name.lower()
    if "qwen" in model_lower:
        return "Qwen\nSeries"
    elif "llama" in model_lower or "codellama" in model_lower:
        return "Llama\nSeries"
    elif "deepseek" in model_lower:
        return "DeepSeek\nSeries"
    else:
        return "Other Series"


def create_rq1_chart(model_symptom_counts, output_dir):
    """Create RQ1 horizontal stacked chart aggregated by series"""
    print(
        "Generating series-aggregated symptom distribution horizontal stacked chart..."
    )

    if not model_symptom_counts:
        print("Warning: No model symptom data")
        return

    # Define 5 main symptom categories in fixed order
    symptom_categories = [
        "Crash",
        "Incorrect Functionality",
        "Loading Failure",
        "Hang",
        "Poor Performance",
    ]

    # Aggregate data by series
    series_symptom_counts = defaultdict(lambda: defaultdict(int))

    print("Starting series data aggregation...")
    for model, symptoms in model_symptom_counts.items():
        series = get_model_series(model)
        print(f"  {model} -> {series}")

        for symptom, count in symptoms.items():
            series_symptom_counts[series][symptom] += count

    # Define series display order (top to bottom)
    series_order = [
        "Qwen\nSeries",
        "Llama\nSeries",
        "DeepSeek\nSeries",
    ]

    # Keep only series with actual data
    available_series = [
        series for series in series_order if series in series_symptom_counts
    ]

    if not available_series:
        print("Warning: No valid series data found")
        return

    print(f"Found series: {available_series}")

    # Print summary for each series
    for series in available_series:
        total_for_series = sum(series_symptom_counts[series].values())
        print(f"  {series}: Total symptoms = {total_for_series}")

    # Prepare data matrix for each series
    data_matrix = []
    for series in available_series:
        series_data = []
        for symptom in symptom_categories:
            count = series_symptom_counts[series].get(symptom, 0)
            series_data.append(count)
        data_matrix.append(series_data)

    data_array = np.array(data_matrix)

    # Use red gradient colors
    colors = [
        "#c12746",  # Dark red - Crash
        "#f1695d",  # Medium red - Incorrect Functionality
        "#f59700",  # Orange red - Loading Failure
        "#f9cb40",  # Yellow orange - Hang
        "#d77efc",  # Purple red - Poor Performance
    ]

    # Create horizontal stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 5))

    y_positions = np.arange(len(available_series))
    left_positions = np.zeros(len(available_series))
    bar_height = 0.6

    # Draw stacked bars for each symptom category
    for symptom_idx, symptom_name in enumerate(symptom_categories):
        symptom_values = data_array[:, symptom_idx]

        bars = ax.barh(
            y_positions,
            symptom_values,
            height=bar_height,
            left=left_positions,
            color=colors[symptom_idx],
            label=symptom_name,
            edgecolor="white",
            linewidth=0.8,
        )

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, symptom_values)):
            if value >= 5:
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
                    fontsize=25,
                    fontname="Times New Roman",
                )

        left_positions += symptom_values

    # Set chart properties
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        available_series, fontweight="bold", fontsize=35, fontname="Times New Roman"
    )

    # X-axis tick labels
    ax.tick_params(axis="x", labelsize=35)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
        label.set_fontname("Times New Roman")

    # Add grid lines
    ax.grid(axis="x", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Set legend
    ax.legend(
        loc="upper right",
        ncol=1,
        frameon=True,
        fancybox=True,
        shadow=True,
        prop={
            "family": "Times New Roman",
            "weight": "bold",
            "size": 25,
        },
        framealpha=0.95,
        edgecolor="gray",
        facecolor="white",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "RQ1-chart.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Generated series-aggregated horizontal stacked chart")
    print(f"Included series: {', '.join(available_series)}")


def main():
    """Main function: Generate RQ1 chart"""
    print("Starting RQ1 chart generation process...\n")

    # 1. Load data
    all_data = load_json_files()
    if not all_data:
        print("No data found, exiting.")
        return

    # 2. Create output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "RQ1"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory ready: {output_dir}")

    # 3. Analyze symptom statistics
    model_symptom_counts = analyze_symptom_data(all_data)

    if not model_symptom_counts:
        print("No valid symptom data found, exiting.")
        return

    # 4. Generate RQ1 chart
    print("\n" + "=" * 50)
    print("Generating RQ1 chart...")
    print("=" * 50)

    create_rq1_chart(model_symptom_counts, output_dir)

    # 5. Output summary
    print("\n" + "=" * 50)
    print("RQ1 chart generation complete!")
    print("=" * 50)
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Generated file: RQ1-chart.pdf")
    print("✓ Chart shows symptom distribution by model series")


if __name__ == "__main__":
    main()
