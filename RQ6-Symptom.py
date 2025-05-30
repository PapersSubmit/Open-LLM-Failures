import json
import os
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set up font support and Times New Roman font verification
available_fonts = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
print("Available font names (partial list):")
for font_name in available_fonts[:20]:  # Show only first 20 to avoid long list
    print(font_name)
if "Times New Roman" in available_fonts:
    print("\nMatplotlib found 'Times New Roman' in its list")
else:
    print(
        "\nMatplotlib did not find a font named 'Times New Roman' in its list. Please check the printed list above for similar names (e.g., 'TimesNewRomanPSMT' or other variants)."
    )

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# Target JSON filenames list
TARGET_FILENAMES = [
    "DeepSeek-R1.json",
    "DeepSeek-V3.json",
    "meta-llama.json",
    "meta-llama3.json",
    "Qwen.json",
    "Qwen2dot5.json",
]

# Define required fields
REQUIRED_FIELDS = ["model", "SYMPTOM"]


def extract_top_level_symptom(symptom_text):
    """
    Extract top-level symptom classification

    This function focuses on 5 main symptom categories to provide a clear
    macro perspective and avoid over-fragmentation of analysis.

    Args:
    - symptom_text: Original symptom text, may contain hierarchical structure

    Returns:
    - Top-level symptom classification, or None if not in main categories
    """
    if not symptom_text or not isinstance(symptom_text, str):
        return None

    # Define the 5 top-level symptom categories we focus on
    target_categories = {
        "Crash",  # System crashes
        "Incorrect Functionality",  # Functional errors
        "Loading Failure",  # Loading failures
        "Hang",  # System hangs
        "Poor Performance",  # Performance issues
    }

    # Extract first level classification (first part after splitting by " - ")
    first_level = symptom_text.split(" - ")[0].strip()

    # Only return symptoms that belong to target categories
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

    print(
        f"Starting search for target JSON files in directory {script_dir} and subdirectories..."
    )
    found_files_count = 0

    for target_name in TARGET_FILENAMES:
        for filepath in script_dir.rglob(target_name):
            found_files_count += 1
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
                        else:
                            print(
                                f"    Warning: File {filepath} content is not a valid JSON object or array."
                            )
                    except json.JSONDecodeError:
                        print(
                            f"    Warning: File {filepath} content is not standard JSON. Trying to parse as JSON Lines..."
                        )
                        file_items = []
                        for line in content.splitlines():
                            line = line.strip()
                            if line:
                                try:
                                    item = json.loads(line)
                                    file_items.append(item)
                                except json.JSONDecodeError as e_line:
                                    print(
                                        f"      Warning: Skipping invalid JSON line: {line[:100]}... - {e_line}"
                                    )
                        if file_items:
                            all_json_data.extend(file_items)
                        else:
                            print(
                                f"      Failed to parse any data from {filepath} as JSON Lines format."
                            )
            except Exception as e:
                print(f"    Error reading or parsing file {filepath}: {e}")

    print(
        f"\nTotal loaded {len(all_json_data)} JSON records from {found_files_count} files.\n"
    )
    return all_json_data


def analyze_symptom_statistics_top_level(data_list):
    """
    Analyze top-level symptom statistics

    This function uses a simplified analysis approach, focusing on 5 main symptom categories.

    Returns:
    - overall_symptom_counts: Overall symptom count dictionary
    - model_symptom_counts: Symptom count dictionary grouped by model
    - valid_records: Number of valid records
    """
    print(
        "Starting analysis of top-level symptom data (focusing on 5 main categories)..."
    )

    # Initialize counters
    overall_symptom_counts = Counter()
    model_symptom_counts = defaultdict(Counter)
    valid_records = 0

    # Count filtered records
    filtered_records = 0

    for item in data_list:
        model = item.get("model")
        symptom_text = item.get("SYMPTOM")

        # Skip records missing required fields
        if not model or not symptom_text:
            continue

        # Extract top-level symptom classification
        top_level_symptom = extract_top_level_symptom(symptom_text)

        if top_level_symptom:
            valid_records += 1
            overall_symptom_counts[top_level_symptom] += 1
            model_symptom_counts[model][top_level_symptom] += 1
        else:
            filtered_records += 1

    print(f"Analysis completed. Processed {valid_records} valid records.")
    print(f"Filtered {filtered_records} records not belonging to main categories.")
    print(f"Involved {len(model_symptom_counts)} different models.")
    print(f"Found main symptom categories: {list(overall_symptom_counts.keys())}")

    return overall_symptom_counts, model_symptom_counts, valid_records


def create_output_directory():
    """Create RQ6 output directory"""
    script_dir = Path(__file__).parent
    output_dir = script_dir / "RQ6"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory prepared: {output_dir}")
    return output_dir


def create_model_symptom_stacked_chart_simplified(model_symptom_counts, output_dir):
    """
    Create horizontal stacked bar chart for selected 6 models
    Fixed order display:
    DeepSeek-V3, DeepSeek-R1, Llama, Llama3, Qwen, Qwen2.5
    Show only specific model names, not series names
    """
    print("  Generating horizontal stacked bar chart for selected 6 models...")

    if not model_symptom_counts:
        print("    Warning: No model symptom data")
        return

    # Define selected 6 target models and their fixed display order
    target_models_order = [
        "DeepSeek-V3",  # DeepSeek series
        "DeepSeek-R1",  # DeepSeek series
        "Llama",  # Llama series
        "Llama3",  # Llama series
        "Qwen",  # Qwen series
        "Qwen2.5",  # Qwen series
    ]

    # Define fixed order of 5 main symptom categories
    symptom_categories = [
        "Crash",
        "Incorrect Functionality",
        "Loading Failure",
        "Hang",
        "Poor Performance",
    ]

    # Filter and organize target model data - using fuzzy matching
    filtered_models = {}
    for model_name, symptoms in model_symptom_counts.items():
        # Use fuzzy matching to identify target models
        model_lower = model_name.lower()
        matched_target = None

        if "deepseek v3" in model_lower or "deepseek-v3" in model_lower:
            matched_target = "DeepSeek-V3"
        elif "deepseek r1" in model_lower or "deepseek-r1" in model_lower:
            matched_target = "DeepSeek-R1"
        elif "llama3" in model_lower or "llama-3" in model_lower:
            matched_target = "Llama3"
        elif (
            "llama" in model_lower
            and "3" not in model_lower
            and "codellama" not in model_lower
        ):
            matched_target = "Llama"
        elif "qwen2.5" in model_lower or "qwen2dot5" in model_lower:
            matched_target = "Qwen2.5"
        elif "qwen" in model_lower and "2" not in model_lower:
            matched_target = "Qwen"

        if matched_target and matched_target in target_models_order:
            if matched_target not in filtered_models:
                filtered_models[matched_target] = symptoms
            else:
                # If multiple matches, merge symptom data
                for symptom, count in symptoms.items():
                    filtered_models[matched_target][symptom] = (
                        filtered_models[matched_target].get(symptom, 0) + count
                    )

    if not filtered_models:
        print("    Warning: No target model data found")
        print(f"    Available models: {list(model_symptom_counts.keys())}")
        return

    print(f"    Successfully filtered {len(filtered_models)} target models")
    print(f"    Target models: {list(filtered_models.keys())}")

    # Organize models in specified order
    ordered_models = []
    ordered_model_names = []
    for model_name in target_models_order:
        if model_name in filtered_models:
            ordered_models.append(filtered_models[model_name])
            ordered_model_names.append(model_name)

    if not ordered_models:
        print("    Warning: No valid model data found")
        return

    print(f"    Final included models: {ordered_model_names}")

    # Important modification: Reverse list order to adapt to Matplotlib's Y-axis bottom-to-top display
    # This ensures DeepSeek-V3 appears at the top, followed by DeepSeek-R1, Llama, Llama3, Qwen, Qwen2.5
    ordered_models.reverse()
    ordered_model_names.reverse()

    # Create series boundaries for separator lines - adjust boundary positions for reversed order
    series_boundaries = []
    if len(ordered_model_names) >= 2:  # Between DeepSeek series and Llama series
        series_boundaries.append(
            (
                "DeepSeek-Llama",
                len(ordered_model_names) - 3 if len(ordered_model_names) >= 4 else 0,
                None,
            )
        )
    if len(ordered_model_names) >= 4:  # Between Llama series and Qwen series
        series_boundaries.append(
            (
                "Llama-Qwen",
                len(ordered_model_names) - 5 if len(ordered_model_names) >= 6 else 0,
                None,
            )
        )

    # Prepare data matrix for each model
    data_matrix = []
    for symptoms in ordered_models:
        model_data = []
        for symptom in symptom_categories:
            count = symptoms.get(symptom, 0)
            model_data.append(count)
        data_matrix.append(model_data)

    data_array = np.array(data_matrix)

    # Use red gradient colors consistent with the example
    colors = [
        "#c12746",  # Deep red - Crash
        "#f1695d",  # Medium red - Incorrect Functionality
        "#f59700",  # Orange red - Loading Failure
        "#f9cb40",  # Yellow orange - Hang
        "#d77efc",  # Purple red - Poor Performance
    ]

    # Create optimized chart layout
    fig, ax = plt.subplots(figsize=(14, 6))

    # Generate Y-axis positions
    y_positions = list(range(len(ordered_model_names)))

    left_positions = np.zeros(len(ordered_model_names))

    # Draw stacked bars for each symptom category
    for symptom_idx, symptom_name in enumerate(symptom_categories):
        symptom_values = data_array[:, symptom_idx].copy()

        bars = ax.barh(
            y_positions,
            symptom_values,
            left=left_positions,
            color=colors[symptom_idx],
            label=symptom_name,
            edgecolor="white",
            linewidth=0.8,
        )

        # Add value labels - only show labels for original values ≥ 2
        for i, (bar, value) in enumerate(zip(bars, symptom_values)):
            if value >= 2:
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
                    fontsize=20,
                    fontname="Times New Roman",
                )

        left_positions += symptom_values

    # Set chart properties, apply Times New Roman font
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        ordered_model_names,
        fontweight="bold",
        fontsize=27,
        fontname="Times New Roman",
    )

    # Apply Times New Roman font to X-axis tick labels
    ax.tick_params(axis="x", labelsize=27)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
        label.set_fontname("Times New Roman")

    # Add fine grid lines
    ax.grid(axis="x", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Add red dashed separators for series grouping
    for i, (series, boundary_pos, _) in enumerate(series_boundaries):
        if boundary_pos is not None and boundary_pos < len(ordered_model_names) - 1:
            separator_y = boundary_pos + 0.5
            ax.axhline(
                y=separator_y,
                color="red",
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
            )

    # Optimize legend position
    ax.legend(
        loc="upper right",
        ncol=1,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        edgecolor="gray",
        facecolor="white",
        prop={"family": "Times New Roman", "weight": "bold", "size": 25},
    )

    plt.tight_layout()
    plt.savefig(output_dir / "RQ6-Symptom.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"    ✓ Generated horizontal stacked bar chart for selected models")
    print(
        f"    ✓ Includes 6 models in fixed order (top to bottom: DeepSeek-V3, DeepSeek-R1, Llama, Llama3, Qwen, Qwen2.5)"
    )
    print(f"    ✓ Uses red dashed lines to separate different series")
    print(f"    ✓ All text elements use Times New Roman font")


def main():
    """
    Main function: Coordinate the entire top-level classification-based symptom analysis process
    """
    print(
        "Starting top-level classification-based symptom analysis and visualization process...\n"
    )
    print(
        "Focusing on 5 main symptom categories: Crash, Incorrect Functionality, Loading Failure, Hang, Poor Performance\n"
    )

    # 1. Load data
    all_data = load_json_files()
    if not all_data:
        print("No data found, program exits.")
        return

    # 2. Create output directory
    output_dir = create_output_directory()

    # 3. Analyze symptom statistics using simplified method
    overall_symptom_counts, model_symptom_counts, valid_records = (
        analyze_symptom_statistics_top_level(all_data)
    )

    if not overall_symptom_counts:
        print("No valid symptom data found for the 5 main categories, program exits.")
        return

    # 4. Generate only the required PDF chart
    print("\n" + "=" * 50)
    print("Generating RQ6-Symptom.pdf chart...")
    print("=" * 50)

    # Generate horizontal stacked chart for model symptom distribution
    create_model_symptom_stacked_chart_simplified(model_symptom_counts, output_dir)

    # 5. Output summary information
    print("\n" + "=" * 50)
    print("Top-level classification analysis completed! Results summary:")
    print("=" * 50)
    print(f"✓ Processed records: {valid_records}")
    print(f"✓ Found main symptom categories: {list(overall_symptom_counts.keys())}")
    print(f"✓ Involved models: {len(model_symptom_counts)}")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Generated file:")
    print(
        f"  - RQ6-Symptom.pdf (horizontal stacked chart for model symptom distribution)"
    )
    print(
        "\n🎯 Focus: Generated horizontal stacked chart designed exactly as requested!"
    )
    print(
        "📊 Analysis focuses on 5 top-level symptom categories, providing clear macro perspective."
    )
    print(
        "🔍 All hierarchical symptoms (e.g., 'Incorrect Functionality - Model Behavior Anomaly')"
    )
    print(
        "   are correctly classified into corresponding top-level categories (e.g., 'Incorrect Functionality')."
    )
    print(
        "✨ All chart text now uses Times New Roman font for professional visual consistency!"
    )


if __name__ == "__main__":
    main()
