import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
import warnings

warnings.filterwarnings("ignore")

# Set font and style - enhanced version, specifically focusing on tick labels and legend
plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
# Key improvement: explicitly set tick label font size
plt.rcParams["xtick.labelsize"] = 30  # X-axis tick label font size
plt.rcParams["ytick.labelsize"] = 30  # Y-axis tick label font size
plt.rcParams["legend.fontsize"] = 20  # Default legend font size
sns.set_style("whitegrid")

# Data definition
fine_tuning_data = {
    "symptoms": {
        "Crash": 81,
        "Incorrect Functionality": 30,
        "Loading Failure": 27,
        "Hang": 7,
        "Poor Performance": 4,
    },
    "root_causes": {
        "Configuration and Parameter Setting Errors": 46,
        "Environment Compatibility Issues": 42,
        "API Usage and Interface Issues": 12,
        "Hardware and Resource Limitations": 11,
        "Data and Computation Issues": 9,
        "Distributed and Parallel Computing Issues": 9,
        "Algorithm Implementation Defects": 8,
        "Tokenizer and Text Processing Issues": 8,
        "Model Corruption Issues": 4,
    },
}

inference_data = {
    "symptoms": {
        "Crash": 214,
        "Incorrect Functionality": 210,
        "Loading Failure": 110,
        "Hang": 12,
        "Poor Performance": 11,
    },
    "root_causes": {
        "Environment Compatibility Issues": 181,
        "Configuration and Parameter Setting Errors": 84,
        "Data and Computation Issues": 62,
        "API Usage and Interface Issues": 53,
        "Distributed and Parallel Computing Issues": 45,
        "Tokenizer and Text Processing Issues": 43,
        "Algorithm Implementation Defects": 38,
        "Hardware and Resource Limitations": 36,
        "Model Corruption Issues": 15,
    },
}

# Calculate totals
ft_total_symptoms = sum(fine_tuning_data["symptoms"].values())
ft_total_causes = sum(fine_tuning_data["root_causes"].values())
inf_total_symptoms = sum(inference_data["symptoms"].values())
inf_total_causes = sum(inference_data["root_causes"].values())

# Create save directory
script_directory = os.path.dirname(os.path.abspath(__file__))
rq5_directory = os.path.join(script_directory, "RQ5")

if not os.path.exists(rq5_directory):
    os.makedirs(rq5_directory)


def create_horizontal_symptom_percentage_stack():
    """Create horizontal percentage stacked chart for symptoms - with x-axis percentage version"""
    # Import formatter
    from matplotlib.ticker import FuncFormatter

    plt.figure(figsize=(16, 5))

    # Original data processing code remains unchanged
    symptoms = list(fine_tuning_data["symptoms"].keys())
    ft_symptom_counts = [fine_tuning_data["symptoms"][s] for s in symptoms]
    inf_symptom_counts = [inference_data["symptoms"][s] for s in symptoms]

    # Percentage calculation and precision correction (keeping previous improvements)
    ft_symptom_pcts = [(count / ft_total_symptoms) * 100 for count in ft_symptom_counts]
    inf_symptom_pcts = [
        (count / inf_total_symptoms) * 100 for count in inf_symptom_counts
    ]

    ft_sum = sum(ft_symptom_pcts)
    inf_sum = sum(inf_symptom_pcts)

    if len(ft_symptom_pcts) > 0:
        ft_symptom_pcts[-1] += 100.0 - ft_sum
    if len(inf_symptom_pcts) > 0:
        inf_symptom_pcts[-1] += 100.0 - inf_sum

    # Original plotting code remains unchanged
    y_labels = ["Fine-tuning", "Inference"]
    y_pos = np.arange(len(y_labels))
    bar_height = 0.6
    colors = ["#c12746", "#f1695d", "#f59700", "#f9cb40", "#d77efc"]

    # Drawing bar chart code remains completely unchanged
    left_ft = 0
    for i, symptom in enumerate(symptoms):
        ft_pct = ft_symptom_pcts[i]
        plt.barh(
            y_pos[0],
            ft_pct,
            left=left_ft,
            height=bar_height,
            color=colors[i],
            alpha=1,
            edgecolor="white",
            linewidth=1.5,
        )

        label_x = left_ft + ft_pct / 2
        if ft_pct >= 2.8:
            plt.text(
                label_x,
                y_pos[0],
                f"{ft_pct:.1f}",
                ha="center",
                va="center",
                fontsize=40,
                fontweight="bold",
                color="white" if ft_pct > 8 else "black",
                fontfamily="Times New Roman",
            )
        left_ft += ft_pct

    left_inf = 0
    for i, symptom in enumerate(symptoms):
        inf_pct = inf_symptom_pcts[i]
        plt.barh(
            y_pos[1],
            inf_pct,
            left=left_inf,
            height=bar_height,
            color=colors[i],
            alpha=1,
            edgecolor="white",
            linewidth=1.5,
        )

        label_x = left_inf + inf_pct / 2
        if inf_pct >= 2.0:
            plt.text(
                label_x,
                y_pos[1],
                f"{inf_pct:.1f}",
                ha="center",
                va="center",
                fontsize=40,
                fontweight="bold",
                color="white" if inf_pct > 8 else "black",
                fontfamily="Times New Roman",
            )
        left_inf += inf_pct

    # Basic style settings remain unchanged
    plt.yticks(
        y_pos, y_labels, fontsize=30, fontweight="bold", fontfamily="Times New Roman"
    )

    # Key addition: define percentage formatter function
    def percent_formatter(x, pos):
        """Convert x-axis ticks to percentage format"""
        return f"{int(x)}%"

    # Apply formatter to x-axis
    plt.gca().xaxis.set_major_formatter(FuncFormatter(percent_formatter))

    # Set tick label font style
    plt.tick_params(axis="x", labelsize=30, labelcolor="black", width=2, length=6)
    plt.tick_params(axis="y", labelsize=30, labelcolor="black", width=2, length=6)

    # Other settings remain unchanged
    legend = plt.legend(
        symptoms,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(symptoms),
        fontsize=30,
        frameon=True,
        fancybox=True,
        shadow=True,
        columnspacing=1.5,
        prop={"family": "Times New Roman", "weight": "bold"},
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)

    plt.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.8)
    plt.xlim(0, 100)
    plt.gca().margins(x=0)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.98)

    plt.savefig(
        os.path.join(rq5_directory, "RQ5-Symptom.pdf"),
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    print("✓ Saved: RQ5-Symptom.pdf")
    plt.close()


def create_horizontal_rootcause_percentage_stack():
    """Create horizontal percentage stacked chart for root causes - fully filled version"""
    # New import: add formatter import at the beginning of function
    from matplotlib.ticker import FuncFormatter

    plt.figure(figsize=(18, 6))

    # Data processing part remains completely unchanged
    all_causes = set(fine_tuning_data["root_causes"].keys()) | set(
        inference_data["root_causes"].keys()
    )
    cause_totals = {
        cause: fine_tuning_data["root_causes"].get(cause, 0)
        + inference_data["root_causes"].get(cause, 0)
        for cause in all_causes
    }
    sorted_causes = sorted(
        cause_totals.keys(), key=lambda x: cause_totals[x], reverse=True
    )

    ft_cause_counts = [fine_tuning_data["root_causes"].get(c, 0) for c in sorted_causes]
    inf_cause_counts = [inference_data["root_causes"].get(c, 0) for c in sorted_causes]

    # Percentage calculation and precision correction part remains completely unchanged
    ft_cause_pcts = [(count / ft_total_causes) * 100 for count in ft_cause_counts]
    inf_cause_pcts = [(count / inf_total_causes) * 100 for count in inf_cause_counts]

    ft_sum = sum(ft_cause_pcts)
    inf_sum = sum(inf_cause_pcts)

    if len(ft_cause_pcts) > 0:
        ft_cause_pcts[-1] += 100.0 - ft_sum
    if len(inf_cause_pcts) > 0:
        inf_cause_pcts[-1] += 100.0 - inf_sum

    # All plotting-related settings remain completely unchanged
    y_labels = ["Fine-tuning", "Inference"]
    y_pos = np.arange(len(y_labels))
    bar_height = 0.9
    colors = [
        "#c12746",
        "#f1695d",
        "#f59700",
        "#f9cb40",
        "#d77efc",
        "#9046cf",
        "#0061b6",
        "#21acb8",
        "#8FCBAD",
    ]

    # Fine-tuning stage drawing code remains completely unchanged
    left_ft = 0
    for i, cause in enumerate(sorted_causes):
        ft_pct = ft_cause_pcts[i]
        plt.barh(
            y_pos[0],
            ft_pct,
            left=left_ft,
            height=bar_height * 0.6,
            color=colors[i],
            alpha=1,
            edgecolor="white",
            linewidth=1.5,
        )

        label_x = left_ft + ft_pct / 2
        if ft_pct >= 1.0:
            plt.text(
                label_x,
                y_pos[0],
                f"{ft_pct:.1f}",
                ha="center",
                va="center",
                fontsize=35,
                fontweight="bold",
                color="white" if ft_pct > 8 else "black",
                fontfamily="Times New Roman",
            )
        left_ft += ft_pct

    # Inference stage drawing code remains completely unchanged
    left_inf = 0
    for i, cause in enumerate(sorted_causes):
        inf_pct = inf_cause_pcts[i]
        plt.barh(
            y_pos[1],
            inf_pct,
            left=left_inf,
            height=bar_height * 0.6,
            color=colors[i],
            alpha=1,
            edgecolor="white",
            linewidth=1.5,
        )

        label_x = left_inf + inf_pct / 2
        if inf_pct >= 1.0:
            plt.text(
                label_x,
                y_pos[1],
                f"{inf_pct:.1f}",
                ha="center",
                va="center",
                fontsize=35,
                fontweight="bold",
                color="white" if inf_pct > 8 else "black",
                fontfamily="Times New Roman",
            )
        left_inf += inf_pct

    # Basic chart style settings remain unchanged
    plt.yticks(
        y_pos, y_labels, fontsize=30, fontweight="bold", fontfamily="Times New Roman"
    )

    # Core new addition: define and apply percentage formatter
    def percent_formatter(x, pos):
        """
        Convert x-axis tick values to percentage format
        This function receives numerical values passed by matplotlib and returns strings with percentage signs
        """
        return f"{int(x)}%"

    # Apply formatter to x-axis - this is the only key new code line
    plt.gca().xaxis.set_major_formatter(FuncFormatter(percent_formatter))

    # Tick parameter settings remain unchanged
    plt.tick_params(axis="x", labelsize=30, labelcolor="black", width=2, length=6)
    plt.tick_params(axis="y", labelsize=30, labelcolor="black", width=2, length=6)

    # Legend settings remain completely unchanged
    legend = plt.legend(
        sorted_causes,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=20,
        frameon=True,
        fancybox=True,
        shadow=True,
        columnspacing=1.2,
        handlelength=1.5,
        prop={"family": "Times New Roman", "weight": "bold"},
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)

    # All layout and save settings remain completely unchanged
    plt.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.8)
    plt.xlim(0, 100)
    plt.gca().margins(x=0)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.05, right=0.98)

    plt.savefig(
        os.path.join(rq5_directory, "RQ5-RootCause.pdf"),
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    print("✓ Saved: RQ5-RootCause.pdf")
    plt.close()


def main():
    """Main function: Generate two key PDF files for RQ5"""
    print("=" * 60)
    print("Starting generation of RQ5 lifecycle stage analysis visualizations")
    print("=" * 60)

    # Generate two key charts
    print("\n🎨 Generating visualization charts...")
    print("   ➤ Creating symptom distribution chart...")
    create_horizontal_symptom_percentage_stack()

    print("   ➤ Creating root cause distribution chart...")
    create_horizontal_rootcause_percentage_stack()

    # Output key statistical information
    print(f"\n📈 Key findings summary:")
    print(
        f"   • Overall failure ratio: Inference stage is {inf_total_symptoms/ft_total_symptoms:.1f} times the Fine-tuning stage"
    )

    print(f"\n✅ All charts successfully saved to: {rq5_directory}")
    print("   📄 Generated files:")
    print("   • RQ5-Symptom.pdf     (Symptom distribution percentage stacked chart)")
    print("   • RQ5-RootCause.pdf   (Root cause distribution percentage stacked chart)")
    print("=" * 60)


if __name__ == "__main__":
    main()
