import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Set global font to Times New Roman, bold
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

# Aggregated data from the previous step
# (Treating ECI, CPSE, etc., as the features for "Root Cause" style correlation)
data_aggregated_features = {
    "Series": ["DeepSeek\nSeries", "Llama\nSeries", "Qwen\nSeries"],
    "ECI": [18, 45, 160],
    "CPSE": [15, 27, 88],
    "DCI": [14, 6, 51],
    "AUII": [12, 9, 44],
    "DPCI": [5, 28, 21],
    "TTPI": [5, 16, 30],
    "HRL": [6, 24, 17],
    "AID": [9, 6, 31],
    "MCI": [2, 10, 7],
}
df_original_agg_features = pd.DataFrame(data_aggregated_features)
df_original_agg_features = df_original_agg_features.set_index("Series")

# Transpose DataFrame for series correlation:
# Model series become columns, Feature types (ECI, CPSE, etc.) become rows
df_for_series_agg_feature_correlation = df_original_agg_features.transpose()

print("Data for Series Correlation (Aggregated Features as rows, Series as columns):")
print(df_for_series_agg_feature_correlation)
print("-" * 50)

# Calculate correlation matrix between model series based on the aggregated feature distribution
series_agg_feature_correlation_matrix = df_for_series_agg_feature_correlation.corr(
    method="pearson"
)

print(
    "\nCorrelation Matrix between Model Series (based on aggregated feature distribution):"
)
print(series_agg_feature_correlation_matrix)
print("-" * 50)

# Remove index and column names from correlation matrix
series_agg_feature_correlation_matrix.index.name = None
series_agg_feature_correlation_matrix.columns.name = None

# Plotting
plt.figure(figsize=(7, 5))  # Adjusted size to match example

# Create a mask to hide the upper triangle (excluding the diagonal)
mask_agg_features = np.triu(
    np.ones_like(series_agg_feature_correlation_matrix, dtype=bool), k=1
)

sns.heatmap(
    series_agg_feature_correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    center=0,
    mask=mask_agg_features,
    cbar=False,  # Remove color bar
    annot_kws={
        "fontfamily": "Times New Roman",
        "weight": "bold",
        "size": 35,
    },  # Adjusted font size
)

# X-axis labels are Series names
plt.xticks(
    rotation=45,
    ha="right",
    fontfamily="Times New Roman",
    fontweight="bold",
    fontsize=30,  # Adjusted font size
)
# Y-axis labels are Series names (from the correlation matrix index)
plt.yticks(
    rotation=0, fontfamily="Times New Roman", fontweight="bold", fontsize=30
)  # Adjusted font size

# Remove axis titles that might be automatically added by matplotlib
plt.xlabel("")
plt.ylabel("")

plt.tight_layout()

# Saving to PDF in RQ4 directory
script_directory = os.getcwd()
rq4_directory = os.path.join(script_directory, "RQ4")

if not os.path.exists(rq4_directory):
    os.makedirs(rq4_directory)
    print(f"Directory created: {rq4_directory}")
else:
    print(f"Directory already exists: {rq4_directory}")

pdf_filename_agg_features = "RQ4-RootCause.pdf"
pdf_save_path_agg_features = os.path.join(rq4_directory, pdf_filename_agg_features)

plt.savefig(pdf_save_path_agg_features, format="pdf")
print(
    f"\nAggregated features correlation heatmap saved to {pdf_save_path_agg_features}"
)

# plt.show() # Uncomment to display plot if running locally
