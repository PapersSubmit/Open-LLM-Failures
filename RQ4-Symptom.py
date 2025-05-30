import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os  # Import os module to handle file paths and directories

# Set global font to Times New Roman, bold
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"  # Axis labels also bold
plt.rcParams["axes.titleweight"] = "bold"  # Titles also bold


# Original data (model series as rows, bug types as columns)
data_original = {
    "Series": ["DeepSeek\nSeries", "Llama\nSeries", "Qwen\nSeries"],
    "Crash": [25, 85, 185],
    "Incorrect Functionality": [41, 32, 167],
    "Loading Failure": [17, 45, 75],
    "Hang": [2, 9, 8],
    "Poor Performance": [1, 0, 14],
}
df_original = pd.DataFrame(data_original)

# To analyze correlations between model series, we need to process the DataFrame
# so that model series become columns and bug types become rows
series_column_values = df_original["Series"].values  # Extract model series names
df_numeric_data = df_original.drop("Series", axis=1)  # Extract numeric data

df_for_series_correlation = df_numeric_data.transpose()  # Transpose numeric data
df_for_series_correlation.columns = (
    series_column_values  # Set model series names as column names
)

# Calculate correlation matrix between model series (Pearson correlation)
series_correlation_matrix = df_for_series_correlation.corr(method="pearson")

# Remove index and column names from correlation matrix
series_correlation_matrix.index.name = None
series_correlation_matrix.columns.name = None

# # Modify tick labels to remove "\nSeries" # <--- This code section has been removed to restore original labels
# new_tick_labels = [label.split("\n")[0] for label in series_correlation_matrix.index]
# series_correlation_matrix.index = new_tick_labels
# series_correlation_matrix.columns = new_tick_labels


# --- Heatmap plotting modifications ---
plt.figure(figsize=(7, 5))

# Create a mask to hide the upper triangle (excluding diagonal k=0)
# np.triu returns an upper triangular matrix, parts with True values will be masked
mask = np.triu(
    np.ones_like(series_correlation_matrix, dtype=bool), k=1
)  # k=1 means excluding diagonal

# Draw heatmap
# annot_kws is used to set font properties for annotation text
sns.heatmap(
    series_correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    center=0,
    mask=mask,  # Apply mask
    cbar=False,  # Remove color bar (legend)
    annot_kws={
        "fontfamily": "Times New Roman",
        "weight": "bold",
        "size": 35,
    },  # Annotation font
)

# X-axis and Y-axis labels will directly use DataFrame's index/column names
# (now "DeepSeek\nSeries", "Llama\nSeries", "Qwen\nSeries")
# The following code sets font and rotation for tick labels
plt.xticks(
    rotation=45,
    ha="right",
    fontfamily="Times New Roman",
    fontweight="bold",
    fontsize=30,
)
plt.yticks(rotation=0, fontfamily="Times New Roman", fontweight="bold", fontsize=30)

# Remove axis titles that might be automatically added by matplotlib (usually won't appear when DataFrame's index/columns name is None)
# But to ensure, explicitly set to empty
plt.xlabel("")
plt.ylabel("")

plt.tight_layout()
# --- End of plotting modifications ---

# --- Save as PDF to RQ4 directory ---
script_directory = os.getcwd()
rq4_directory = os.path.join(script_directory, "RQ4")

if not os.path.exists(rq4_directory):
    os.makedirs(rq4_directory)
    print(f"Directory created: {rq4_directory}")
else:
    print(f"Directory already exists: {rq4_directory}")

pdf_filename = "RQ4-Symptom.pdf"
pdf_save_path = os.path.join(rq4_directory, pdf_filename)

plt.savefig(pdf_save_path, format="pdf")
print(f"\nCustomized heatmap saved to {pdf_save_path}")
# --- End of save code ---

# If you want to display the image after script execution (e.g., in Jupyter Notebook or local IDE), uncomment the line below
# plt.show()
