import json
import os
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置Times New Roman字体支持
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# 目标JSON文件名列表
TARGET_FILENAMES = [
    "DeepSeek-Coder.json",
    "DeepSeek-Coder-V2.json",
    "DeepSeek-Math.json",
    "DeepSeek-MoE.json",
    "DeepSeek-R1.json",
    "DeepSeek-V2.json",
    "DeepSeek-V3.json",
    "meta-llama.json",
    "meta-llama3.json",
    "Qwen.json",
    "Qwen2dot5.json",
    "Qwen2dot5-Coder.json",
    "Qwen2dot5-Math.json",
]


def extract_top_level_root_cause(root_cause_text):
    """提取根因的顶层分类"""
    if not root_cause_text or not isinstance(root_cause_text, str):
        return None

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

    first_level = root_cause_text.split(" - ")[0].strip()

    if first_level in target_categories:
        return first_level
    else:
        return None


def load_json_files():
    """加载所有目标JSON文件并返回数据列表"""
    script_dir = Path(__file__).parent
    all_json_data = []

    print(f"开始在目录 {script_dir} 及其子目录中搜索目标JSON文件...")

    for target_name in TARGET_FILENAMES:
        for filepath in script_dir.rglob(target_name):
            print(f"  找到并读取文件: {filepath}")
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
                        # 尝试按JSON Lines格式解析
                        for line in content.splitlines():
                            line = line.strip()
                            if line:
                                try:
                                    item = json.loads(line)
                                    all_json_data.append(item)
                                except json.JSONDecodeError:
                                    pass
            except Exception as e:
                print(f"    读取文件 {filepath} 时发生错误: {e}")

    print(f"\n总共加载了 {len(all_json_data)} 条JSON记录。\n")
    return all_json_data


def analyze_model_root_cause_counts(data_list):
    """分析按模型分组的根因统计"""
    model_root_cause_counts = defaultdict(Counter)

    for item in data_list:
        model = item.get("model")
        root_cause_text = item.get("ROOT CAUSE")

        if not model or not root_cause_text:
            continue

        top_level_root_cause = extract_top_level_root_cause(root_cause_text)
        if top_level_root_cause:
            model_root_cause_counts[model][top_level_root_cause] += 1

    return model_root_cause_counts


def create_output_directory():
    """创建RQ6输出目录"""
    script_dir = Path(__file__).parent
    output_dir = script_dir / "RQ6"
    output_dir.mkdir(exist_ok=True)
    print(f"输出目录已准备: {output_dir}")
    return output_dir


def create_model_root_cause_stacked_chart(model_root_cause_counts, output_dir):
    """创建专业级分组模型根因分布图 - 严格按照指定的6个核心模型"""
    print("  生成专业级分组模型根因分布图（严格限定6个核心模型）...")

    if not model_root_cause_counts:
        print("    警告: 没有模型根因数据")
        return

    # 定义9个专业根因类别的固定顺序
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

    def find_exact_model_match(target_model, all_models):
        """精确匹配目标模型"""
        for model_name in all_models:
            model_lower = model_name.lower()

            if target_model == "DeepSeek-V3":
                if "deepseek" in model_lower and (
                    "v3" in model_lower or "v-3" in model_lower
                ):
                    return model_name
            elif target_model == "DeepSeek-R1":
                if "deepseek" in model_lower and (
                    "r1" in model_lower or "r-1" in model_lower
                ):
                    return model_name
            elif target_model == "Llama":
                if (
                    "llama" in model_lower and "3" not in model_lower
                ) or "codellama" in model_lower:
                    return model_name
            elif target_model == "Llama3":
                if "llama3" in model_lower and "codellama" not in model_lower:
                    return model_name
            elif target_model == "Qwen":
                if (
                    "qwen" in model_lower
                    and "2.5" not in model_lower
                    and "2dot5" not in model_lower
                ):
                    return model_name
            elif target_model == "Qwen2.5":
                if "qwen2.5" in model_lower or "qwen2dot5" in model_lower:
                    return model_name
        return None

    # 按照指定顺序定义6个目标模型
    target_models_ordered = [
        "DeepSeek-V3",
        "DeepSeek-R1",
        "Llama",
        "Llama3",
        "Qwen",
        "Qwen2.5",
    ]

    # 查找匹配的模型
    found_models = []
    found_display_names = []
    found_data = []

    all_model_names = list(model_root_cause_counts.keys())

    for target_model in target_models_ordered:
        matched_model = find_exact_model_match(target_model, all_model_names)
        if matched_model:
            found_models.append(matched_model)
            found_display_names.append(target_model)
            found_data.append(model_root_cause_counts[matched_model])
            print(f"    ✓ 找到模型: {target_model} -> {matched_model}")
        else:
            print(f"    ✗ 未找到模型: {target_model}")

    if not found_models:
        print("    警告: 没有找到任何匹配的目标模型")
        return

    # 为了让模型从上到下显示，需要反转顺序
    found_models.reverse()
    found_display_names.reverse()
    found_data.reverse()

    # 为每个模型准备数据矩阵
    data_matrix = []
    for root_causes in found_data:
        model_data = []
        for root_cause in root_cause_categories:
            count = root_causes.get(root_cause, 0)
            model_data.append(count)
        data_matrix.append(model_data)

    data_array = np.array(data_matrix)

    # 专业配色方案
    colors = [
        "#c12746",  # ECI - 深红色
        "#f1695d",  # CPSE - 橙红色
        "#f59700",  # DCI - 橙色
        "#f9cb40",  # AUII - 黄色
        "#d77efc",  # DPCI - 浅紫色
        "#9046cf",  # TTPI - 深紫色
        "#0061b6",  # HRL - 蓝色
        "#21acb8",  # AID - 青色
        "#8FCBAD",  # MCI - 浅绿色
    ]

    # 使用统一的条形高度
    uniform_height = 0.8
    adaptive_heights = [uniform_height] * len(found_models)

    # 创建图表
    fig, ax = plt.subplots(figsize=(18, 8))

    # 计算y轴位置
    y_positions = []
    cumulative_y = 0
    for i, height in enumerate(adaptive_heights):
        y_positions.append(cumulative_y + height / 2)
        cumulative_y += height + 0.1

    left_positions = np.zeros(len(found_models))

    # 添加网格
    ax.grid(axis="x", alpha=0.3, linestyle="-", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # 添加红色虚线分组分隔线
    group_boundaries = [1.5, 3.5]
    for boundary_idx in group_boundaries:
        if boundary_idx < len(y_positions):
            if boundary_idx == 1.5:
                boundary_y = y_positions[1] + adaptive_heights[1] / 2 + 0.05
            elif boundary_idx == 3.5:
                boundary_y = y_positions[3] + adaptive_heights[3] / 2 + 0.05

            ax.axhline(
                y=boundary_y,
                color="red",
                linestyle="--",
                alpha=0.7,
                linewidth=2.0,
                zorder=1,
            )

    # 绘制堆叠条形
    for root_cause_idx, root_cause_name in enumerate(root_cause_categories):
        root_cause_values = data_array[:, root_cause_idx]

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

        bars = []
        for i, (y_pos, height, value) in enumerate(
            zip(y_positions, adaptive_heights, root_cause_values)
        ):
            if value > 0:
                bar = ax.barh(
                    y_pos,
                    value,
                    left=left_positions[i],
                    height=height,
                    color=colors[root_cause_idx],
                    label=legend_labels[root_cause_idx] if i == 0 else "",
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=2,
                )
                bars.append((bar, i, value, y_pos, height))

        # 添加数值标签
        for bar_info in bars:
            bar, i, value, y_pos, height = bar_info
            if value >= 2:
                label_x = left_positions[i] + value / 2
                ax.text(
                    label_x,
                    y_pos,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white",
                    fontsize=25,
                    fontfamily="Times New Roman",
                    zorder=3,
                )

        left_positions += root_cause_values

    # 设置Y轴标签
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        found_display_names,
        fontweight="bold",
        fontsize=35,
        fontfamily="Times New Roman",
    )

    # X轴优化
    ax.tick_params(axis="x", labelsize=35)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
        label.set_fontfamily("Times New Roman")

    # 优化图例布局
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    legend = ax.legend(
        unique_handles,
        unique_labels,
        loc="upper right",
        ncol=1,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        edgecolor="gray",
        facecolor="white",
        prop={"family": "Times New Roman", "weight": "bold", "size": 23},
    )

    legend.set_zorder(100)
    legend.get_frame().set_zorder(100)

    plt.tight_layout()
    plt.savefig(
        output_dir / "RQ6-RootCause.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    print(f"    已生成专业级分组堆叠图: RQ6-RootCause.pdf")
    print(f"    模型顺序（从上到下）: {' -> '.join(reversed(found_display_names))}")


def main():
    """主函数：生成RQ6-RootCause.pdf"""
    print("开始生成RQ6-RootCause.pdf文件...\n")

    # 1. 加载数据
    all_data = load_json_files()
    if not all_data:
        print("没有找到任何数据，程序退出。")
        return

    # 2. 创建输出目录
    output_dir = create_output_directory()

    # 3. 分析模型根因统计
    model_root_cause_counts = analyze_model_root_cause_counts(all_data)

    if not model_root_cause_counts:
        print("没有找到有效的根因数据，程序退出。")
        return

    # 4. 生成RQ6-RootCause.pdf
    create_model_root_cause_stacked_chart(model_root_cause_counts, output_dir)

    print(f"\n✓ 成功生成文件: {output_dir}/RQ6-RootCause.pdf")


if __name__ == "__main__":
    main()
