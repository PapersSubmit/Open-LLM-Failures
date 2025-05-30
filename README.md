# Open-LLM-Failures

## 📖 Overview

This repository contains a dataset of **706 failure cases** from open-source Large Language Models, collected from GitHub repositories of DeepSeek, Llama, and Qwen series. The study analyzes patterns in how LLMs fail during fine-tuning and inference stages.

## 🎯 Research Questions

The analysis addresses six questions about LLM reliability:

1. **RQ1**: What symptoms do users experience when LLMs fail?
2. **RQ2**: What technical root causes drive these failures?
3. **RQ3**: How do symptoms relate to underlying technical problems?
4. **RQ4**: Are failure patterns universal across models or model-specific?
5. **RQ5**: How do failures differ between fine-tuning and inference stages?
6. **RQ6**: How have failure patterns evolved across model versions?

## 🗂️ Data Structure

```
Open-LLM-Failures/
│
├── README.md                           # Dataset documentation
├── LICENSE                             # Usage terms
│
├── DeepSeek/                           # DeepSeek series failures (86 cases)
│   ├── R1/DeepSeek-R1.json            # Latest model variant
│   ├── V3/DeepSeek-V3.json            # Previous version  
│   ├── V2/DeepSeek-V2.json            # Earlier version
│   ├── Coder/DeepSeek-Coder.json      # Code-specialized model
│   ├── Coder-V2/DeepSeek-Coder-V2.json
│   ├── Math/DeepSeek-Math.json        # Math-specialized model
│   └── MoE/DeepSeek-MoE.json          # Mixture-of-Experts architecture
│
├── meta-llama/                         # Llama series failures (161 cases)
│   ├── llama3/meta-llama3.json        # Latest release
│   ├── llama/meta-llama.json          # Base Llama 2 series
│   └── codellama/meta-codellama.json  # Code-specialized variant
│
├── QwenLM/                             # Qwen series failures (449 cases)
│   ├── Qwen2.5/Qwen2dot5.json         # Latest major version
│   ├── Qwen/Qwen.json                 # Original version
│   ├── Qwen2.5-Coder/Qwen2dot5-Coder.json
│   └── Qwen2.5-Math/Qwen2dot5-Math.json
│
├── RQ1.py - RQ6.py                     # Analysis scripts
├── RQ1/ - RQ6/                         # Generated visualizations and results
```

## 📊 Data Format

Each failure record contains:

```json
{
  "number": "[GitHub issue number]",
  "link": "[Direct URL to original GitHub issue]",
  "status": "[Issue resolution status]",
  "SYMPTOM": "[Category - Subcategory - Specific Manifestation]",
  "ROOT CAUSE": "[Technical Category - Specific Technical Cause]",
  "STAGE": "[Fine-tuning/Inference]",
  "model": "[Specific model name]",
  "architecture": "[Dense/MoE/Mixed]"
}
```

The hierarchical SYMPTOM and ROOT CAUSE fields use a multi-level taxonomy, enabling analysis at different levels of granularity. Each record links to the original GitHub discussion for full context.

## 🔍 Key Findings

### Symptom Patterns
Crashes are the most common user-experienced failure, followed by incorrect functionality issues. Generation quality problems, particularly meaningless output generation, represent the most frequent specific complaint.

### Root Cause Analysis
Environment compatibility issues cause the majority of failures, primarily due to complex software dependencies and hardware requirements. Configuration and parameter setting errors rank as the second major cause.

### Cross-Model Comparison
Users experience similar symptoms across different LLM series, but the underlying technical causes vary significantly between model architectures. This suggests that symptom-based troubleshooting can be generalized, while solutions require model-specific approaches.

### Lifecycle Stages
Fine-tuning failures are predominantly crashes caused by configuration complexity. Inference failures show a more balanced distribution between crashes and functional problems.

### Temporal Changes
Failure patterns shift as models evolve. Newer versions show fewer basic crashes but more issues related to advanced features like quantization and distributed inference.

## 🚀 Using the Dataset

### Data Organization
Failure data is organized by model series and variants, enabling comparisons within model families and across different architectures. 

### Analysis Scripts
Each research question has a corresponding Python script that generates visualizations from the structured data. Scripts can be run independently to focus on specific aspects of LLM reliability.

## 💡 Applications

The dataset serves different research and practical needs:

**Model Development**: Identifies that environment compatibility and configuration complexity create more barriers than algorithmic issues.

**User Support**: Shows that troubleshooting knowledge transfers between models at the symptom level, though fixes may be model-specific.

**Research**: Documents the evolution from basic stability issues to more complex behavioral problems as the field matures.

## 🔮 Future Work

The dataset enables research into:
- Automated failure prediction systems
- Model-specific debugging tools
- Proactive reliability monitoring
- Longitudinal reliability trend analysis