"""
Wilcoxon signed-rank test comparing per-fold AUC-ROC scores across three models:
  - Model A: Flux Only (theModel_noCentroidInfo.py)
  - Model B: Flux + Centroid (theModel.py, original)
  - Model C: Centroid Only (theModel_centroidOnly.py)

Comparisons:
  1. Model A vs Model B (primary ablation: contribution of centroid channels)
  2. Model A vs Model C (diagnostic: flux vs centroid information value)
"""

import os
import numpy as np
from scipy import stats

# Define file paths
metrics_files = {
    "Model A (Flux Only)": "results_flux_only/metrics_report_fluxonly.txt",
    "Model B (Flux + Centroid)": "results/metrics_report.txt",
    "Model C (Centroid Only)": "results_centroid_only/metrics_report_centroidOnly.txt",
}


def parse_fold_auc_values(file_path):
    """
    Parse per-fold AUC-ROC values from metrics report.

    Looks for the "roc_auc" line in the PER-FOLD RESULTS section.
    Expected format: "roc_auc              0.XXXX  0.XXXX  0.XXXX  0.XXXX  0.XXXX"
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metrics file not found: {file_path}")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the roc_auc line (starts with "roc_auc" and appears after PER-FOLD RESULTS)
    roc_auc_line = None
    for line in lines:
        if line.strip().startswith("roc_auc"):
            roc_auc_line = line.strip()
            break

    if roc_auc_line is None:
        raise ValueError(f"Could not find 'roc_auc' line in {file_path}")

    # Parse the values: "roc_auc" followed by 5 float values
    parts = roc_auc_line.split()
    values = [float(x) for x in parts[1:]]

    if len(values) != 5:
        raise ValueError(
            f"Expected exactly 5 AUC values in {file_path}, got {len(values)}"
        )

    return np.array(values)


def print_results(comparisons, auc_values):
    """Print Wilcoxon test results to stdout and collect lines for file."""
    lines = []

    lines.append("=" * 70)
    lines.append("WILCOXON SIGNED-RANK TEST — PER-FOLD AUC-ROC COMPARISON")
    lines.append("=" * 70)
    lines.append("")

    print("=" * 70)
    print("WILCOXON SIGNED-RANK TEST — PER-FOLD AUC-ROC COMPARISON")
    print("=" * 70)
    print()

    for i, (model_a, model_b, description) in enumerate(comparisons, 1):
        auc_a = auc_values[model_a]
        auc_b = auc_values[model_b]

        # Compute statistics
        mean_a = np.mean(auc_a)
        std_a = np.std(auc_a)
        mean_b = np.mean(auc_b)
        std_b = np.std(auc_b)

        # Run Wilcoxon signed-rank test (two-sided)
        statistic, p_value = stats.wilcoxon(auc_a, auc_b)

        # Determine significance at α = 0.05
        is_significant = p_value < 0.05
        sig_text = (
            "Yes — statistically significant difference (p < 0.05)"
            if is_significant
            else "No — no statistically significant difference (p ≥ 0.05)"
        )

        # Format output
        header = f"Comparison {i}: {description}"
        lines.append(header)
        lines.append("-" * 70)

        result_str = (
            f"  {model_a}:     {mean_a:.4f} ± {std_a:.4f}\n"
            f"  {model_b}: {mean_b:.4f} ± {std_b:.4f}\n"
            f"  Wilcoxon test statistic: {statistic:.1f}\n"
            f"  p-value: {p_value:.4f}\n"
            f"  Significant at α=0.05: {sig_text}"
        )

        print(header)
        print("-" * 70)
        print(result_str)
        print()

        lines.append(result_str)
        lines.append("")

    lines.append("=" * 70)
    lines.append("Notes:")
    lines.append("  - Wilcoxon signed-rank test is non-parametric and suitable for")
    lines.append("    small sample sizes (n=5 folds); tests if distributions differ.")
    lines.append("  - Each comparison uses paired samples (same 5 cross-validation folds).")
    lines.append("=" * 70)

    print("=" * 70)
    print("Notes:")
    print("  - Wilcoxon signed-rank test is non-parametric and suitable for")
    print("    small sample sizes (n=5 folds); tests if distributions differ.")
    print("  - Each comparison uses paired samples (same 5 cross-validation folds).")
    print("=" * 70)

    return lines


def main():
    # Load AUC values
    print("Loading per-fold AUC-ROC values from metrics reports...\n")

    auc_values = {}
    try:
        for model_name, file_path in metrics_files.items():
            auc_values[model_name] = parse_fold_auc_values(file_path)
            mean = np.mean(auc_values[model_name])
            std = np.std(auc_values[model_name])
            print(f"  {model_name}: {mean:.4f} ± {std:.4f}")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure all three model scripts have been run and their")
        print("metrics reports have been saved to the expected directories:")
        for name, path in metrics_files.items():
            print(f"  - {path}")
        return 1

    print()

    # Define comparisons
    comparisons = [
        (
            "Model A (Flux Only)",
            "Model B (Flux + Centroid)",
            "Primary: Centroid Information Ablation",
        ),
        (
            "Model A (Flux Only)",
            "Model C (Centroid Only)",
            "Diagnostic: Flux vs Centroid",
        ),
    ]

    # Generate and print results
    result_lines = print_results(comparisons, auc_values)

    # Save results to file
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "wilcoxon_results.txt")

    with open(output_file, 'w') as f:
        f.write("\n".join(result_lines))

    print(f"✓ Results saved to: {output_file}\n")
    return 0


if __name__ == "__main__":
    exit(main())
