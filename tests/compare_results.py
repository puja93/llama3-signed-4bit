#!/opt/anaconda3/bin/python3
"""
Compare Results: Analyzes and computes accuracy retention between two benchmark JSON files.
Usage:
    python3 tests/compare_results.py [regular_json] [compressed_json]
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "tests" / "results"


def main():
    reg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "benchmark_regular_latest.json"
    comp_path = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS_DIR / "benchmark_compressed_latest.json"

    if not reg_path.exists():
        print(f"Error: Regular benchmark file not found: {reg_path}", file=sys.stderr)
        return 1
    if not comp_path.exists():
        print(f"Error: Compressed benchmark file not found: {comp_path}", file=sys.stderr)
        return 1

    with open(reg_path) as f:
        reg_data = json.load(f)
    with open(comp_path) as f:
        comp_data = json.load(f)

    reg_table = reg_data.get("original_1b_bf16", reg_data).get("results", {})
    comp_table = comp_data.get("signed_4bit_engine", comp_data).get("results", {})

    print("\033[1;36m" + "=" * 96)
    print("  HEAD-TO-HEAD COMPARISON REPORT")
    print(f"  Regular Baseline:    {reg_path.name}")
    print(f"  Compressed 4-Bit:    {comp_path.name}")
    print("=" * 96 + "\033[0m")
    print(f"{'Task':<22} | {'Metric':<18} | {'Regular (BF16)':<18} | {'Compressed (4-Bit)':<18} | {'Retention / Delta':<16}")
    print("-" * 102)

    all_tasks = sorted(list(set(list(reg_table.keys()) + list(comp_table.keys()))))
    retention_scores = []

    for task in all_tasks:
        reg_m = reg_table.get(task, {})
        comp_m = comp_table.get(task, {})
        common_metrics = [m for m in reg_m.keys() if m in comp_m and isinstance(reg_m[m], (int, float)) and not m.endswith("_stderr")]

        for m in common_metrics:
            v_reg = reg_m[m]
            v_comp = comp_m[m]
            is_pct = "acc" in m or "exact_match" in m or "mc2" in m

            if is_pct:
                str_reg = f"{v_reg * 100:.2f}%"
                str_comp = f"{v_comp * 100:.2f}%"
                retention = (v_comp / v_reg * 100.0) if v_reg > 0 else 0.0
                retention_scores.append(retention)
                str_delta = f"{retention:.1f}% recovery"
            elif "word_perplexity" in m or "perplexity" in m:
                str_reg = f"{v_reg:.2f} PPL"
                str_comp = f"{v_comp:.2f} PPL"
                delta_ppl = v_comp - v_reg
                sign = "+" if delta_ppl >= 0 else ""
                str_delta = f"{sign}{delta_ppl:.2f} PPL"
            else:
                str_reg = f"{v_reg:.4f}"
                str_comp = f"{v_comp:.4f}"
                str_delta = f"{v_comp - v_reg:+.4f}"

            print(f"{task:<22} | {m:<18} | {str_reg:<18} | {str_comp:<18} | {str_delta:<16}")

    print("-" * 102)
    if retention_scores:
        avg_ret = sum(retention_scores) / len(retention_scores)
        status_color = "\033[1;32m" if avg_ret >= 98.0 else "\033[1;33m"
        print(f"{'MEAN ACCURACY RETENTION':<43} | {status_color}{avg_ret:.2f}%\033[0m")
    print("=" * 96 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
