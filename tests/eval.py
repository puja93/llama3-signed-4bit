#!/opt/anaconda3/bin/python3
"""
CLI Evaluation Runner using EleutherAI lm-evaluation-harness.
Supports evaluating:
1. Native Signed 4-Bit ARM NEON SIMD Engine (Signed4BitLM)
2. Original LLaMA-3.2-1B (BF16 model.safetensors baseline)
3. Head-to-head comparison ('both') with automated accuracy retention calculation.
"""

import sys
import argparse
import json
import time
import gc
from pathlib import Path

# Add project root to sys.path
TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from lm_eval import simple_evaluate

try:
    from tests.lm_eval_adapter import Signed4BitLM
except ImportError:
    from lm_eval_adapter import Signed4BitLM

try:
    from lm_eval.models.huggingface import HFLM
except ImportError:
    HFLM = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LM Evaluation Harness on LLaMA-3.2 1B (Signed 4-Bit vs Original BF16)."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["signed_4bit", "hf", "both"],
        default="signed_4bit",
        help="Model to evaluate: 'signed_4bit', 'hf' (original BF16), or 'both' (side-by-side comparison)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_challenge,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k",
        help="Comma-separated list of tasks to evaluate (default: arc_challenge,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k - add wikitext if perplexity is needed)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of examples per task (useful for quick checks)",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (defaults to task specification)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(ROOT_DIR / "models" / "llama3_signed_4bit_b128.bin"),
        help="Path to the signed 4-bit model binary",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=str(ROOT_DIR / "models"),
        help="Directory containing config.json and model.safetensors for baseline model",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=str(ROOT_DIR / "models" / "tokenizer.json"),
        help="Path to tokenizer.json",
    )
    parser.add_argument(
        "--hf_device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device for baseline model evaluation (mps, cuda, or cpu)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save evaluation results as JSON",
    )
    return parser.parse_args()


def run_eval_for_model(model_instance, task_list, limit, num_fewshot):
    combined_results = {"results": {}, "configs": {}, "versions": {}}
    total_tasks = len(task_list)

    for idx, t in enumerate(task_list, 1):
        print(f"\n\033[1;33m[{idx}/{total_tasks}] Evaluating task: {t} (limit={limit if limit is not None else 'full'})...\033[0m", flush=True)
        t_start_task = time.time()
        try:
            res = simple_evaluate(
                model=model_instance,
                tasks=[t],
                limit=limit,
                num_fewshot=num_fewshot,
                batch_size=1,
            )
            elapsed = time.time() - t_start_task
            res_dict = res.to_dict() if hasattr(res, "to_dict") else res
            task_res = res_dict.get("results", {})
            combined_results["results"].update(task_res)
            combined_results["configs"].update(res_dict.get("configs", {}))

            # Live score display
            for sub_name, metrics in task_res.items():
                for m_name, val in metrics.items():
                    if isinstance(val, (int, float)) and not m_name.endswith("_stderr"):
                        v_str = f"{val * 100:.2f}%" if ("acc" in m_name or "exact_match" in m_name or "mc2" in m_name) else f"{val:.4f}"
                        print(f"   \033[0;32m✓ {sub_name} | {m_name}: {v_str}\033[0m", flush=True)
            print(f"   \033[0;36m• Finished {t} in {elapsed:.1f}s\033[0m\n", flush=True)
        except Exception as e:
            print(f"   \033[1;31m✗ Error evaluating {t}: {e}\033[0m", flush=True)

    return combined_results


def print_single_results(results, model_name):
    print("\n\033[1;32m" + "=" * 80)
    print(f"  EVALUATION RESULTS: {model_name}")
    print("=" * 80 + "\033[0m")

    res_dict = results.to_dict() if hasattr(results, "to_dict") else results
    table_results = res_dict.get("results", {})

    print(f"{'Task':<28} | {'Metric':<22} | {'Value':<12}")
    print("-" * 80)
    for task_name, metrics in table_results.items():
        for metric_name, val in metrics.items():
            if isinstance(val, (int, float)) and not metric_name.endswith("_stderr"):
                formatted_val = f"{val * 100:.2f}%" if ("acc" in metric_name or "exact_match" in metric_name) else f"{val:.4f}"
                print(f"{task_name:<28} | {metric_name:<22} | {formatted_val:<12}")


def print_comparison_results(hf_results, signed_results):
    print("\n\033[1;36m" + "=" * 90)
    print("  HEAD-TO-HEAD COMPARISON: ORIGINAL 1B (BF16) VS SIGNED 4-BIT ENGINE")
    print("=" * 90 + "\033[0m")

    hf_dict = hf_results.to_dict() if hasattr(hf_results, "to_dict") else hf_results
    sgn_dict = signed_results.to_dict() if hasattr(signed_results, "to_dict") else signed_results

    hf_table = hf_dict.get("results", {})
    sgn_table = sgn_dict.get("results", {})

    print(f"{'Task':<22} | {'Metric':<18} | {'Original 1B (BF16)':<18} | {'Signed 4-Bit':<14} | {'Retention / Delta':<16}")
    print("-" * 96)

    all_tasks = sorted(list(set(list(hf_table.keys()) + list(sgn_table.keys()))))
    for task in all_tasks:
        hf_m = hf_table.get(task, {})
        sgn_m = sgn_table.get(task, {})
        common_metrics = [m for m in hf_m.keys() if m in sgn_m and isinstance(hf_m[m], (int, float)) and not m.endswith("_stderr")]

        for m in common_metrics:
            v_hf = hf_m[m]
            v_sgn = sgn_m[m]
            is_pct = "acc" in m or "exact_match" in m or "mc2" in m

            if is_pct:
                str_hf = f"{v_hf * 100:.2f}%"
                if v_hf == 0.0 and v_sgn == 0.0:
                    str_delta = "Tie (Both 0.0%)"
                elif v_hf > 0:
                    retention = (v_sgn / v_hf * 100.0)
                    str_delta = f"{retention:.1f}% recovery"
                else:
                    str_delta = f"+{v_sgn * 100:.1f}%"
            elif "word_perplexity" in m or "perplexity" in m:
                str_hf = f"{v_hf:.2f} PPL"
                str_sgn = f"{v_sgn:.2f} PPL"
                delta_ppl = v_sgn - v_hf
                sign = "+" if delta_ppl >= 0 else ""
                str_delta = f"{sign}{delta_ppl:.2f} PPL"
            else:
                str_hf = f"{v_hf:.4f}"
                str_sgn = f"{v_sgn:.4f}"
                str_delta = f"{v_sgn - v_hf:+.4f}"

            print(f"{task:<22} | {m:<18} | {str_hf:<18} | {str_sgn:<14} | {str_delta:<16}")

    print("=" * 96 + "\n")


def main():
    args = parse_args()
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print("\033[1;36m================================================================================")
    print("  LLaMA-3.2 1B - ELEUTHERAI LM EVALUATION HARNESS")
    print(f"  Target: {args.model_type.upper()}")
    print(f"  Tasks: {', '.join(task_list)}")
    print(f"  Limit per task: {args.limit if args.limit else 'Full Benchmark'}")
    print("================================================================================\033[0m\n")

    t_global_start = time.time()
    all_output_data = {}

    if args.model_type in ["hf", "both"]:
        if HFLM is None:
            print("Error: transformers / lm_eval HFLM could not be imported.", file=sys.stderr)
            return 1

        print("\033[1;33m[Step] Loading Original Model (BF16 model.safetensors) via HFLM on " + args.hf_device.upper() + "...\033[0m")
        t0 = time.time()
        hf_model = HFLM(
            pretrained=args.models_dir,
            dtype="bfloat16" if args.hf_device in ["mps", "cuda"] else "float32",
            device=args.hf_device,
            batch_size=1,
        )
        print(f" • Baseline Model ready in {time.time() - t0:.2f}s. Evaluating...")
        hf_results = run_eval_for_model(hf_model, task_list, args.limit, args.num_fewshot)
        all_output_data["original_1b_bf16"] = hf_results.to_dict() if hasattr(hf_results, "to_dict") else hf_results

        if args.model_type == "hf":
            print_single_results(hf_results, "Original LLaMA-3.2-1B (BF16)")

        del hf_model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if args.model_type in ["signed_4bit", "both"]:
        print("\033[1;33m[Step] Initializing Signed 4-Bit Native Engine (ARM NEON SIMD)...\033[0m")
        t0 = time.time()
        signed_model = Signed4BitLM(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            batch_size=1,
        )
        print(f" • Signed 4-bit Engine ready in {time.time() - t0:.2f}s. Evaluating...")
        signed_results = run_eval_for_model(signed_model, task_list, args.limit, args.num_fewshot)
        all_output_data["signed_4bit_engine"] = signed_results.to_dict() if hasattr(signed_results, "to_dict") else signed_results

        if args.model_type == "signed_4bit":
            print_single_results(signed_results, "LLaMA-3.2 Signed 4-Bit Engine (626 MB)")

    if args.model_type == "both":
        print_comparison_results(hf_results, signed_results)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(all_output_data, f, indent=2, default=str)
        print(f"[Saved full results to {args.output_json}]")

    print(f"\n\033[0;37m[Total evaluation run completed in {time.time() - t_global_start:.1f}s]\033[0m\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
