"""Convert eval result JSON files to a flat CSV for analysis and graphing.

Usage:
  python convert_to_csv.py                        # scans all known result dirs
  python convert_to_csv.py --dirs "Claude results" "Claude Haiku Results"
  python convert_to_csv.py --files "Claude results/level-1/quadratic_programming__quadratic_programming__claude-sonnet-4-6.json" "Claude Haiku Results/level-1/quadratic_programming__quadratic_programming__claude-haiku-4-5-20251001.json"
  python convert_to_csv.py --output results.csv
"""

import csv
import json
import os
import argparse

DEFAULT_DIRS = [
    "Claude results",
    "Claude Haiku Results",
]

# Strip model suffix from filename to get module name
MODEL_SUFFIXES = [
    "__claude-sonnet-4-6.json",
    "__claude-haiku-4-5-20251001.json",
    "__gpt-5.5.json",
    "__gpt-5.4.json",
    "__gpt-5.json",
    "__gpt-4.1.json",
    "__gpt-4.1-mini.json",
    "__gpt-4o-mini.json",
    "__gemini-2.5-pro.json",
    "__gemini-2.5-flash.json",
    "__gemini-2.0-flash.json",
    "__deepseek-chat.json",
    "__deepseek-reasoner.json",
]

MODULE_LABELS = {
    "geometric__basic_geometric_programming": "GP",
    "linear_programming__non_trivial_linear_programming": "LP",
    "quadratic_constrained_quadratic_programming__basic_qcqp": "QCQP",
    "quadratic_programming__quadratic_programming": "QP",
    "semidefinite_programming__basic_semidefinite_programming": "SDP",
}

COLUMNS = [
    "model",
    "level",
    "module",
    "module_short",
    "problem_index",
    "expected_answer",
    "model_answer",
    "answered",
    "correct",
    "finish_reason",
    "input_tokens",
    "output_tokens",
]


def strip_suffix(fname):
    for suffix in MODEL_SUFFIXES:
        if fname.endswith(suffix):
            return fname[: -len(suffix)]
    return fname[:-5] if fname.endswith(".json") else fname


def load_dir(base_dir):
    rows = []
    base = os.path.expanduser(base_dir)
    if not os.path.isdir(base):
        return rows
    for level_name in sorted(os.listdir(base)):
        level_path = os.path.join(base, level_name)
        if not os.path.isdir(level_path) or not level_name.startswith("level-"):
            continue
        for fname in sorted(os.listdir(level_path)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(level_path, fname)
            module = strip_suffix(fname)
            module_short = MODULE_LABELS.get(module, module)
            with open(fpath) as f:
                data = json.load(f)
            if not isinstance(data, dict) or "results" not in data:
                continue
            model = data.get("model", "unknown")
            for i, r in enumerate(data.get("results", [])):
                model_answer = r.get("model_answer")
                rows.append({
                    "model": model,
                    "level": level_name,
                    "module": module,
                    "module_short": module_short,
                    "problem_index": i + 1,
                    "expected_answer": r.get("expected_answer"),
                    "model_answer": model_answer,
                    "answered": model_answer is not None,
                    "correct": r.get("correct"),
                    "finish_reason": r.get("finish_reason"),
                    "input_tokens": r.get("input_tokens"),
                    "output_tokens": r.get("output_tokens"),
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS)
    parser.add_argument("--output", default="eval_results.csv")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    all_rows = []
    for d in args.dirs:
        path = d if os.path.isabs(d) else os.path.join(base, d)
        rows = load_dir(path)
        print(f"  {d}: {len(rows)} rows")
        all_rows.extend(rows)

    out_path = args.output if os.path.isabs(args.output) else os.path.join(base, args.output)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
