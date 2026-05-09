from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from openai import OpenAI

MAX_COMPLETION_TOKENS = 1200
DEFAULT_MODEL = "gpt-4.1-mini"

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_error_type": {
            "type": "string",
            "enum": ["algebra", "constraints", "complexity", "rounding", "other", "none"],
        },
        "secondary_error_types": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["algebra", "constraints", "complexity", "rounding", "other"],
            },
        },
        "stepwise_error_tags": {
            "type": "object",
            "properties": {
                "logical_gap": {"type": "boolean"},
                "numerical_computation_error": {"type": "boolean"},
                "numerical_approximation_error": {"type": "boolean"},
            },
            "required": [
                "logical_gap",
                "numerical_computation_error",
                "numerical_approximation_error",
            ],
            "additionalProperties": False,
        },
        "final_answer_status": {
            "type": "string",
            "enum": ["correct", "incorrect", "missing_or_unparseable"],
        },
        "confidence": {"type": "number"},
        "short_rationale": {"type": "string"},
        "evidence_span": {"type": "string"},
    },
    "required": [
        "primary_error_type",
        "secondary_error_types",
        "stepwise_error_tags",
        "final_answer_status",
        "confidence",
        "short_rationale",
        "evidence_span",
    ],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You are a careful math error-analysis judge.

Your task is to label the model's failure mode for a single math example.

Use these primary error types:
- algebra: symbolic manipulation mistakes, sign errors, solving equations
  incorrectly, invalid substitutions, incorrect derivation.
- constraints: misreading, dropping, flipping, or inventing constraints or
  assumptions, including unstated non-negativity assumptions in optimization.
- complexity: the solution strategy is too shallow or incomplete for the task,
  such as failing to consider edge cases, unboundedness, all branches, or the
  full search space.
- rounding: approximation, precision, or rounding mistakes.
- other: wrong, but none of the above fits best.
- none: the answer and reasoning appear correct.

Also set these step-wise error tags:
- logical_gap: unsupported leap or unjustified claim.
- numerical_computation_error: arithmetic or exact computation mistake.
- numerical_approximation_error: decimal/rounding/approximation issue.
- toy_case_overgeneralization: infers a general claim from a special case.

Judge from the provided question, gold answer, parsed model answer, and raw
response. Prefer the earliest root cause over downstream consequences.
Return strict JSON only."""


def _build_user_prompt(example):
    return json.dumps(
        {
            "question": example.get("question"),
            "expected_answer": example.get("expected_answer"),
            "model_answer": example.get("model_answer"),
            "correct": example.get("correct"),
            "raw_response": example.get("raw_response"),
        },
        indent=2,
        ensure_ascii=True,
    )


def _judge_example(client, model, example):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(example)},
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "math_error_judgment",
                "schema": _JUDGE_SCHEMA,
                "strict": True,
            },
        },
    )
    content = response.choices[0].message.content
    return json.loads(content)


def _summarize(judged_examples):
    primary_counts = Counter()
    secondary_counts = Counter()
    tag_counts = Counter()

    for item in judged_examples:
        judgment = item["judgment"]
        primary_counts[judgment["primary_error_type"]] += 1
        for error_type in judgment["secondary_error_types"]:
            secondary_counts[error_type] += 1
        for tag_name, present in judgment["stepwise_error_tags"].items():
            if present:
                tag_counts[tag_name] += 1

    return {
        "num_judged_examples": len(judged_examples),
        "primary_error_type_counts": dict(primary_counts),
        "secondary_error_type_counts": dict(secondary_counts),
        "stepwise_error_tag_counts": dict(tag_counts),
    }


def _load_examples(input_json_path):
    with open(input_json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        return data["results"]

    if isinstance(data, list):
        if not data:
            return []

        first = data[0]
        if isinstance(first, dict) and (
            "raw_response" in first or "model_answer" in first or "expected_answer" in first
        ):
            return data

        if isinstance(first, dict) and {"question", "answer", "level"}.issubset(first.keys()):
            raise ValueError(
                "This looks like a raw output_json dataset file with questions and gold answers only. "
                "judge_errors.py needs model outputs too, such as an eval_results file or a list of "
                "records containing raw_response/model_answer/expected_answer."
            )

    raise ValueError(
        "Unsupported input JSON format. Expected either an eval_results-style object with a "
        "'results' field or a list of model-output records."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Path to eval_results JSON file")
    parser.add_argument("--output_json", required=True, help="Where to write judged output JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Judge model to use")
    parser.add_argument(
        "--include_correct",
        action="store_true",
        help="Judge all examples instead of only incorrect ones",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set")

    examples = _load_examples(args.input_json)
    if not args.include_correct:
        examples = [example for example in examples if not example.get("correct", False)]

    client = OpenAI()
    judged_examples = []
    for index, example in enumerate(examples):
        judgment = _judge_example(client, args.model, example)
        judged_examples.append(
            {
                "index": index,
                "question": example.get("question"),
                "expected_answer": example.get("expected_answer"),
                "model_answer": example.get("model_answer"),
                "correct": example.get("correct"),
                "judgment": judgment,
            }
        )

    output = {
        "source_file": args.input_json,
        "judge_model": args.model,
        "include_correct": args.include_correct,
        "summary": _summarize(judged_examples),
        "judged_examples": judged_examples,
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
