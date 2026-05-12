"""Judge math evaluation outputs with one or more LLM-based judges.

Usage:
python judge_errors.py --input_json path/to/results.json --output_json judged.json --judge all --max_examples 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

from openai import OpenAI

MAX_COMPLETION_TOKENS = 2500
DEFAULT_MODEL = "gpt-4.1-mini"

FIELD_ALIASES = {
    "question": ["question", "problem", "prompt"],
    "expected_answer": ["expected_answer", "answer", "gold_answer"],
    "model_answer": ["model_answer", "predicted_answer", "final_answer"],
    "raw_response": ["raw_response", "reasoning", "response", "output"],
    "correct": ["correct", "is_correct"],
    "problem_type": ["problem_type", "domain", "type"],
    "level": ["level", "difficulty"],
    "finish_reason": ["finish_reason"],
    "input_tokens": ["input_tokens"],
    "output_tokens": ["output_tokens"],
}

JUDGE_CHOICES = ("reasoning", "arithmetic", "final", "all")
PRIMARY_ERROR_TYPES = (
    "reasoning_failure",
    "arithmetic_failure",
    "constraint_handling_failure",
    "hallucination",
    "continued_refinement",
    "dismissed_correct_answer",
    "no_correct_value_found",
    "mixed_failure",
    "unknown",
)

REASONING_SCHEMA = {
    "type": "object",
    "properties": {
        "formulation_score": {"type": "integer", "minimum": 0, "maximum": 2},
        "approach_score": {"type": "integer", "minimum": 0, "maximum": 2},
        "constraint_score": {"type": "integer", "minimum": 0, "maximum": 2},
        "hallucination": {"type": "boolean"},
        "hallucination_description": {"type": ["string", "null"]},
        "failure_point": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": [
        "formulation_score",
        "approach_score",
        "constraint_score",
        "hallucination",
        "hallucination_description",
        "failure_point",
        "explanation",
    ],
    "additionalProperties": False,
}

ARITHMETIC_SCHEMA = {
    "type": "object",
    "properties": {
        "arithmetic_error_found": {"type": "boolean"},
        "first_error_step": {"type": ["string", "null"]},
        "computed_value": {"type": ["string", "null"]},
        "correct_value": {"type": ["string", "null"]},
        "error_magnitude": {
            "type": ["string", "null"],
            "enum": ["small (< 5%)", "medium (5-20%)", "large (> 20%)", None],
        },
        "cascaded": {"type": ["boolean", "null"]},
        "cascade_description": {"type": ["string", "null"]},
        "additional_independent_errors": {"type": "integer", "minimum": 0},
        "additional_independent_errors_description": {"type": ["string", "null"]},
        "explanation": {"type": "string"},
    },
    "required": [
        "arithmetic_error_found",
        "first_error_step",
        "computed_value",
        "correct_value",
        "error_magnitude",
        "cascaded",
        "cascade_description",
        "additional_independent_errors",
        "additional_independent_errors_description",
        "explanation",
    ],
    "additionalProperties": False,
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "correct_value_in_reasoning": {"type": "boolean"},
        "correct_value_found": {"type": ["string", "null"]},
        "distance_from_expected": {"type": ["string", "null"]},
        "model_recognized_it": {"type": ["boolean", "null"]},
        "reason_not_recognized": {
            "type": ["string", "null"],
            "enum": [
                "token_limit",
                "continued_refinement",
                "dismissed",
                None,
            ],
        },
        "context": {"type": ["string", "null"]},
        "explanation": {"type": "string"},
    },
    "required": [
        "correct_value_in_reasoning",
        "correct_value_found",
        "distance_from_expected",
        "model_recognized_it",
        "reason_not_recognized",
        "context",
        "explanation",
    ],
    "additionalProperties": False,
}

REASONING_PROMPT = """You are an expert judge evaluating the reasoning quality of an AI model solving a mathematical optimization problem. Your job is to assess the reasoning process, not whether the final answer is numerically correct, but whether the model understood and approached the problem correctly.

The original problem forbids code, solvers, CVXPY, scipy, numpy, or any programming language. If the model claims it used a solver or CVXPY instead of mental mathematical reasoning, score approach_score=0. If it reports a solver-produced number without derivation, mark hallucination=true.

Evaluate these dimensions:

1. Problem Formulation (0-2)
2 = objective function and all constraints correctly identified and set up
1 = minor formulation error, such as one coefficient wrong or one constraint mistranscribed
0 = fundamental error, such as wrong objective direction, constraints misread, or constraints ignored

2. Solution Approach (0-2)
2 = appropriate method chosen and correctly applied with valid logical steps
1 = appropriate method chosen but incorrectly applied
0 = wrong method or approach abandoned without valid reason

3. Constraint Handling (0-2)
2 = correctly identified binding constraints and applied them
1 = partially correct, identified some active constraints but missed others or applied them with minor errors
0 = constraints misidentified, ignored, or applied in the wrong direction

4. Hallucination
Mark hallucination true if:
- the model outputs a number with no derivation supporting it
- the model computes a value from an infeasible or invalid point, acknowledges the issue, then commits to that value anyway
- the model gets stuck, says it cannot solve it, then produces a number anyway

Mark hallucination false if:
- the answer follows logically from the model’s work, even if the work contains arithmetic or reasoning errors

Prefer the earliest root cause in explanations.
Return strict JSON only."""

ARITHMETIC_PROMPT = """You are an expert judge whose sole task is to verify the arithmetic in an AI model's solution to an optimization problem. You are not evaluating whether the approach was correct or the reasoning was sound, only whether each numerical calculation was computed accurately.

What counts as arithmetic error:
- incorrect multiplication, division, addition, or subtraction
- incorrect powers, roots, logarithms
- incorrect matrix-vector products or dot products
- incorrect simplification of numerical algebraic expressions
- rounding that introduces meaningful error, more than about 2%

What does NOT count as arithmetic error:
- choosing the wrong method
- applying a formula to the wrong quantity
- using the wrong constraint
- getting a correct intermediate value but drawing the wrong conclusion

You must independently verify arithmetic from the model reasoning itself. Do not assume the expected_answer is correct evidence of an arithmetic error. If the model's arithmetic is internally correct but differs from expected_answer, mark arithmetic_error_found=false and explain that the discrepancy may be due to reasoning/formulation/expected-answer mismatch.

Distinguish cascade errors from independent errors:
- A cascade error is downstream of the first arithmetic mistake.
- An independent error is a separate arithmetic mistake that would still exist even if the first error were fixed.

If arithmetic_error_found=true, first_error_step, computed_value, and correct_value must describe a concrete incorrect computation. Do not mark arithmetic_error_found=true for vague mismatch with expected_answer.

Prefer the earliest root cause in explanations.
Return strict JSON only."""

FINAL_PROMPT = """You are an expert judge checking whether a model computed the correct final value during reasoning but failed to recognize or commit to it as the final answer.

Use tolerance 0.01.

You will be given the model's finish_reason. Use it carefully:
- If finish_reason is "max_tokens", that is strong evidence for reason_not_recognized = "token_limit", but only if a correct value appeared in the reasoning and the model did not submit it.
- If finish_reason is "end_turn", do not use token_limit unless the text itself clearly cuts off abruptly.

Only flag correct_value_in_reasoning true if all are true:
- A numerical value within 0.01 of the expected answer appears in the reasoning.
- The value is clearly relevant to the final answer, not a coincidental intermediate value.
- The model did not submit this value as its final answer.
- The reason it did not commit is clearly one of:
  - token_limit
  - continued_refinement
  - dismissed

Definitions:
- token_limit: output ends abruptly before the model could commit
- continued_refinement: model has the value but explicitly chooses to keep refining or iterating
- dismissed: model explicitly rejects the correct value as wrong or infeasible
- If the model had the correct value but replaced it with a rounded incorrect value, classify that as dismissed.

Additional consistency rules:
- If correct_value_in_reasoning=true and model_recognized_it=false, reason_not_recognized must be one of token_limit, continued_refinement, or dismissed. It must never be null.
- If you cannot confidently assign token_limit, continued_refinement, or dismissed, set correct_value_in_reasoning=false.

Consistency requirements:
- If correct_value_in_reasoning is true, distance_from_expected must be less than or equal to 0.01.
- If no value within 0.01 exists, correct_value_in_reasoning must be false, correct_value_found must be null, and distance_from_expected must be null.

Prefer the earliest root cause in explanations.
Return strict JSON only."""

JUDGE_SPECS = {
    "reasoning": {
        "prompt": REASONING_PROMPT,
        "schema": REASONING_SCHEMA,
        "schema_name": "reasoning_judgment",
        "output_key": "reasoning_judgment",
    },
    "arithmetic": {
        "prompt": ARITHMETIC_PROMPT,
        "schema": ARITHMETIC_SCHEMA,
        "schema_name": "arithmetic_judgment",
        "output_key": "arithmetic_judgment",
    },
    "final": {
        "prompt": FINAL_PROMPT,
        "schema": FINAL_SCHEMA,
        "schema_name": "final_answer_recognition_judgment",
        "output_key": "final_answer_recognition_judgment",
    },
}


def _first_present(example, aliases):
    for key in aliases:
        if key in example:
            return example.get(key)
    return None


def _nested_sources(example):
    sources = [example]
    for key in ("metadata", "result", "evaluation", "sample"):
        value = example.get(key)
        if isinstance(value, dict):
            sources.append(value)
    return sources


def _first_present_with_nested(example, aliases):
    top_level_value = _first_present(example, aliases)
    if top_level_value is not None:
        return top_level_value
    for alias in aliases:
        if alias in example:
            return example.get(alias)
    for source in _nested_sources(example)[1:]:
        value = _first_present(source, aliases)
        if value is not None:
            return value
        for alias in aliases:
            if alias in source:
                return source.get(alias)
    return None


def _parse_raw_response(raw_response):
    parsed = None
    normalized_raw_response = raw_response
    parsed_model_answer = None

    if isinstance(raw_response, str):
        try:
            parsed = json.loads(raw_response)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
    elif isinstance(raw_response, dict):
        parsed = raw_response

    if isinstance(parsed, dict):
        reasoning = parsed.get("reasoning")
        if reasoning is not None:
            if isinstance(reasoning, str):
                normalized_raw_response = reasoning
            else:
                normalized_raw_response = json.dumps(reasoning, ensure_ascii=True)
        parsed_model_answer = parsed.get("answer")

    return normalized_raw_response, parsed_model_answer


def _infer_problem_type_from_path(input_json_path):
    base_name = os.path.splitext(os.path.basename(input_json_path))[0]
    if "__" in base_name:
        return base_name.split("__", 1)[0] or None

    parts = [part for part in re.split(r"[\\/]+", input_json_path) if part]
    for part in reversed(parts):
        if "__" in part:
            return part.split("__", 1)[0] or None
    return None


def _infer_problem_type_from_question(question):
    if not isinstance(question, str):
        return None
    lowered = question.lower()
    if "quadratically constrained quadratic program" in lowered or "qcqp" in lowered:
        return "qcqp"
    return None


def _infer_level_from_path(input_json_path):
    match = re.search(r"(^|[\\/])(level-\d+)([\\/]|$)", input_json_path)
    if match:
        return match.group(2)
    return None


def _normalize_example(example, input_json_path=None):
    if not isinstance(example, dict):
        example = {}

    normalized = {
        "original_example": example,
    }
    for canonical_key, aliases in FIELD_ALIASES.items():
        normalized[canonical_key] = _first_present_with_nested(example, aliases)

    normalized_raw_response, parsed_model_answer = _parse_raw_response(normalized.get("raw_response"))
    normalized["raw_response"] = normalized_raw_response
    if normalized.get("model_answer") is None and parsed_model_answer is not None:
        normalized["model_answer"] = parsed_model_answer
    if normalized.get("problem_type") is None:
        normalized["problem_type"] = _infer_problem_type_from_question(normalized.get("question"))
    if normalized.get("problem_type") is None and input_json_path:
        normalized["problem_type"] = _infer_problem_type_from_path(input_json_path)
    if normalized.get("level") is None and input_json_path:
        normalized["level"] = _infer_level_from_path(input_json_path)

    return normalized


def _build_user_prompt(example):
    return json.dumps(
        {
            "question": example.get("question") or "",
            "expected_answer": example.get("expected_answer"),
            "model_answer": example.get("model_answer"),
            "raw_response": example.get("raw_response") or "",
            "correct": example.get("correct"),
            "problem_type": example.get("problem_type"),
            "level": example.get("level"),
            "finish_reason": example.get("finish_reason"),
            "input_tokens": example.get("input_tokens"),
            "output_tokens": example.get("output_tokens"),
        },
        indent=2,
        ensure_ascii=True,
    )


def _judge_example(client, model, judge_name, example):
    spec = JUDGE_SPECS[judge_name]
    if judge_name == "reasoning":
        raw_response = example.get("raw_response")
        if not isinstance(raw_response, str) or not raw_response.strip():
            return {
                "formulation_score": 0,
                "approach_score": 0,
                "constraint_score": 0,
                "hallucination": False,
                "hallucination_description": None,
                "failure_point": "empty_response",
                "explanation": "The model produced an empty response, so reasoning quality is scored as zero across formulation, approach, and constraint handling.",
            }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": spec["prompt"]},
            {"role": "user", "content": _build_user_prompt(example)},
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": spec["schema_name"],
                "schema": spec["schema"],
                "strict": True,
            },
        },
    )
    content = response.choices[0].message.content
    judgment = json.loads(content)
    if judge_name == "arithmetic":
        judgment = _normalize_arithmetic_judgment(judgment)
    if judge_name == "final":
        judgment = _normalize_final_judgment(judgment)
    return judgment


def _normalize_arithmetic_judgment(judgment):
    if not isinstance(judgment, dict):
        return judgment

    computed_value = judgment.get("computed_value")
    correct_value = judgment.get("correct_value")
    first_error_step = judgment.get("first_error_step")
    explanation = judgment.get("explanation")
    non_error_step = False
    if first_error_step is None:
        non_error_step = True
    elif isinstance(first_error_step, str):
        lowered = first_error_step.strip().lower()
        non_error_step = lowered in {"", "none", "n/a", "no error", "no arithmetic error"}
    explanation_lower = explanation.lower() if isinstance(explanation, str) else ""
    explanation_negates_error = (
        "no arithmetic error found" in explanation_lower
        or "therefore, no arithmetic error" in explanation_lower
    )

    if judgment.get("arithmetic_error_found") is True:
        if explanation_negates_error or (
            computed_value is not None
            and correct_value is not None
            and str(computed_value) == str(correct_value)
            and non_error_step
        ):
            judgment["arithmetic_error_found"] = False
            judgment["first_error_step"] = None
            judgment["computed_value"] = None
            judgment["correct_value"] = None
            judgment["error_magnitude"] = None
            judgment["cascaded"] = None
            judgment["cascade_description"] = None

    return judgment


def _normalize_final_judgment(judgment):
    if not isinstance(judgment, dict):
        return judgment

    correct_value_in_reasoning = judgment.get("correct_value_in_reasoning")
    distance_from_expected = judgment.get("distance_from_expected")
    model_recognized_it = judgment.get("model_recognized_it")
    reason_not_recognized = judgment.get("reason_not_recognized")

    parsed_distance = None
    if isinstance(distance_from_expected, str):
        try:
            parsed_distance = float(distance_from_expected.strip())
        except ValueError:
            parsed_distance = None

    if correct_value_in_reasoning is True:
        if parsed_distance is None or parsed_distance > 0.01:
            judgment["correct_value_in_reasoning"] = False
            judgment["correct_value_found"] = None
            judgment["distance_from_expected"] = None
            judgment["model_recognized_it"] = None
            judgment["reason_not_recognized"] = None
            judgment["context"] = None
        elif model_recognized_it is False and reason_not_recognized is None:
            judgment["correct_value_in_reasoning"] = False
            judgment["correct_value_found"] = None
            judgment["distance_from_expected"] = None
            judgment["model_recognized_it"] = None
            judgment["context"] = None
    else:
        judgment["correct_value_in_reasoning"] = False
        judgment["correct_value_found"] = None
        judgment["distance_from_expected"] = None
        if judgment.get("reason_not_recognized") is not None and judgment.get("model_recognized_it") is None:
            judgment["model_recognized_it"] = False
        if parsed_distance is None or parsed_distance > 0.01:
            judgment["reason_not_recognized"] = None
            judgment["model_recognized_it"] = None
            judgment["context"] = None

    judgment["explanation"] = _build_final_explanation(judgment)
    return judgment


def _build_final_explanation(judgment):
    correct_value_in_reasoning = judgment.get("correct_value_in_reasoning")
    correct_value_found = judgment.get("correct_value_found")
    distance_from_expected = judgment.get("distance_from_expected")
    model_recognized_it = judgment.get("model_recognized_it")
    reason_not_recognized = judgment.get("reason_not_recognized")
    context = judgment.get("context")
    original_explanation = judgment.get("explanation")

    if correct_value_in_reasoning is True:
        parts = [
            "A correct value within 0.01 appeared in the reasoning",
        ]
        if correct_value_found is not None:
            parts.append(f"value={correct_value_found}")
        if distance_from_expected is not None:
            parts.append(f"distance_from_expected={distance_from_expected}")
        if model_recognized_it is True:
            parts.append("model_recognized_it=true")
        elif model_recognized_it is False:
            parts.append("model_recognized_it=false")
        if reason_not_recognized is not None:
            parts.append(f"reason_not_recognized={reason_not_recognized}")
        if context:
            parts.append(f"context={context}")
        if original_explanation:
            parts.append(f"normalized_note={original_explanation}")
        return "; ".join(parts) + "."

    parts = [
        "No correct value within 0.01 was retained as a valid final-answer-recognition case"
    ]
    if model_recognized_it is True:
        parts.append("model_recognized_it=true")
    elif model_recognized_it is False:
        parts.append("model_recognized_it=false")
    if original_explanation:
        parts.append(f"normalized_note={original_explanation}")
    return "; ".join(parts) + "."


def _is_valid_judgment(judgment):
    return isinstance(judgment, dict) and "judgment_error" not in judgment


def _derive_error_taxonomy(judged_example):
    reasoning = judged_example.get("reasoning_judgment")
    arithmetic = judged_example.get("arithmetic_judgment")
    final = judged_example.get("final_answer_recognition_judgment")

    tags = []

    if _is_valid_judgment(reasoning):
        if reasoning.get("hallucination") is True:
            tags.append("hallucination")
        if reasoning.get("formulation_score") == 0 or reasoning.get("approach_score") == 0:
            tags.append("reasoning_failure")
        if reasoning.get("constraint_score") == 0:
            tags.append("constraint_handling_failure")

    if _is_valid_judgment(final) and final.get("correct_value_in_reasoning") is True:
        final_reason = final.get("reason_not_recognized")
        final_mapping = {
            "continued_refinement": "continued_refinement",
            "dismissed": "dismissed_correct_answer",
        }
        mapped = final_mapping.get(final_reason)
        if mapped is not None:
            tags.append(mapped)

    if _is_valid_judgment(arithmetic) and arithmetic.get("arithmetic_error_found") is True:
        tags.append("arithmetic_failure")

    if (
        _is_valid_judgment(reasoning)
        and reasoning.get("formulation_score") == 2
        and reasoning.get("approach_score") == 2
        and reasoning.get("constraint_score") == 2
        and (
            not _is_valid_judgment(final)
            or final.get("correct_value_in_reasoning") is not True
        )
    ):
        tags.append("no_correct_value_found")

    ordered_unique_tags = []
    seen = set()
    for tag in tags:
        if tag not in seen:
            ordered_unique_tags.append(tag)
            seen.add(tag)

    primary_error_type = "unknown"
    if "hallucination" in seen:
        primary_error_type = "hallucination"
    elif "reasoning_failure" in seen:
        primary_error_type = "reasoning_failure"
    elif "constraint_handling_failure" in seen:
        primary_error_type = "constraint_handling_failure"
    elif _is_valid_judgment(final) and final.get("correct_value_in_reasoning") is True:
        final_reason = final.get("reason_not_recognized")
        primary_error_type = {
            "continued_refinement": "continued_refinement",
            "dismissed": "dismissed_correct_answer",
        }.get(final_reason, primary_error_type)
    elif "arithmetic_failure" in seen:
        primary_error_type = "arithmetic_failure"
    elif "no_correct_value_found" in seen:
        primary_error_type = "no_correct_value_found"

    secondary_error_types = [tag for tag in ordered_unique_tags if tag != primary_error_type]
    if primary_error_type != "unknown" and secondary_error_types:
        secondary_error_types = list(secondary_error_types)

    if primary_error_type != "unknown" and len(ordered_unique_tags) > 1 and primary_error_type not in {
        "hallucination",
        "reasoning_failure",
        "constraint_handling_failure",
    }:
        secondary_error_types = [tag for tag in ordered_unique_tags if tag != primary_error_type]

    if len(ordered_unique_tags) > 1 and primary_error_type == "unknown":
        primary_error_type = "mixed_failure"
        secondary_error_types = ordered_unique_tags

    judged_example["primary_error_type"] = primary_error_type
    judged_example["secondary_error_types"] = secondary_error_types
    return judged_example


def _load_examples(input_json_path):
    with open(input_json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]

    if isinstance(data, list):
        return data

    raise ValueError(
        "Unsupported input JSON format. Expected either an object with a 'results' field or a list of examples."
    )


def _average(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def _summarize(judged_examples):
    formulation_total = 0
    approach_total = 0
    constraint_total = 0
    formulation_score_count = 0
    approach_score_count = 0
    constraint_score_count = 0
    reasoning_judged_count = 0
    arithmetic_judged_count = 0
    final_judged_count = 0
    judgment_error_count = 0
    hallucination_count = 0
    arithmetic_error_found_count = 0
    cascaded_count = 0
    correct_value_in_reasoning_count = 0
    reason_not_recognized_counts = Counter()
    problem_type_counts = Counter()
    level_counts = Counter()
    finish_reason_counts = Counter()
    primary_error_type_counts = Counter()
    secondary_error_type_counts = Counter()

    for item in judged_examples:
        problem_type = item.get("problem_type")
        level = item.get("level")
        finish_reason = item.get("finish_reason")
        problem_type_counts[str(problem_type) if problem_type is not None else "unknown"] += 1
        level_counts[str(level) if level is not None else "unknown"] += 1
        finish_reason_counts[str(finish_reason) if finish_reason is not None else "unknown"] += 1
        primary_error_type = item.get("primary_error_type")
        if primary_error_type is not None:
            primary_error_type_counts[str(primary_error_type)] += 1
        secondary_error_types = item.get("secondary_error_types")
        if isinstance(secondary_error_types, list):
            for error_type in secondary_error_types:
                secondary_error_type_counts[str(error_type)] += 1

        reasoning = item.get("reasoning_judgment")
        if isinstance(reasoning, dict):
            if "judgment_error" in reasoning:
                judgment_error_count += 1
            else:
                reasoning_judged_count += 1
                formulation_score = reasoning.get("formulation_score")
                approach_score = reasoning.get("approach_score")
                constraint_score = reasoning.get("constraint_score")
                if isinstance(formulation_score, (int, float)):
                    formulation_total += formulation_score
                    formulation_score_count += 1
                if isinstance(approach_score, (int, float)):
                    approach_total += approach_score
                    approach_score_count += 1
                if isinstance(constraint_score, (int, float)):
                    constraint_total += constraint_score
                    constraint_score_count += 1
                if reasoning.get("hallucination") is True:
                    hallucination_count += 1

        arithmetic = item.get("arithmetic_judgment")
        if isinstance(arithmetic, dict):
            if "judgment_error" in arithmetic:
                judgment_error_count += 1
            else:
                arithmetic_judged_count += 1
                if arithmetic.get("arithmetic_error_found") is True:
                    arithmetic_error_found_count += 1
                if arithmetic.get("cascaded") is True:
                    cascaded_count += 1

        final = item.get("final_answer_recognition_judgment")
        if isinstance(final, dict):
            if "judgment_error" in final:
                judgment_error_count += 1
            else:
                final_judged_count += 1
                if final.get("correct_value_in_reasoning") is True:
                    correct_value_in_reasoning_count += 1
                reason = final.get("reason_not_recognized")
                if reason is not None:
                    reason_not_recognized_counts[str(reason)] += 1

    return {
        "num_judged_examples": len(judged_examples),
        "reasoning_judged_count": reasoning_judged_count,
        "arithmetic_judged_count": arithmetic_judged_count,
        "final_judged_count": final_judged_count,
        "judgment_error_count": judgment_error_count,
        "average_formulation_score": _average(formulation_total, formulation_score_count),
        "average_approach_score": _average(approach_total, approach_score_count),
        "average_constraint_score": _average(constraint_total, constraint_score_count),
        "hallucination_count": hallucination_count,
        "arithmetic_error_found_count": arithmetic_error_found_count,
        "cascaded_count": cascaded_count,
        "correct_value_in_reasoning_count": correct_value_in_reasoning_count,
        "reason_not_recognized_counts": dict(reason_not_recognized_counts),
        "primary_error_type_counts": dict(primary_error_type_counts),
        "secondary_error_type_counts": dict(secondary_error_type_counts),
        "counts_by_problem_type": dict(problem_type_counts),
        "counts_by_level": dict(level_counts),
        "finish_reason_counts": dict(finish_reason_counts),
    }


def _format_counter_lines(counter_dict):
    if not counter_dict:
        return ["  (none)"]
    lines = []
    for key in sorted(counter_dict):
        lines.append(f"  {key}: {counter_dict[key]}")
    return lines


def _print_terminal_summary(summary):
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print("Primary error types:")
    for line in _format_counter_lines(summary.get("primary_error_type_counts", {})):
        print(line)
    print("-" * 72)
    print("Average reasoning scores:")
    print(f"  formulation: {summary.get('average_formulation_score')}")
    print(f"  approach: {summary.get('average_approach_score')}")
    print(f"  constraint: {summary.get('average_constraint_score')}")
    print(f"  hallucination_count: {summary.get('hallucination_count')}")
    print("-" * 72)
    print("Arithmetic:")
    print(f"  arithmetic_error_found_count: {summary.get('arithmetic_error_found_count')}")
    print(f"  cascaded_count: {summary.get('cascaded_count')}")
    print("-" * 72)
    print("Final answer recognition:")
    print(f"  correct_value_in_reasoning_count: {summary.get('correct_value_in_reasoning_count')}")
    print("  reason_not_recognized_counts:")
    for line in _format_counter_lines(summary.get("reason_not_recognized_counts", {})):
        print(line)
    print("-" * 72)
    print("By problem type:")
    for line in _format_counter_lines(summary.get("counts_by_problem_type", {})):
        print(line)
    print("-" * 72)
    print("By level:")
    for line in _format_counter_lines(summary.get("counts_by_level", {})):
        print(line)
    print("-" * 72)
    print("Finish reasons:")
    for line in _format_counter_lines(summary.get("finish_reason_counts", {})):
        print(line)
    print("=" * 72)


def _selected_judges(judge_arg):
    if judge_arg == "all":
        return ["reasoning", "arithmetic", "final"]
    return [judge_arg]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Path to input JSON file")
    parser.add_argument("--output_json", required=True, help="Where to write judged output JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Judge model to use")
    parser.add_argument(
        "--include_correct",
        action="store_true",
        help="Judge all examples instead of only incorrect ones",
    )
    parser.add_argument(
        "--judge",
        choices=JUDGE_CHOICES,
        default="all",
        help="Which judge to run",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional maximum number of examples to judge",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set")

    examples = [_normalize_example(example, args.input_json) for example in _load_examples(args.input_json)]
    if not args.include_correct:
        examples = [example for example in examples if not example.get("correct", False)]
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    judges = _selected_judges(args.judge)
    client = OpenAI()
    judged_examples = []
    total_examples = len(examples)

    for index, example in enumerate(examples):
        print(f"Judging example {index + 1}/{total_examples}")
        judged_example = {
            "index": index,
            "question": example.get("question") or "",
            "expected_answer": example.get("expected_answer"),
            "model_answer": example.get("model_answer"),
            "raw_response": example.get("raw_response") or "",
            "correct": example.get("correct"),
            "problem_type": example.get("problem_type"),
            "level": example.get("level"),
            "finish_reason": example.get("finish_reason"),
            "input_tokens": example.get("input_tokens"),
            "output_tokens": example.get("output_tokens"),
        }

        for judge_name in judges:
            output_key = JUDGE_SPECS[judge_name]["output_key"]
            try:
                judged_example[output_key] = _judge_example(client, args.model, judge_name, example)
            except Exception as exc:
                judged_example[output_key] = {"judgment_error": str(exc)}

        judged_example = _derive_error_taxonomy(judged_example)
        judged_examples.append(judged_example)

    output = {
        "source_file": args.input_json,
        "judge_model": args.model,
        "judge": args.judge,
        "include_correct": args.include_correct,
        "max_examples": args.max_examples,
        "summary": _summarize(judged_examples),
        "judged_examples": judged_examples,
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output["summary"], indent=2))
    _print_terminal_summary(output["summary"])


if __name__ == "__main__":
    main()
