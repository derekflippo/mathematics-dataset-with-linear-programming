"""Judge math evaluation outputs with one or more LLM-based judges.

Usage:
python judge_errors.py --input_json path/to/results.json --output_json judged.json --judge all --max_examples 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

from openai import OpenAI

MAX_COMPLETION_TOKENS = 5000
DEFAULT_MODEL = "gpt-4.1"

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
    "partial_reasoning_failure",
    "partial_constraint_handling_failure",
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
        "explanation": {"type": "string"},
        "formulation_score": {"type": "integer", "minimum": 0, "maximum": 2},
        "approach_score": {"type": "integer", "minimum": 0, "maximum": 2},
        "constraint_score": {"type": "integer", "minimum": 0, "maximum": 2},
        "hallucination": {"type": "boolean"},
        "hallucination_description": {"type": ["string", "null"]},
        "failure_point": {"type": "string"},
    },
    "required": [
        "explanation",
        "formulation_score",
        "approach_score",
        "constraint_score",
        "hallucination",
        "hallucination_description",
        "failure_point",
    ],
    "additionalProperties": False,
}

ARITHMETIC_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
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
    },
    "required": [
        "explanation",
        "arithmetic_error_found",
        "first_error_step",
        "computed_value",
        "correct_value",
        "error_magnitude",
        "cascaded",
        "cascade_description",
        "additional_independent_errors",
        "additional_independent_errors_description",
    ],
    "additionalProperties": False,
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
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
    },
    "required": [
        "explanation",
        "correct_value_in_reasoning",
        "correct_value_found",
        "distance_from_expected",
        "model_recognized_it",
        "reason_not_recognized",
        "context",
    ],
    "additionalProperties": False,
}

REASONING_PROMPT = """You are an expert judge evaluating the reasoning quality of an AI model solving a mathematical optimization problem.

Your job is to assess the reasoning process, not merely whether the final numerical answer matches the expected answer. Evaluate whether the model correctly understood the problem, chose a mathematically valid solution strategy, handled constraints properly, and justified global optimality.

The original problem forbids code, solvers, CVXPY, scipy, numpy, or any programming language. If the model claims it used a solver, CVXPY, scipy, numpy, Python, or another computational tool instead of mental mathematical reasoning, set approach_score=0. If it reports a solver-produced number without derivation, mark hallucination=true.

Evaluate these dimensions:

1. Problem Formulation (0-2)

2 = The objective function and all constraints are correctly identified and set up, including:
- correct quadratic forms
- correct linear terms
- correct inequality directions
- correct right-hand sides
- correct norm/ball constraint if present

1 = Minor formulation error, such as:
- one coefficient copied incorrectly
- one linear term omitted
- one constraint slightly mistranscribed
- notation is sloppy but the intended problem is mostly correct

0 = Fundamental formulation error, such as:
- wrong objective direction
- objective function substantially incorrect
- constraints ignored
- constraints applied with wrong inequality direction
- major constraint missing
- solving a different problem from the one stated


2. Solution Approach (0-2)

2 = The model chooses and correctly applies a method that establishes or justifies global optimality for the original optimization problem.

For optimization problems, approach_score=2 requires one of the following:
- correctly finds the unconstrained minimizer, verifies it satisfies all constraints, and therefore concludes it is the constrained global optimum
- correctly applies KKT conditions, including stationarity, feasibility, complementary slackness, and appropriate active constraints
- correctly performs active-set or boundary analysis and solves the relevant active-constraint system
- gives a valid convexity, symmetry, monotonicity, or geometric argument proving that the proposed point is globally optimal
- otherwise provides a mathematically valid proof that no feasible point can achieve a better objective value

1 = The model uses a partially valid approach but does not fully justify global optimality. Examples include:
- checks the unconstrained minimizer but stops after finding it infeasible
- identifies a likely active constraint but does not solve the active-set problem
- writes down KKT or Lagrange multiplier equations but abandons them or does not solve them
- performs a line search, ray search, coordinate-axis search, diagonal search, or reduced one-dimensional search without proving the optimizer lies there
- tests feasible points and chooses the best sampled point
- follows a descent direction without proving it reaches the global constrained optimum
- gives an approximate answer from heuristic exploration
- gives a plausible but incomplete boundary argument

0 = The model uses a wrong or invalid approach, or abandons the optimization problem without a valid path to the solution. Examples include:
- ignores constraints entirely when they matter
- treats an infeasible point as feasible
- uses only unsupported guessing
- applies a method unrelated to the problem
- claims global optimality from no meaningful argument
- abandons the solution and gives a number without a coherent derivation

Strict global-optimality rule:
A method is only fully correct if it provides a valid path to proving global optimality for the original problem. Testing feasible points, following a descent direction, performing local search, or solving a simplified subproblem does not establish global optimality unless the model proves why that search covers the global optimizer.

Restricted-subspace rule:
If the model restricts the search to a line, ray, axis, diagonal, coordinate slice, or special form such as x = t v, x = [a,a], x = [a,0], x = [0,a], x1 = 0, or x2 = 0, then the maximum approach_score is 1 unless the model gives a valid mathematical argument proving that the global optimizer must lie in that restricted set.

Valid justifications for a restricted subspace may include:
- symmetry of the objective and all constraints forcing the optimizer into that subspace
- KKT conditions implying that relation
- full active-set equations deriving that relation
- a convexity or monotonicity argument proving all other directions are suboptimal

Invalid justifications include:
- “this seems likely”
- “let me try”
- “this is close”
- point testing
- convenience
- intuition without proof

Abandoned-method rule:
If the model starts an exact method such as KKT, Lagrange multipliers, or active-set analysis, then says it is too complex and switches to trial-and-error, point sampling, or heuristic guessing, the maximum approach_score is 1.


3. Constraint Handling (0-2)

2 = The model correctly handles the constraints in a way that supports the claimed optimizer. This means one of:
- correctly verifies that the unconstrained minimizer satisfies all constraints, so no constraints are active
- correctly identifies which constraints are active and inactive at the proposed optimizer
- correctly applies the active constraints in KKT, active-set, boundary, or equivalent analysis
- correctly verifies feasibility of the final proposed optimizer and explains why the relevant constraints determine optimality

1 = The model partially handles constraints but does not fully justify the active set or constrained optimum. Examples include:
- checks constraints only at sampled points
- identifies a likely active constraint but does not prove it is the correct active set
- solves constraints only after imposing an unjustified restricted subspace
- verifies feasibility of a candidate point but does not prove optimality
- misses one potentially active constraint
- handles constraints correctly in arithmetic but not in the optimization logic

0 = The model mishandles constraints in a fundamental way. Examples include:
- ignores constraints that affect the solution
- treats an infeasible point as feasible
- applies inequality direction incorrectly
- uses a point that violates constraints as the final optimizer
- fails to check constraints after proposing a candidate
- claims a constraint is binding or inactive without basis and uses that error centrally


4. Hallucination

Hallucination means the model commits to a final answer that is unsupported by its own derivation, contradicts its own feasibility or validity checks, or is presented as a valid solution even though the model itself established that it is invalid.

Mark hallucination=true if any of the following occur:

A. Unsupported final number
- the model outputs a final numerical answer with no derivation supporting it
- the model gets stuck, says it cannot solve the problem, then produces a number anyway
- the model jumps from incomplete work to a final answer without explaining where the number came from

B. Infeasible or invalid point used as final answer
- the model computes a value from a point that violates one or more constraints, acknowledges that violation, and then still commits to that value as the final answer
- the model says a candidate is infeasible, invalid, or violates a constraint, but later reports that same candidate’s objective value as the minimum
- the model finds the unconstrained minimizer, determines it violates a constraint, but still gives the unconstrained objective value as the constrained optimum
- the model says the optimum must lie elsewhere, such as on a boundary or active constraint, but then reports the old invalid point’s value as the final answer without solving the new case

C. Forbidden tool use
- the model claims it used Python, CVXPY, scipy, numpy, MATLAB, a solver, or any computational tool to obtain the answer
- the model reports a solver-produced number without mental derivation

D. Invented problem facts
- the model invents constraints, assumptions, active sets, feasibility claims, or numerical values not derived from the problem
- the model changes the problem parameters and solves the modified problem
- the model assumes a constraint is inactive, active, or irrelevant with no derivation and then treats that assumption as proven

Category B is an absolute override. If the model explicitly acknowledged a constraint violation at a point — in any form, such as marking ✗, writing "> limit", saying "infeasible", "violated", or "not satisfied" — and then committed to that same point’s objective value as the final answer, hallucination=true. This applies even if the number follows logically from the model’s own calculations. "Follows from calculations" does not excuse committing to a known-infeasible point.

Mark hallucination=false if:
- the final answer follows logically from the model’s own shown work, AND the model did not explicitly acknowledge that the point was infeasible or violated a constraint before committing
- the model uses an incomplete heuristic method, restricted line search, point sampling, or unjustified subspace search, but the final number is derived from a point the model did not identify as infeasible
- the model makes a reasoning mistake but does not knowingly commit to an infeasible, invalid, or unsupported value
- the model solves a restricted subproblem (such as x2=0 or x=[a,a]) and reports the value from that restricted subproblem — this is usually a reasoning failure, not hallucination, unless the model also contradicts its own feasibility checks
- the model gives a feasible sampled point with a low objective value and incorrectly claims it is globally optimal — this is usually incomplete reasoning, not hallucination

Required check before deciding hallucination:
Scan the model’s reasoning for any constraint violation acknowledgment (✗, "> limit", "infeasible", "violated", "not satisfied", "does not satisfy"). If any such acknowledgment exists, check whether the final reported answer is the objective value at that same point. If yes, hallucination=true under category B regardless of anything else.

Important distinction:
A wrong answer is not automatically a hallucination. A heuristic or internally derived answer is usually a reasoning failure, not hallucination. A known-invalid or unsupported final answer is hallucination.

Hallucination-specific examples:

Example H1: The model finds unconstrained minimizer x*, checks constraints, constraint 2 is violated, then reports f(x*) as the final minimum. → hallucination=true (knowingly used infeasible point)
Example H2: The model says "this point is infeasible" but later says "therefore the minimum is the objective value at this point." → hallucination=true
Example H3: The model says "the optimum must lie on the boundary" after finding the unconstrained minimizer is infeasible, then gives the unconstrained objective value as the final answer without solving the boundary problem. → hallucination=true
Example H4: The model tries several feasible points and picks the best one. → hallucination=false, approach_score=1 (weak reasoning, not hallucination)
Example H5: The model restricts to x2=0 and solves the one-dimensional problem. → hallucination=false, approach_score=1 (unjustified restriction, not hallucination)
Example H6: The model outputs "the answer is -2.31" with no supporting derivation. → hallucination=true
Example H7: The model says "I cannot solve this exactly" then gives a final numerical answer with no derivation. → hallucination=true
Example H8: The model reports "CVXPY gives the answer -1.23" even though tool use is forbidden. → hallucination=true

When hallucination=true, the explanation must clearly state what made the answer unsupported, invalid, or contradictory.
When hallucination=false but the answer is still wrong, explain whether the issue is reasoning, arithmetic, formulation, or constraint handling instead.

Prefer the earliest root cause in explanations.

When explaining the judgment:
- explicitly state whether the model proved global optimality
- if the model used a restricted subspace, say whether that restriction was justified
- if constraints were only checked at sampled points, do not give constraint_score=2 unless optimality was also justified
- distinguish arithmetic mistakes from reasoning mistakes
- do not treat a mismatch with expected_answer as evidence of reasoning failure by itself

Concrete examples:

Example 1:
If the model finds the unconstrained minimizer, checks every constraint, and the point is feasible, then approach_score=2 and constraint_score=2. This is a complete global optimality argument for a convex problem.

Example 2:
If the model finds the unconstrained minimizer, discovers it violates a constraint, but then reports the unconstrained objective value as the final minimum, then approach_score=1, constraint_score=1, and hallucination=true if it explicitly acknowledged infeasibility before committing to that value.

Example 3:
If the model searches only along x = t v or along the negative gradient direction without proving the optimizer lies on that line, then approach_score=1. Constraint_score is at most 1 unless the active constraints are correctly used to prove optimality.

Example 4:
If the model sets x2 = 0, x1 = 0, x = [a,a], x = [a,0], or x = [0,a] and solves the resulting one-dimensional problem without proving that the global optimum lies in that subspace, then approach_score=1. Do not give approach_score=2 merely because the restricted one-dimensional algebra is correct.

Example 5:
If the model writes KKT or Lagrange multiplier conditions but abandons them because they are complex and switches to point sampling or guessing, then approach_score=1.

Example 6:
If the model tests several feasible points and picks the best one found, then approach_score=1 at best. Feasibility plus a low objective value is not a proof of global optimality.

Example 7:
If the model uses a point that violates constraints as its final optimizer, constraint_score=0 or 1 depending on whether it recognized the violation. If it recognized the violation and still committed to that value, hallucination=true.

Example 8:
If the final answer is numerically wrong only because of arithmetic, but the method and constraint logic are otherwise valid, do not lower the reasoning scores for the arithmetic mistake. The arithmetic judge should handle that.

Return strict JSON only."""

EXECUTE_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Execute Python code to verify arithmetic claims. Use numpy for matrix/vector "
            "operations. Always print() your results. Use this to back-substitute claimed "
            "solutions into gradient or constraint equations to verify they are correct."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Must use print() to output results.",
                }
            },
            "required": ["code"],
        },
    },
}

MAX_TOOL_CALLS = 6


def _execute_python(code, timeout=15):
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout.strip()
        if result.stderr.strip():
            output = (output + "\nSTDERR: " + result.stderr.strip()).strip()
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out"
    except Exception as e:
        return f"Error: {e}"


ARITHMETIC_PROMPT = """You are an expert judge whose sole task is to verify the arithmetic in an AI model's solution to an optimization problem. You are not evaluating whether the approach was correct or the reasoning was sound, only whether each numerical calculation was computed accurately.

You have access to a Python execution tool. Use it to verify critical arithmetic claims, especially:
- When the model claims a point x* is the unconstrained minimum, back-substitute into the gradient and verify it equals zero.
- When the model solves a system of linear equations, substitute the claimed solution back into the original equations to verify.
- When the model evaluates the objective or constraints at a point, verify the numerical result.

Always run at least one verification before concluding.

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


def _judge_arithmetic_with_tools(client, model, example):
    spec = JUDGE_SPECS["arithmetic"]
    messages = [
        {"role": "system", "content": spec["prompt"]},
        {"role": "user", "content": _build_user_prompt(example)},
    ]

    for _ in range(MAX_TOOL_CALLS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            tools=[EXECUTE_PYTHON_TOOL],
            tool_choice="auto",
        )
        choice = response.choices[0]
        msg = choice.message
        messages.append(msg)

        if choice.finish_reason != "tool_calls":
            break

        for tool_call in msg.tool_calls:
            if tool_call.function.name == "execute_python":
                code = json.loads(tool_call.function.arguments)["code"]
                result = _execute_python(code)
            else:
                result = "Unknown tool"
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    messages.append({
        "role": "user",
        "content": "Based on your analysis and any code execution results, produce the final arithmetic judgment.",
    })
    final = client.chat.completions.create(
        model=model,
        messages=messages,
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
    judgment = json.loads(final.choices[0].message.content)
    return _normalize_arithmetic_judgment(judgment)


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
    if judge_name == "arithmetic":
        return _judge_arithmetic_with_tools(client, model, example)
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
        elif reasoning.get("approach_score") == 1:
            tags.append("partial_reasoning_failure")
        if reasoning.get("constraint_score") == 0:
            tags.append("constraint_handling_failure")
        elif reasoning.get("constraint_score") == 1:
            tags.append("partial_constraint_handling_failure")

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
    elif "partial_constraint_handling_failure" in seen:
        primary_error_type = "partial_constraint_handling_failure"
    elif "partial_reasoning_failure" in seen:
        primary_error_type = "partial_reasoning_failure"
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
            for attempt in range(5):
                try:
                    judged_example[output_key] = _judge_example(client, args.model, judge_name, example)
                    break
                except Exception as exc:
                    msg = str(exc)
                    if "429" in msg and attempt < 4:
                        wait = 5 * (attempt + 1)
                        print(f"  Rate limit hit, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        judged_example[output_key] = {"judgment_error": msg}
                        break

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
