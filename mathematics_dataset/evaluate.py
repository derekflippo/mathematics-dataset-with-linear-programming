"""Evaluate an LLM on generated optimization problems.

Reads a generated JSON file of question/answer pairs, sends each question
to an LLM, and compares the response to the verified answer.

Usage:
  export OPENAI_API_KEY=sk-...
  export ANTHROPIC_API_KEY=sk-ant-...
  export GEMINI_API_KEY=...
  export DEEPSEEK_API_KEY=sk-...
  export DASHSCOPE_API_KEY=sk-...
  python -m mathematics_dataset.evaluate \
    --input_json=output_json/level-1/quadratic_programming__quadratic_programming.json \
    --output_dir=eval_results
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import time
import re

from absl import app
from absl import flags
from absl import logging
from openai import OpenAI
import anthropic
from google import genai as google_genai
from google.genai import types as google_types

FLAGS = flags.FLAGS

flags.DEFINE_string('input_json', None, 'Path to a single generated JSON file to evaluate')
flags.DEFINE_string('input_dir', None, 'Path to output_json directory to evaluate all levels')
flags.DEFINE_string('output_dir', None, 'Directory to write per-model evaluation results')
flags.DEFINE_string('levels', None, 'Levels to evaluate, e.g. "1-4" or "2,5,7" (only used with --input_dir)')
flags.mark_flag_as_required('output_dir')

# ── Configurations ────────────────────────────────────────────────────────────

ENGINES = [
    # "gpt-4o-mini",
    # "gpt-4.1-mini",
    # "gpt-4.1",
    # "gpt-5",
    # "o4-mini",
    # "o3",
    # "gpt-5.5",
    # "gpt-5.4",
    # "gpt-5.4-mini",
    # "claude-opus-4-7",
    "claude-sonnet-5",
    # "claude-sonnet-4-6",
    # "claude-haiku-4-5-20251001",
    # "deepseek-chat",
    # "deepseek-reasoner",
    # "gemini-2.5-pro",
    # "gemini-2.5-flash",
    # "qwen3-235b-a22b-thinking-2507",
]

REL_TOLERANCE = 1e-3

# ── Token envelope ────────────────────────────────────────────────────────────
# Input tokens are always separate and are never counted against the caps below.
# THINKING_BUDGET is a hard reasoning-token cap for the models that accept one
# (Gemini, Qwen). Claude Sonnet 5 has no thinking-budget knob — it uses adaptive
# thinking bounded only by MAX_OUTPUT_TOKENS (thinking + answer share it) — and
# OpenAI/DeepSeek expose only an effort level, so for those three MAX_OUTPUT_TOKENS
# is the real limit. ANSWER_HEADROOM keeps ~1k for the answer on top of the budget.
THINKING_BUDGET = 15000
ANSWER_HEADROOM = 1000
MAX_OUTPUT_TOKENS = THINKING_BUDGET + ANSWER_HEADROOM

SYSTEM_PROMPT = (
    "You are a math solver. Solve the given optimization problem. "
    "Report the final objective value to at least 4 significant figures."
)

# ─────────────────────────────────────────────────────────────────────────────

ANTHROPIC_MODELS = {
    "claude-opus-4-7",
    "claude-opus-4-5",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
}

OPENAI_MODELS = {
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-5",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "o4-mini",
    "o3",
}

GEMINI_MODELS = {
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
}

DEEPSEEK_MODELS = {
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
}

QWEN_MODELS = {
    "qwen3-235b-a22b-thinking-2507",
}

_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "number"},
    },
    "required": ["answer"],
    "additionalProperties": False,
}


def _is_anthropic(model):
    return model in ANTHROPIC_MODELS or model.startswith("claude")

def _is_openai(model):
    return model in OPENAI_MODELS or model.startswith(("gpt-", "o1", "o3", "o4"))

def _is_gemini(model):
    return model in GEMINI_MODELS or model.startswith("gemini")

def _is_deepseek(model):
    return model in DEEPSEEK_MODELS or model.startswith("deepseek")

def _is_qwen(model):
    return model in QWEN_MODELS or model.startswith("qwen")


def _allowed_objective_error(reference):
    return REL_TOLERANCE * abs(reference)


def _score_answer(reference, candidate):
    allowed_error = _allowed_objective_error(reference)
    if candidate is None:
        return None, None, allowed_error
    try:
        candidate = float(candidate)
    except (TypeError, ValueError):
        return False, None, allowed_error
    if not math.isfinite(reference) or not math.isfinite(candidate):
        return False, None, allowed_error
    objective_error = abs(reference - candidate)
    return objective_error <= allowed_error, objective_error, allowed_error


def _parse_levels(levels_str):
    levels = set()
    for part in levels_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            levels.update(range(int(start), int(end) + 1))
        else:
            levels.add(int(part))
    return levels


OPENAI_RESPONSES_MODELS = {"gpt-5.5", "gpt-5.4", "gpt-5.4-mini"}


def _evaluate_openai_responses(client, question, model):
    """Use the Responses API for GPT-5.x reasoning models.

    Effort is left at the model default (medium for gpt-5.5). OpenAI only exposes
    a reasoning *summary*, not the raw chain, and its summaries are too sparse to
    be useful, so we do not request them.
    """
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        reasoning={"effort": "medium"},
        max_output_tokens=MAX_OUTPUT_TOKENS,
        text={
            "format": {
                "type": "json_schema",
                "name": "math_answer",
                "schema": _ANSWER_SCHEMA,
                "strict": True,
            }
        },
    )

    text_content = ""
    for item in response.output:
        if item.type == "message":
            for c in (item.content or []):
                if c.type == "output_text":
                    text_content += c.text

    finish_reason = response.status
    input_tokens = response.usage.input_tokens if response.usage else None
    output_tokens = response.usage.output_tokens if response.usage else None

    raw_response = text_content

    if not text_content:
        logging.warning('Responses API returned no text (status=%s)', finish_reason)
        return None, raw_response, finish_reason, input_tokens, output_tokens
    try:
        return json.loads(text_content)["answer"], raw_response, finish_reason, input_tokens, output_tokens
    except (json.JSONDecodeError, KeyError):
        logging.warning('Failed to parse Responses API output (status=%s): %s', finish_reason, text_content[:80])
        return None, raw_response, finish_reason, input_tokens, output_tokens


def _evaluate_openai(client, question, model):
    if model in OPENAI_RESPONSES_MODELS:
        return _evaluate_openai_responses(client, question, model)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "math_answer",
                "schema": _ANSWER_SCHEMA,
                "strict": True,
            },
        },
    )
    content = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    input_tokens = response.usage.prompt_tokens if response.usage else None
    output_tokens = response.usage.completion_tokens if response.usage else None
    if not content:
        logging.warning('Model returned no content (finish_reason=%s)', finish_reason)
        return None, '', finish_reason, input_tokens, output_tokens
    try:
        return json.loads(content)["answer"], content, finish_reason, input_tokens, output_tokens
    except (json.JSONDecodeError, KeyError):
        logging.warning('Failed to parse structured output (finish_reason=%s): %s', finish_reason, content[:80])
        return None, content, finish_reason, input_tokens, output_tokens


def _evaluate_anthropic(client, question, model):
    """Evaluate a Claude model via the Messages API.

    Targets Claude Sonnet 5, which uses adaptive thinking: manual
    thinking={"type":"enabled","budget_tokens":N} is rejected with a 400 error, so
    there is no fixed thinking-budget knob. Reasoning depth is governed by
    effort="high" and bounded only by MAX_OUTPUT_TOKENS (thinking + answer share
    it). display="summarized" is required because Sonnet 5 defaults to "omitted"
    (empty thinking); Claude only ever returns a summary, never the raw chain.
    """
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive", "display": "summarized"},
                messages=[{"role": "user", "content": question}],
                output_config={"format": {"type": "json_schema", "schema": _ANSWER_SCHEMA}, "effort": "high"},
            )

            thinking_summary = ""
            text_content = ""
            for block in response.content:
                if block.type == "thinking":
                    thinking_summary = block.thinking or ""
                elif block.type == "text":
                    text_content = block.text or ""

            stop_reason = response.stop_reason
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            raw_response = (
                f"[THINKING SUMMARY]\n{thinking_summary}\n\n[RESPONSE]\n{text_content}"
                if thinking_summary else text_content
            )

            if not text_content:
                logging.warning('Model returned no text content (stop_reason=%s)', stop_reason)
                return None, raw_response, stop_reason, input_tokens, output_tokens
            try:
                return json.loads(text_content)["answer"], raw_response, stop_reason, input_tokens, output_tokens
            except (json.JSONDecodeError, KeyError):
                logging.warning('Failed to parse structured output (stop_reason=%s): %s', stop_reason, text_content[:80])
                return None, raw_response, stop_reason, input_tokens, output_tokens
        except anthropic.InternalServerError:
            if attempt < 2:
                logging.warning('Anthropic 500 error on attempt %d, retrying in 5s...', attempt + 1)
                time.sleep(5)
            else:
                logging.warning('Anthropic 500 error after 3 attempts, skipping problem')
                return None, '', 'error_500', None, None


_GEMINI_THINKING_MODELS = {"gemini-2.5-pro", "gemini-2.5-flash"}


def _evaluate_gemini(client, question, model):
    for attempt in range(3):
        try:
            thinking_config = (
                google_types.ThinkingConfig(thinking_budget=THINKING_BUDGET, include_thoughts=False)
                if model in _GEMINI_THINKING_MODELS else None
            )
            config = google_types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=_ANSWER_SCHEMA,
                temperature=0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                system_instruction=SYSTEM_PROMPT or None,
                thinking_config=thinking_config,
            )
            response = client.models.generate_content(
                model=model,
                contents=question,
                config=config,
            )
            thinking_text = ""
            content = ""
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if getattr(part, 'thought', False):
                        thinking_text += part.text or ""
                    else:
                        content += part.text or ""
            else:
                content = response.text or ""
            finish_reason = str(response.candidates[0].finish_reason) if response.candidates else 'unknown'
            input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else None
            output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else None
            if not content:
                logging.warning('Gemini returned no content (finish_reason=%s)', finish_reason)
                return None, thinking_text, finish_reason, input_tokens, output_tokens
            raw_response = (
                f"[THINKING]\n{thinking_text}\n\n[RESPONSE]\n{content}"
                if thinking_text else content
            )
            try:
                return json.loads(content)["answer"], raw_response, finish_reason, input_tokens, output_tokens
            except (json.JSONDecodeError, KeyError):
                logging.warning('Failed to parse Gemini structured output (finish_reason=%s): %s', finish_reason, content[:80])
                return None, raw_response, finish_reason, input_tokens, output_tokens
        except Exception as e:
            if attempt < 2:
                logging.warning('Gemini error on attempt %d (%s), retrying in 5s...', attempt + 1, e)
                time.sleep(5)
            else:
                logging.warning('Gemini error after 3 attempts, skipping problem: %s', e)
                return None, '', 'error', None, None


_DEEPSEEK_SYSTEM_PROMPT = (
    SYSTEM_PROMPT +
    "\n\nYou must respond in JSON format with exactly one key. Example:\n"
    '{"answer": 3.14}'
)


_DEEPSEEK_THINKING_MODELS = {"deepseek-v4-pro", "deepseek-v4-flash", "deepseek-reasoner"}


def _evaluate_deepseek(client, question, model):
    use_thinking = model in _DEEPSEEK_THINKING_MODELS
    for attempt in range(3):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": _DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )
            if use_thinking:
                kwargs["reasoning_effort"] = "high"
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            response = client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            content = msg.content or ''
            reasoning_content = getattr(msg, 'reasoning_content', None) or ''
            finish_reason = response.choices[0].finish_reason
            input_tokens = response.usage.prompt_tokens if response.usage else None
            output_tokens = response.usage.completion_tokens if response.usage else None
            if not content:
                logging.warning('DeepSeek returned no content (finish_reason=%s)', finish_reason)
                return None, reasoning_content, finish_reason, input_tokens, output_tokens
            model_answer = _extract_answer_from_text(content)
            if model_answer is None:
                logging.warning(
                    'Failed to extract answer (finish_reason=%s): %s',
                    finish_reason,
                    content[:120]
                )
            raw_response = (
                f"[REASONING]\n{reasoning_content}\n\n[RESPONSE]\n{content}"
                if reasoning_content else content
            )
            return model_answer, raw_response, finish_reason, input_tokens, output_tokens
        except Exception as e:
            if attempt < 2:
                logging.warning('DeepSeek error on attempt %d (%s), retrying in 5s...', attempt + 1, e)
                time.sleep(5)
            else:
                logging.warning('DeepSeek error after 3 attempts, skipping problem: %s', e)
                return None, '', 'error', None, None


def _evaluate_qwen(client, question, model):
    """Evaluate a Qwen model via the DashScope OpenAI-compatible endpoint.

    We evaluate Qwen3 thinking-only variants (e.g. qwen3-235b-a22b-thinking-2507).
    Per the Alibaba Cloud Model Studio docs these always reason and ignore
    enable_thinking, and open-source Qwen3 models support streaming output only.
    We therefore stream and cap reasoning at THINKING_BUDGET (matching the
    15k budget used for the other reasoning models) while capping the response at
    MAX_OUTPUT_TOKENS. thinking_budget is separate from max_tokens on
    DashScope; if left unset the model would default to its ~82k max chain-of-
    thought length. The answer is parsed from the accumulated text (response_format
    json is not documented as compatible with thinking mode).
    """
    for attempt in range(3):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                extra_body={"thinking_budget": THINKING_BUDGET},
                stream=True,
                stream_options={"include_usage": True},
            )
            reasoning_content = ""
            content = ""
            finish_reason = None
            input_tokens = None
            output_tokens = None
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if getattr(delta, "reasoning_content", None):
                    reasoning_content += delta.reasoning_content
                if getattr(delta, "content", None):
                    content += delta.content
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            if not content:
                logging.warning('Qwen returned no content (finish_reason=%s)', finish_reason)
                return None, reasoning_content, finish_reason, input_tokens, output_tokens
            model_answer = _extract_answer_from_text(content)
            if model_answer is None:
                logging.warning(
                    'Failed to extract Qwen answer (finish_reason=%s): %s',
                    finish_reason,
                    content[:120]
                )
            raw_response = (
                f"[REASONING]\n{reasoning_content}\n\n[RESPONSE]\n{content}"
                if reasoning_content else content
            )
            return model_answer, raw_response, finish_reason, input_tokens, output_tokens
        except Exception as e:
            if attempt < 2:
                logging.warning('Qwen error on attempt %d (%s), retrying in 5s...', attempt + 1, e)
                time.sleep(5)
            else:
                logging.warning('Qwen error after 3 attempts, skipping problem: %s', e)
                return None, '', 'error', None, None


def evaluate_problem(openai_client, anthropic_client, gemini_client, deepseek_client, qwen_client, question, model):
    if _is_anthropic(model):
        return _evaluate_anthropic(anthropic_client, question, model)
    if _is_openai(model):
        return _evaluate_openai(openai_client, question, model)
    if _is_gemini(model):
        return _evaluate_gemini(gemini_client, question, model)
    if _is_deepseek(model):
        return _evaluate_deepseek(deepseek_client, question, model)
    if _is_qwen(model):
        return _evaluate_qwen(qwen_client, question, model)
    raise ValueError(f'Unknown model: {model}')


def _build_output(model, problems, results, evaluated=None):
    answered = sum(1 for r in results if r.get('model_answer') is not None)
    correct_count = sum(1 for r in results if r.get('correct'))
    total_input = sum(r.get('input_tokens') for r in results if r.get('input_tokens') is not None)
    total_output = sum(r.get('output_tokens') for r in results if r.get('output_tokens') is not None)
    token_count = sum(1 for r in results if r.get('input_tokens') is not None)
    return {
        'model': model,
        'total': len(problems),
        'evaluated': len(results) if evaluated is None else evaluated,
        'answered': answered,
        'correct': correct_count,
        'accuracy': correct_count / answered if answered > 0 else 0,
        'avg_input_tokens': total_input / token_count if token_count > 0 else 0,
        'avg_output_tokens': total_output / token_count if token_count > 0 else 0,
        'tolerance': {
            'type': 'relative',
            'rel_tolerance': REL_TOLERANCE,
        },
        'results': results,
    }


def evaluate_model(openai_client, anthropic_client, gemini_client, deepseek_client, qwen_client, model, problems, output_path):
    results = []

    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
            results = existing.get('results', [])
            for result in results:
                if 'expected_answer' not in result:
                    continue
                is_correct, objective_error, allowed_error = _score_answer(
                    float(result['expected_answer']),
                    result.get('model_answer'),
                )
                result['correct'] = is_correct
                result['objective_error'] = objective_error
                result['allowed_error'] = allowed_error
            logging.info('[%s] Resuming from %d/%d completed results', model, len(results), len(problems))
        except Exception:
            results = []

    start_index = len(results)
    output = _build_output(model, problems, results, evaluated=start_index)
    if start_index >= len(problems):
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

    for i, problem in enumerate(problems):
        if i < start_index:
            continue
        question = problem['question']
        expected = float(problem['answer'])

        logging.info('[%s] Problem %d/%d', model, i + 1, len(problems))
        model_answer, raw_response, finish_reason, input_tokens, output_tokens = evaluate_problem(
            openai_client, anthropic_client, gemini_client, deepseek_client, qwen_client, question, model
        )

        is_correct, objective_error, allowed_error = _score_answer(expected, model_answer)

        results.append({
            'question': question,
            'expected_answer': expected,
            'model_answer': model_answer,
            'raw_response': raw_response,
            'finish_reason': finish_reason,
            'correct': is_correct if model_answer is not None else None,
            'objective_error': objective_error,
            'allowed_error': allowed_error,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        })

        evaluated = i + 1
        output = _build_output(model, problems, results, evaluated=evaluated)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

    logging.info('[%s] Accuracy: %d/%d answered (%.1f%%)',
                 model,
                 output['correct'],
                 output['answered'],
                 100 * output['correct'] / output['answered'] if output['answered'] > 0 else 0)
    return output

def _extract_answer_from_text(content):
    if not content:
        return None

    # Remove markdown fences
    content = content.strip()
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    # Try normal JSON first
    try:
        parsed = json.loads(content)
        answer = parsed.get("answer")
        if isinstance(answer, (int, float)):
            return float(answer)
        if isinstance(answer, str):
            return float(answer)
    except Exception:
        pass

    # Try to find "answer": 47.7439
    match = re.search(
        r'"answer"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)',
        content
    )
    if match:
        return float(match.group(1))

    # Try to find fractions like "answer": 55/18
    match = re.search(
        r'"answer"\s*:\s*(-?\d+)\s*/\s*(-?\d+)',
        content
    )
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator != 0:
            return numerator / denominator

    # Last fallback: grab the last decimal-looking number
    numbers = re.findall(
        r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',
        content
    )
    if numbers:
        return float(numbers[-1])

    return None

def main(unused_argv):
    openai_key = os.environ.get('OPENAI_API_KEY')
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    gemini_key = os.environ.get('GEMINI_API_KEY')
    deepseek_key = os.environ.get('DEEPSEEK_API_KEY')
    qwen_key = os.environ.get('DASHSCOPE_API_KEY')

    openai_client = OpenAI(api_key=openai_key) if openai_key else None
    anthropic_client = anthropic.Anthropic(api_key=anthropic_key) if anthropic_key else None
    gemini_client = google_genai.Client(api_key=gemini_key) if gemini_key else None
    deepseek_client = OpenAI(api_key=deepseek_key, base_url='https://api.deepseek.com') if deepseek_key else None
    qwen_client = OpenAI(api_key=qwen_key, base_url='https://dashscope-intl.aliyuncs.com/compatible-mode/v1') if qwen_key else None

    for model in ENGINES:
        if _is_anthropic(model) and anthropic_client is None:
            logging.fatal('ANTHROPIC_API_KEY not set but model %s requires it', model)
            return
        if _is_openai(model) and openai_client is None:
            logging.fatal('OPENAI_API_KEY not set but model %s requires it', model)
            return
        if _is_gemini(model) and gemini_client is None:
            logging.fatal('GEMINI_API_KEY not set but model %s requires it', model)
            return
        if _is_deepseek(model) and deepseek_client is None:
            logging.fatal('DEEPSEEK_API_KEY not set but model %s requires it', model)
            return
        if _is_qwen(model) and qwen_client is None:
            logging.fatal('DASHSCOPE_API_KEY not set but model %s requires it', model)
            return

    if not FLAGS.input_json and not FLAGS.input_dir:
        logging.fatal('Must specify either --input_json or --input_dir')
        return

    allowed_levels = _parse_levels(FLAGS.levels) if FLAGS.levels else None

    jobs = []
    if FLAGS.input_dir:
        input_dir = os.path.expanduser(FLAGS.input_dir)
        for level_name in sorted(os.listdir(input_dir)):
            level_path = os.path.join(input_dir, level_name)
            if not os.path.isdir(level_path):
                continue
            if allowed_levels is not None:
                try:
                    level_num = int(level_name.split('-')[-1])
                except ValueError:
                    continue
                if level_num not in allowed_levels:
                    continue
            for fname in os.listdir(level_path):
                if fname.endswith('.json'):
                    module_name = fname[:-5]
                    jobs.append((os.path.join(level_path, fname), level_name, module_name))
    else:
        level = os.path.basename(os.path.dirname(os.path.abspath(FLAGS.input_json)))
        module_name = os.path.basename(FLAGS.input_json)[:-5]
        jobs.append((FLAGS.input_json, level, module_name))

    for input_path, level, module_name in jobs:
        with open(input_path, 'r') as f:
            problems = json.load(f)

        output_dir = os.path.expanduser(os.path.join(FLAGS.output_dir, level))
        os.makedirs(output_dir, exist_ok=True)

        for model in ENGINES:
            logging.info('Evaluating model=%s level=%s module=%s', model, level, module_name)
            output_path = os.path.join(output_dir, module_name + '__' + model.replace('/', '_') + '.json')
            evaluate_model(openai_client, anthropic_client, gemini_client, deepseek_client, qwen_client, model, problems, output_path)
            logging.info('Results written to %s', output_path)

    logging.info('All evaluations complete.')


if __name__ == '__main__':
    app.run(main)
