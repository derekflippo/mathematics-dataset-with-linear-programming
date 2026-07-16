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
from datetime import datetime, timezone

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
    "gpt-5.5",
    # "gpt-5.4",
    # "gpt-5.4-mini",
    # "claude-opus-4-7",
    "claude-sonnet-5",
    # "claude-sonnet-4-6",
    # "claude-haiku-4-5-20251001",
    # "deepseek-chat",
    # "deepseek-reasoner",
    "deepseek-v4-pro",
    # "gemini-2.5-pro",
    # "gemini-2.5-flash",
    "qwen3-235b-a22b-thinking-2507",
]

ABS_TOLERANCE = 1e-2
REL_TOLERANCE = 1e-3

def _allowed_objective_error(reference):
    return max(
        ABS_TOLERANCE,
        REL_TOLERANCE * abs(reference),
    )

MAX_API_ATTEMPTS = 3
INITIAL_RETRY_DELAY = 5
MAX_RETRY_DELAY = 20


def _utc_now_iso():
    """Return an ISO-8601 UTC timestamp with an explicit Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _error_status_code(exc):
    """Best-effort extraction of an HTTP-like status code from an SDK error."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(exc, "code", None)
    try:
        return int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable_provider_error(exc):
    """Return True only for temporary provider or transport failures."""
    status_code = _error_status_code(exc)

    if status_code in {408, 409, 429}:
        return True
    if status_code is not None and 500 <= status_code < 600:
        return True

    retryable_exception_names = {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "DeadlineExceeded",
        "ResourceExhausted",
        "TimeoutError",
        "ConnectionError",
    }
    return type(exc).__name__ in retryable_exception_names


def _run_with_retries(request_fn, model, provider):
    """Run one request under one provider-independent retry policy.

    Parsing is intentionally outside this function. A malformed model response is
    therefore never retried, while transient transport/provider failures receive
    the same attempt count and deterministic backoff for every provider.
    """
    retry_errors = []
    retry_delays = []
    request_started_at = _utc_now_iso()
    request_started_monotonic = time.monotonic()

    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            payload = request_fn()
            return payload, {
                'request_started_at': request_started_at,
                'request_completed_at': _utc_now_iso(),
                'elapsed_seconds': round(time.monotonic() - request_started_monotonic, 6),
                'api_attempts': attempt,
                'retry_count': attempt - 1,
                'retry_delays_seconds': retry_delays,
                'retry_errors': retry_errors,
            }
        except Exception as exc:
            retryable = _is_retryable_provider_error(exc)
            status_code = _error_status_code(exc)
            error_record = {
                'attempt': attempt,
                'error_type': type(exc).__name__,
                'status_code': status_code,
                'retryable': retryable,
                'message': str(exc),
            }
            retry_errors.append(error_record)

            if not retryable:
                logging.error(
                    "[%s] Permanent %s error on attempt %d/%d: %s",
                    model, provider, attempt, MAX_API_ATTEMPTS, exc,
                )
            elif attempt == MAX_API_ATTEMPTS:
                logging.warning(
                    "[%s] %s failed after %d attempts: %s",
                    model, provider, attempt, exc,
                )
            else:
                delay = min(
                    INITIAL_RETRY_DELAY * (2 ** (attempt - 1)),
                    MAX_RETRY_DELAY,
                )
                retry_delays.append(delay)
                logging.warning(
                    "[%s] Temporary %s error on attempt %d/%d: %s. Retrying in %ds.",
                    model, provider, attempt, MAX_API_ATTEMPTS, exc, delay,
                )
                time.sleep(delay)
                continue

            error_payload = {
                'response_text': '',
                'raw_response': '',
                'finish_reason': 'api_error',
                'input_tokens': None,
                'output_tokens': None,
                'api_error': error_record,
            }
            return error_payload, {
                'request_started_at': request_started_at,
                'request_completed_at': _utc_now_iso(),
                'elapsed_seconds': round(time.monotonic() - request_started_monotonic, 6),
                'api_attempts': attempt,
                'retry_count': attempt - 1,
                'retry_delays_seconds': retry_delays,
                'retry_errors': retry_errors,
            }

    raise AssertionError('Retry loop exited unexpectedly')

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

GENERIC_JSON_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + ' Return only valid JSON with exactly one key named answer, for example: '
    + '{"answer": 3.14}'
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


def _provider_payload(response_text, raw_response, finish_reason,
                      input_tokens, output_tokens, api_error=None,
                      reasoning_tokens=None, total_tokens=None):
    """Normalize every provider response before the shared parser runs."""
    return {
        'response_text': response_text or '',
        'raw_response': raw_response or '',
        'finish_reason': finish_reason,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'reasoning_tokens': reasoning_tokens,
        'total_tokens': total_tokens,
        'api_error': api_error,
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

    Reasoning effort is explicitly set to medium. OpenAI only exposes
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
    output_details = getattr(response.usage, 'output_tokens_details', None) if response.usage else None
    reasoning_tokens = getattr(output_details, 'reasoning_tokens', None)
    total_tokens = getattr(response.usage, 'total_tokens', None) if response.usage else None

    raw_response = text_content

    if not text_content:
        logging.warning('Responses API returned no text (status=%s)', finish_reason)
    return _provider_payload(
        text_content, raw_response, finish_reason, input_tokens, output_tokens,
        reasoning_tokens=reasoning_tokens, total_tokens=total_tokens,
    )


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
    completion_details = getattr(response.usage, 'completion_tokens_details', None) if response.usage else None
    reasoning_tokens = getattr(completion_details, 'reasoning_tokens', None)
    total_tokens = getattr(response.usage, 'total_tokens', None) if response.usage else None
    if not content:
        logging.warning('Model returned no content (finish_reason=%s)', finish_reason)
    return _provider_payload(
        content, content, finish_reason, input_tokens, output_tokens,
        reasoning_tokens=reasoning_tokens, total_tokens=total_tokens,
    )


def _evaluate_anthropic(client, question, model):
    """Evaluate a Claude model via the Messages API."""
    response = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive", "display": "summarized"},
        messages=[{"role": "user", "content": question}],
        output_config={
            "format": {"type": "json_schema", "schema": _ANSWER_SCHEMA},
            "effort": "high",
        },
    )

    thinking_summary = ""
    text_content = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_summary += block.thinking or ""
        elif block.type == "text":
            text_content += block.text or ""

    stop_reason = response.stop_reason
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    total_tokens = input_tokens + output_tokens
    raw_response = (
        f"[THINKING SUMMARY]\n{thinking_summary}\n\n[RESPONSE]\n{text_content}"
        if thinking_summary else text_content
    )

    if not text_content:
        logging.warning('Model returned no text content (stop_reason=%s)', stop_reason)
    return _provider_payload(
        text_content, raw_response, stop_reason, input_tokens, output_tokens,
        total_tokens=total_tokens,
    )


_GEMINI_THINKING_MODELS = {"gemini-2.5-pro", "gemini-2.5-flash"}


def _evaluate_gemini(client, question, model):
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
    reasoning_tokens = response.usage_metadata.thoughts_token_count if response.usage_metadata else None
    total_tokens = response.usage_metadata.total_token_count if response.usage_metadata else None
    if not content:
        logging.warning('Gemini returned no content (finish_reason=%s)', finish_reason)
    raw_response = (
        f"[THINKING]\n{thinking_text}\n\n[RESPONSE]\n{content}"
        if thinking_text else content
    )
    return _provider_payload(
        content, raw_response, finish_reason, input_tokens, output_tokens,
        reasoning_tokens=reasoning_tokens, total_tokens=total_tokens,
    )


_DEEPSEEK_THINKING_MODELS = {"deepseek-v4-pro", "deepseek-v4-flash", "deepseek-reasoner"}


def _provider_for_model(model):
    if _is_anthropic(model):
        return 'Anthropic'
    if _is_openai(model):
        return 'OpenAI'
    if _is_gemini(model):
        return 'Gemini'
    if _is_deepseek(model):
        return 'DeepSeek'
    if _is_qwen(model):
        return 'Qwen'
    raise ValueError(f'Unknown model: {model}')


def _request_configuration(model):
    """Return the exact request settings that should be recorded in output."""
    if _is_deepseek(model) or _is_qwen(model):
        system_prompt = GENERIC_JSON_SYSTEM_PROMPT
    else:
        system_prompt = SYSTEM_PROMPT

    thinking_budget = None
    if model in _GEMINI_THINKING_MODELS or _is_qwen(model):
        thinking_budget = THINKING_BUDGET

    thinking_mode = None
    if _is_anthropic(model):
        thinking_mode = 'adaptive'
    elif model in _GEMINI_THINKING_MODELS:
        thinking_mode = 'enabled'
    elif model in _DEEPSEEK_THINKING_MODELS:
        thinking_mode = 'enabled'
    elif _is_qwen(model):
        thinking_mode = 'enabled'
    elif model in OPENAI_RESPONSES_MODELS:
        thinking_mode = 'reasoning'

    reasoning_effort = None
    if model in OPENAI_RESPONSES_MODELS:
        reasoning_effort = 'medium'
    elif _is_anthropic(model):
        reasoning_effort = 'high'
    elif model in _DEEPSEEK_THINKING_MODELS:
        reasoning_effort = 'high'

    if _is_openai(model) or _is_anthropic(model) or _is_gemini(model):
        output_format = 'json_schema'
    elif _is_deepseek(model):
        output_format = 'json_object'
    else:
        output_format = 'prompt_only_json'

    return {
        'provider': _provider_for_model(model),
        'model': model,
        'system_prompt': system_prompt,
        'thinking_mode': thinking_mode,
        'thinking_budget_tokens': thinking_budget,
        'max_output_tokens': MAX_OUTPUT_TOKENS,
        'reasoning_effort': reasoning_effort,
        'temperature': 0 if _is_gemini(model) else None,
        'output_format': output_format,
        'retry_policy': {
            'max_api_attempts': MAX_API_ATTEMPTS,
            'initial_retry_delay_seconds': INITIAL_RETRY_DELAY,
            'max_retry_delay_seconds': MAX_RETRY_DELAY,
            'backoff': 'deterministic_exponential',
            'parse_failures_are_retried': False,
        },
        'parser': {
            'name': 'shared_explicit_answer_parser',
            'accepted_key': 'answer',
            'last_number_fallback': False,
        },
    }


def _evaluate_deepseek(client, question, model):
    use_thinking = model in _DEEPSEEK_THINKING_MODELS
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": GENERIC_JSON_SYSTEM_PROMPT},
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
    completion_details = getattr(response.usage, 'completion_tokens_details', None) if response.usage else None
    reasoning_tokens = getattr(completion_details, 'reasoning_tokens', None)
    total_tokens = getattr(response.usage, 'total_tokens', None) if response.usage else None
    if not content:
        logging.warning('DeepSeek returned no content (finish_reason=%s)', finish_reason)
    raw_response = (
        f"[REASONING]\n{reasoning_content}\n\n[RESPONSE]\n{content}"
        if reasoning_content else content
    )
    return _provider_payload(
        content, raw_response, finish_reason, input_tokens, output_tokens,
        reasoning_tokens=reasoning_tokens, total_tokens=total_tokens,
    )


def _evaluate_qwen(client, question, model):
    """Evaluate a Qwen thinking model through the DashScope streaming endpoint."""
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GENERIC_JSON_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        extra_body={
            "enable_thinking": True,
            "thinking_budget": THINKING_BUDGET,
        },
        stream=True,
        stream_options={"include_usage": True},
    )

    reasoning_content = ""
    content = ""
    finish_reason = None
    input_tokens = None
    output_tokens = None
    reasoning_tokens = None
    total_tokens = None
    for chunk in stream:
        if getattr(chunk, "usage", None):
            input_tokens = chunk.usage.prompt_tokens
            output_tokens = chunk.usage.completion_tokens
            completion_details = getattr(chunk.usage, 'completion_tokens_details', None)
            reasoning_tokens = getattr(completion_details, 'reasoning_tokens', None)
            total_tokens = getattr(chunk.usage, 'total_tokens', None)
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
    raw_response = (
        f"[REASONING]\n{reasoning_content}\n\n[RESPONSE]\n{content}"
        if reasoning_content else content
    )
    return _provider_payload(
        content, raw_response, finish_reason, input_tokens, output_tokens,
        reasoning_tokens=reasoning_tokens, total_tokens=total_tokens,
    )


def evaluate_problem(openai_client, anthropic_client, gemini_client, deepseek_client, qwen_client, question, model):
    if _is_anthropic(model):
        request_fn = lambda: _evaluate_anthropic(anthropic_client, question, model)
    elif _is_openai(model):
        request_fn = lambda: _evaluate_openai(openai_client, question, model)
    elif _is_gemini(model):
        request_fn = lambda: _evaluate_gemini(gemini_client, question, model)
    elif _is_deepseek(model):
        request_fn = lambda: _evaluate_deepseek(deepseek_client, question, model)
    elif _is_qwen(model):
        request_fn = lambda: _evaluate_qwen(qwen_client, question, model)
    else:
        raise ValueError(f'Unknown model: {model}')

    provider = _provider_for_model(model)
    payload, retry_metadata = _run_with_retries(request_fn, model, provider)
    model_answer = _extract_answer_from_text(payload.get('response_text', ''))

    if payload.get('response_text') and model_answer is None:
        logging.warning(
            '[%s] Shared parser could not extract an explicit JSON answer: %s',
            model, payload['response_text'][:120],
        )

    return {
        'model_answer': model_answer,
        'raw_response': payload.get('raw_response', ''),
        'finish_reason': payload.get('finish_reason'),
        'input_tokens': payload.get('input_tokens'),
        'output_tokens': payload.get('output_tokens'),
        'reasoning_tokens': payload.get('reasoning_tokens'),
        'total_tokens': payload.get('total_tokens'),
        'api_error': payload.get('api_error'),
        'parse_success': model_answer is not None,
        **retry_metadata,
    }


def _build_output(model, problems, results, run_started_at,
                  runtime_seconds, evaluated=None):
    evaluated_count = len(results) if evaluated is None else evaluated
    answered = sum(1 for r in results if r.get('model_answer') is not None)
    correct_count = sum(1 for r in results if r.get('correct') is True)
    api_error_count = sum(1 for r in results if r.get('finish_reason') == 'api_error')
    parse_failure_count = sum(
        1 for r in results
        if r.get('finish_reason') != 'api_error' and r.get('model_answer') is None
    )

    input_values = [r.get('input_tokens') for r in results if r.get('input_tokens') is not None]
    output_values = [r.get('output_tokens') for r in results if r.get('output_tokens') is not None]
    reasoning_values = [
        r.get('reasoning_tokens') for r in results
        if r.get('reasoning_tokens') is not None
    ]
    total_token_values = [
        r.get('total_tokens') for r in results
        if r.get('total_tokens') is not None
    ]
    total_input = sum(input_values)
    total_output = sum(output_values)
    total_reasoning = sum(reasoning_values)

    return {
        'model': model,
        'run_started_at': run_started_at,
        'last_updated_at': _utc_now_iso(),
        'run_completed_at': _utc_now_iso() if evaluated_count >= len(problems) else None,
        'runtime_seconds': round(runtime_seconds, 6),
        'configuration': _request_configuration(model),
        'total': len(problems),
        'evaluated': evaluated_count,
        'answered': answered,
        'correct': correct_count,
        'accuracy': correct_count / evaluated_count if evaluated_count else 0,
        'answered_accuracy': correct_count / answered if answered else 0,
        'answer_rate': answered / evaluated_count if evaluated_count else 0,
        'api_error_count': api_error_count,
        'parse_failure_count': parse_failure_count,
        'total_input_tokens': total_input,
        'total_output_tokens': total_output,
        'total_reasoning_tokens': total_reasoning,
        'provider_total_tokens': sum(total_token_values),
        'input_token_samples': len(input_values),
        'output_token_samples': len(output_values),
        'reasoning_token_samples': len(reasoning_values),
        'provider_total_token_samples': len(total_token_values),
        'avg_input_tokens': total_input / len(input_values) if input_values else 0,
        'avg_output_tokens': total_output / len(output_values) if output_values else 0,
        'avg_reasoning_tokens': total_reasoning / len(reasoning_values) if reasoning_values else 0,
        'token_metric_note': 'Provider-reported token counts; definitions may differ across APIs.',
        'tolerance': {
            'type': 'absolute_or_relative',
            'absolute_tolerance': ABS_TOLERANCE,
            'relative_tolerance': REL_TOLERANCE,
            'formula': 'max(abs_tol, rel_tol * abs(reference))',
        },
        'results': results,
    }


def _write_json_atomic(output_path, payload):
    """Write JSON without leaving a partially written result file on interruption."""
    temporary_path = output_path + '.tmp'
    with open(temporary_path, 'w') as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_path, output_path)


def _validated_resume_results(existing, model, problems):
    """Return only a prefix that is aligned with the current input and config."""
    if existing.get('model') not in (None, model):
        logging.warning('[%s] Existing output belongs to model %s; starting over.',
                        model, existing.get('model'))
        return []

    existing_config = existing.get('configuration')
    current_config = _request_configuration(model)
    if existing_config is not None and existing_config != current_config:
        logging.warning('[%s] Existing output uses a different configuration; starting over.', model)
        return []

    raw_results = existing.get('results', [])
    if not isinstance(raw_results, list):
        logging.warning('[%s] Existing results are not a list; starting over.', model)
        return []

    valid_results = []
    for index, result in enumerate(raw_results):
        if index >= len(problems) or not isinstance(result, dict):
            break
        problem = problems[index]
        try:
            aligned = (
                result.get('question') == problem['question']
                and float(result.get('expected_answer')) == float(problem['answer'])
            )
        except (KeyError, TypeError, ValueError):
            aligned = False
        if not aligned:
            logging.warning('[%s] Resume mismatch at result %d; truncating resume prefix.',
                            model, index + 1)
            break

        is_correct, objective_error, allowed_error = _score_answer(
            float(result['expected_answer']), result.get('model_answer'),
        )
        result['correct'] = is_correct
        result['objective_error'] = objective_error
        result['allowed_error'] = allowed_error
        valid_results.append(result)

    return valid_results


def evaluate_model(openai_client, anthropic_client, gemini_client,
                   deepseek_client, qwen_client, model, problems, output_path):
    session_started_monotonic = time.monotonic()
    run_started_at = _utc_now_iso()
    prior_runtime_seconds = 0.0
    results = []

    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
            results = _validated_resume_results(existing, model, problems)
            if results:
                run_started_at = existing.get('run_started_at') or run_started_at
                prior_runtime_seconds = float(existing.get('runtime_seconds') or 0.0)
            logging.info('[%s] Resuming from %d/%d completed results',
                         model, len(results), len(problems))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logging.warning('[%s] Could not resume from %s: %s. Starting over.',
                            model, output_path, exc)
            results = []

    def current_runtime_seconds():
        return prior_runtime_seconds + (time.monotonic() - session_started_monotonic)

    start_index = len(results)
    output = _build_output(
        model, problems, results, run_started_at,
        current_runtime_seconds(), evaluated=start_index,
    )
    if start_index >= len(problems):
        _write_json_atomic(output_path, output)
        return output

    for i, problem in enumerate(problems):
        if i < start_index:
            continue
        question = problem['question']
        expected = float(problem['answer'])

        logging.info('[%s] Problem %d/%d', model, i + 1, len(problems))
        response_result = evaluate_problem(
            openai_client, anthropic_client, gemini_client,
            deepseek_client, qwen_client, question, model,
        )

        model_answer = response_result['model_answer']
        is_correct, objective_error, allowed_error = _score_answer(expected, model_answer)

        results.append({
            'problem_index': i,
            'model': model,
            'question': question,
            'expected_answer': expected,
            'model_answer': model_answer,
            'raw_response': response_result['raw_response'],
            'finish_reason': response_result['finish_reason'],
            'api_error': response_result['api_error'],
            'parse_success': response_result['parse_success'],
            'correct': is_correct,
            'objective_error': objective_error,
            'allowed_error': allowed_error,
            'input_tokens': response_result['input_tokens'],
            'output_tokens': response_result['output_tokens'],
            'reasoning_tokens': response_result['reasoning_tokens'],
            'total_tokens': response_result['total_tokens'],
            'request_started_at': response_result['request_started_at'],
            'request_completed_at': response_result['request_completed_at'],
            'elapsed_seconds': response_result['elapsed_seconds'],
            'api_attempts': response_result['api_attempts'],
            'retry_count': response_result['retry_count'],
            'retry_delays_seconds': response_result['retry_delays_seconds'],
            'retry_errors': response_result['retry_errors'],
        })

        evaluated = i + 1
        output = _build_output(
            model, problems, results, run_started_at,
            current_runtime_seconds(), evaluated=evaluated,
        )
        _write_json_atomic(output_path, output)

    logging.info('[%s] Accuracy: %d/%d evaluated (%.1f%%); answered=%d',
                 model,
                 output['correct'],
                 output['evaluated'],
                 100 * output['accuracy'],
                 output['answered'])
    return output

def _coerce_explicit_answer(value):
    """Convert one explicit answer value to a finite float."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        candidate = float(value)
        return candidate if math.isfinite(candidate) else None
    if not isinstance(value, str):
        return None

    value = value.strip()
    numeric = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    fraction = re.fullmatch(rf"({numeric})\s*/\s*({numeric})", value)
    if fraction:
        numerator = float(fraction.group(1))
        denominator = float(fraction.group(2))
        if denominator == 0:
            return None
        candidate = numerator / denominator
    else:
        try:
            candidate = float(value)
        except ValueError:
            return None
    return candidate if math.isfinite(candidate) else None


def _extract_answer_from_text(content):
    """Shared parser: extract only an explicitly labeled ``answer`` value.

    Every provider is passed through this exact function. The parser accepts a
    JSON object, a fenced JSON object, or a JSON-style ``"answer": value`` field.
    It never guesses from an unlabeled or final-occurring number.
    """
    if not content or not isinstance(content, str):
        return None

    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        parsed = None

    if isinstance(parsed, dict) and "answer" in parsed:
        return _coerce_explicit_answer(parsed["answer"])

    numeric = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    fraction_match = re.search(
        rf'"answer"\s*:\s*"?({numeric})\s*/\s*({numeric})"?',
        content,
    )
    if fraction_match:
        return _coerce_explicit_answer(
            f"{fraction_match.group(1)}/{fraction_match.group(2)}"
        )

    number_match = re.search(
        rf'"answer"\s*:\s*"?({numeric})"?',
        content,
    )
    if number_match:
        return _coerce_explicit_answer(number_match.group(1))

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

    if bool(FLAGS.input_json) == bool(FLAGS.input_dir):
        logging.fatal('Specify exactly one of --input_json or --input_dir')
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
            for fname in sorted(os.listdir(level_path)):
                if fname.endswith('.json'):
                    module_name = fname[:-5]
                    jobs.append((os.path.join(level_path, fname), level_name, module_name))
    else:
        input_json = os.path.expanduser(FLAGS.input_json)
        level = os.path.basename(os.path.dirname(os.path.abspath(input_json)))
        module_name = os.path.basename(input_json)[:-5]
        jobs.append((input_json, level, module_name))

    if not jobs:
        logging.fatal('No JSON evaluation files were found.')
        return

    for input_path, level, module_name in jobs:
        with open(input_path, 'r') as f:
            problems = json.load(f)
        if not isinstance(problems, list):
            raise ValueError(f'Expected a JSON list of problems in {input_path}')
        for index, problem in enumerate(problems):
            if not isinstance(problem, dict) or 'question' not in problem or 'answer' not in problem:
                raise ValueError(
                    f'Problem {index} in {input_path} must contain question and answer fields'
                )
            if not isinstance(problem['question'], str) or not problem['question'].strip():
                raise ValueError(f'Problem {index} in {input_path} has an invalid question')
            try:
                expected_answer = float(problem['answer'])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'Problem {index} in {input_path} has a non-numeric answer'
                ) from exc
            if not math.isfinite(expected_answer):
                raise ValueError(f'Problem {index} in {input_path} has a non-finite answer')

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
