"""Evaluate an LLM on generated math problems.

Reads a generated JSON file of question/answer pairs, sends each question
to an LLM, and compares the response to the verified answer.

Usage:
  export OPENAI_API_KEY=sk-...
  export ANTHROPIC_API_KEY=sk-ant-...
  python -m mathematics_dataset.evaluate \
    --input_json=output_json/train-easy/quadratic_programming__quadratic_programming.json \
    --output_dir=eval_results
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import re

from absl import app
from absl import flags
from absl import logging
from openai import OpenAI
import anthropic

FLAGS = flags.FLAGS

flags.DEFINE_string('input_json', None, 'Path to generated JSON file to evaluate')
flags.DEFINE_string('output_dir', None, 'Directory to write per-model evaluation results')
flags.mark_flag_as_required('input_json')
flags.mark_flag_as_required('output_dir')

############## Configurations ###############
ENGINES = [
    # "gpt-4o-mini",
    # "gpt-4.1-mini",
    # "gpt-4.1",
    # "gpt-5",
    # "o4-mini",
    # "o3",
    "claude-opus-4-7",
    # "claude-sonnet-4-6",
    # "claude-haiku-4-5-20251001",
]

TOLERANCE = 0.01
MAX_COMPLETION_TOKENS = 16000
#############################################

# Models that route to the Anthropic API
ANTHROPIC_MODELS = {
    "claude-opus-4-7",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
}

SYSTEM_PROMPT = (
    "You are a math solver. The user will give you an optimization problem. "
    "Respond with ONLY the optimal numeric value, nothing else. "
    "No words, no units, no explanation. Do not include any reasoning, steps, "
    "or explanation. Output a single number only."
)


def _is_anthropic(model):
    return model in ANTHROPIC_MODELS or model.startswith("claude")


def _parse_answer(answer_text):
    """Parse a float from model output. Returns (float or None)."""
    try:
        return float(answer_text.strip())
    except ValueError:
        matches = re.findall(r'-?\d+\.?\d*', answer_text)
        if matches:
            logging.warning('Extracted number from response: %s -> %s', answer_text, matches[-1])
            return float(matches[-1])
        logging.warning('Could not parse model response as float: %s', answer_text)
        return None


def _evaluate_openai(client, question, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=1,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    content = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    raw = content if content else ''
    if not content:
        logging.warning('Model returned no content (finish_reason=%s)', finish_reason)
        return None, raw, finish_reason
    return _parse_answer(content), raw, finish_reason


def _evaluate_anthropic(client, question, model):
    response = client.messages.create(
        model=model,
        max_tokens=MAX_COMPLETION_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    content = response.content[0].text if response.content else ''
    stop_reason = response.stop_reason
    raw = content if content else ''
    if not content:
        logging.warning('Model returned no content (stop_reason=%s)', stop_reason)
        return None, raw, stop_reason
    return _parse_answer(content), raw, stop_reason


def evaluate_problem(openai_client, anthropic_client, question, model):
    if _is_anthropic(model):
        return _evaluate_anthropic(anthropic_client, question, model)
    return _evaluate_openai(openai_client, question, model)


def evaluate_model(openai_client, anthropic_client, model, problems, output_path):
    """Evaluates a model on all problems, writing results to file after each answer."""
    results = []
    correct_count = 0

    for i, problem in enumerate(problems):
        question = problem['question']
        expected = float(problem['answer'])

        logging.info('[%s] Problem %d/%d', model, i + 1, len(problems))
        model_answer, raw_response, finish_reason = evaluate_problem(
            openai_client, anthropic_client, question, model
        )

        is_correct = (
            model_answer is not None
            and abs(expected - model_answer) <= TOLERANCE
        )
        if is_correct:
            correct_count += 1

        results.append({
            'question': question,
            'expected_answer': expected,
            'model_answer': model_answer,
            'raw_response': raw_response,
            'finish_reason': finish_reason,
            'correct': is_correct,
        })

        # Write after every answer so progress is never lost
        evaluated = i + 1
        output = {
            'model': model,
            'total': len(problems),
            'evaluated': evaluated,
            'correct': correct_count,
            'accuracy': correct_count / evaluated,
            'tolerance': TOLERANCE,
            'results': results,
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

    logging.info('[%s] Accuracy: %d/%d (%.1f%%)',
                 model, correct_count, len(problems), 100 * correct_count / len(problems))
    return output


def main(unused_argv):
    openai_key = os.environ.get('OPENAI_API_KEY')
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')

    openai_client = OpenAI(api_key=openai_key) if openai_key else None
    anthropic_client = anthropic.Anthropic(api_key=anthropic_key) if anthropic_key else None

    # Validate that required clients are available
    for model in ENGINES:
        if _is_anthropic(model) and anthropic_client is None:
            logging.fatal('ANTHROPIC_API_KEY not set but model %s requires it', model)
            return
        if not _is_anthropic(model) and openai_client is None:
            logging.fatal('OPENAI_API_KEY not set but model %s requires it', model)
            return

    with open(FLAGS.input_json, 'r') as f:
        problems = json.load(f)

    output_dir = os.path.expanduser(FLAGS.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for model in ENGINES:
        logging.info('Starting evaluation with model: %s', model)
        output_path = os.path.join(output_dir, model.replace('/', '_') + '.json')
        evaluate_model(openai_client, anthropic_client, model, problems, output_path)
        logging.info('Results written to %s', output_path)

    logging.info('All evaluations complete.')


if __name__ == '__main__':
    app.run(main)
