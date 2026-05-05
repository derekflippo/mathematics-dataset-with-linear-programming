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

flags.DEFINE_string('input_json', None, 'Path to a single generated JSON file to evaluate')
flags.DEFINE_string('input_dir', None, 'Path to output_json directory to evaluate all levels')
flags.DEFINE_string('output_dir', None, 'Directory to write per-model evaluation results')
flags.DEFINE_string('levels', None, 'Levels to evaluate, e.g. "1-4" or "2,5,7" (only used with --input_dir)')
flags.mark_flag_as_required('output_dir')

############## Configurations ###############
ENGINES = [
    # "gpt-4o-mini",
    # "gpt-4.1-mini",
    # "gpt-4.1",
    # "gpt-5",
    # "o4-mini",
    # "o3",
    # "claude-opus-4-7",
    # "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
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
    # "You are a math solver. The user will give you an optimization problem. "
    # "You must solve it using only your own mathematical reasoning — no tools, no code, no solvers. "
    # "Do NOT use or call any of the following: Python, MATLAB, Julia, R, CVXPY, scipy, numpy, "
    # "Gurobi, CPLEX, MOSEK, or any other solver or programming language. "
    # "Show your reasoning in the 'reasoning' field, then provide the final numeric answer in the 'answer' field."
)


def _is_anthropic(model):
    return model in ANTHROPIC_MODELS or model.startswith("claude")


def _parse_answer(answer_text):
    """Parse a float from model output. Returns (float or None)."""
    text = answer_text.strip()

    # 1. Try the last non-empty line first — model consistently puts final answer there
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        last_line = lines[-1]
        try:
            return float(last_line)
        except ValueError:
            pass

    # 2. Direct float of entire response
    try:
        return float(text)
    except ValueError:
        pass

    # 3. LaTeX boxed: \boxed{...}
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        inner = boxed.group(1).strip()
        frac = re.search(r'\\d?frac\{(-?\d+)\}\{(\d+)\}', inner)
        if frac:
            return int(frac.group(1)) / int(frac.group(2))
        frac = re.search(r'(-?\d+)\s*/\s*(\d+)', inner)
        if frac:
            return int(frac.group(1)) / int(frac.group(2))
        try:
            return float(inner)
        except ValueError:
            pass

    # 4. Last number in entire text
    matches = re.findall(r'-?\d+\.?\d*', text)
    if matches:
        logging.warning('Falling back to last number: %s -> %s', text[:80], matches[-1])
        return float(matches[-1])

    logging.warning('Could not parse model response as float: %s', text[:80])
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
    return _parse_answer(content), raw, finish_reason, None, None


_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": "number"},
    },
    "required": ["reasoning", "answer"],
    "additionalProperties": False,
}

def _evaluate_anthropic(client, question, model):
    response = client.messages.create(
        model=model,
        max_tokens=MAX_COMPLETION_TOKENS,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
        output_config={"format": {"type": "json_schema", "schema": _ANSWER_SCHEMA}},
    )
    content = response.content[0].text if response.content else ''
    stop_reason = response.stop_reason
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    if not content:
        logging.warning('Model returned no content (stop_reason=%s)', stop_reason)
        return None, content, stop_reason, input_tokens, output_tokens
    try:
        return json.loads(content)["answer"], content, stop_reason, input_tokens, output_tokens
    except (json.JSONDecodeError, KeyError):
        logging.warning('Failed to parse structured output (stop_reason=%s): %s', stop_reason, content[:80])
        return None, content, stop_reason, input_tokens, output_tokens


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
        model_answer, raw_response, finish_reason, input_tokens, output_tokens = evaluate_problem(
            openai_client, anthropic_client, question, model
        )

        is_correct = (
            model_answer is not None
            and abs(expected - model_answer) <= TOLERANCE
        )
        if model_answer is not None:
            if is_correct:
                correct_count += 1

        results.append({
            'question': question,
            'expected_answer': expected,
            'model_answer': model_answer,
            'raw_response': raw_response,
            'finish_reason': finish_reason,
            'correct': is_correct if model_answer is not None else None,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        })

        # Write after every answer so progress is never lost
        evaluated = i + 1
        answered = sum(1 for r in results if r['model_answer'] is not None)
        output = {
            'model': model,
            'total': len(problems),
            'evaluated': evaluated,
            'answered': answered,
            'correct': correct_count,
            'accuracy': correct_count / answered if answered > 0 else 0,
            'tolerance': TOLERANCE,
            'results': results,
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

    logging.info('[%s] Accuracy: %d/%d answered (%.1f%%)',
                 model, correct_count, answered, 100 * correct_count / answered if answered > 0 else 0)
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

    if not FLAGS.input_json and not FLAGS.input_dir:
        logging.fatal('Must specify either --input_json or --input_dir')
        return

    # Parse --levels into a set of ints, e.g. "1-4" -> {1,2,3,4}, "2,5,7" -> {2,5,7}
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

    allowed_levels = _parse_levels(FLAGS.levels) if FLAGS.levels else None

    # Build list of (input_json_path, level) to evaluate
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
                    module_name = fname[:-5]  # strip .json
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
            logging.info('Starting evaluation with model: %s, level: %s, module: %s', model, level, module_name)
            output_path = os.path.join(output_dir, module_name + '__' + model.replace('/', '_') + '.json')
            evaluate_model(openai_client, anthropic_client, model, problems, output_path)
            logging.info('Results written to %s', output_path)

    logging.info('All evaluations complete.')


if __name__ == '__main__':
    app.run(main)
