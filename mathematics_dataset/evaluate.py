"""Evaluate an LLM on generated math problems.

Reads a generated JSON file of question/answer pairs, sends each question
to OpenAI models listed in ENGINES, and compares the response to the verified
answer.

Usage:
  export OPENAI_API_KEY=sk-...
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

FLAGS = flags.FLAGS

flags.DEFINE_string('input_json', None, 'Path to generated JSON file to evaluate')
flags.DEFINE_string('output_dir', None, 'Directory to write per-model evaluation results')
flags.mark_flag_as_required('input_json')
flags.mark_flag_as_required('output_dir')

############## Configurations ###############
ENGINES = [
    "gpt-4o-mini",
    #"gpt-5-mini",
    "gpt-4.1-mini",
    "gpt-4.1",
    # "gpt-5",
    # "gpt-5-nano",
    # "o4-mini",
    # "o3",
    # "o3-mini",
    # "o3-pro",
    # "o1",
    # "o1-pro",
    # "gpt-4.1",
    # "gpt-4.1-mini",
    # "gpt-4o",
]

TOLERANCE = 0.01
MAX_COMPLETION_TOKENS = 16000
#############################################

SYSTEM_PROMPT = (
    "You are a math solver. The user will give you an optimization problem. "
    "Respond with ONLY the optimal numeric value, nothing else. "
    "No words, no units, no explanation. Do not include any reasoning, steps, "
    "or explanation. Output a single number only."
)


def evaluate_problem(client, question, model):
  """Sends a question to the model and returns the response and parsed float."""
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
  raw_response = content if content else ''
  if content is None:
    logging.warning('Model returned no content (finish_reason=%s)', finish_reason)
    return None, raw_response, finish_reason
  answer_text = content.strip()
  try:
    return float(answer_text), raw_response, finish_reason
  except ValueError:
    # Fallback: extract the last number from the response
    matches = re.findall(r'-?\d+\.?\d*', answer_text)
    if matches:
      logging.warning('Extracted number from response: %s -> %s', answer_text, matches[-1])
      return float(matches[-1]), raw_response, finish_reason
    logging.warning('Could not parse model response as float: %s', answer_text)
    return None, raw_response, finish_reason


def evaluate_model(client, model, problems):
  """Evaluates a single model on all problems. Returns results dict."""
  results = []
  correct_count = 0

  for i, problem in enumerate(problems):
    question = problem['question']
    expected = float(problem['answer'])

    logging.info('[%s] Evaluating problem %d/%d', model, i + 1, len(problems))
    model_answer, raw_response, finish_reason = evaluate_problem(client, question, model)

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

  accuracy = correct_count / len(problems) if problems else 0
  logging.info('[%s] Accuracy: %d/%d (%.1f%%)',
               model, correct_count, len(problems), 100 * accuracy)

  return {
      'model': model,
      'total': len(problems),
      'correct': correct_count,
      'accuracy': accuracy,
      'tolerance': TOLERANCE,
      'results': results,
  }


def main(unused_argv):
  api_key = os.environ.get('OPENAI_API_KEY')
  if not api_key:
    logging.fatal('OPENAI_API_KEY environment variable not set')
    return

  client = OpenAI(api_key=api_key)

  with open(FLAGS.input_json, 'r') as f:
    problems = json.load(f)

  output_dir = os.path.expanduser(FLAGS.output_dir)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  for model in ENGINES:
    logging.info('Starting evaluation with model: %s', model)
    output = evaluate_model(client, model, problems)

    path = os.path.join(output_dir, model + '.json')
    with open(path, 'w') as f:
      json.dump(output, f, indent=2)
    logging.info('Results written to %s', path)

  logging.info('All evaluations complete.')


if __name__ == '__main__':
  app.run(main)
