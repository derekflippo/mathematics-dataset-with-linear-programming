"""Evaluate an LLM on generated math problems.

Reads a generated JSON file of question/answer pairs, sends each question
to an OpenAI model, and compares the response to the verified answer.

Usage:
  export OPENAI_API_KEY=sk-...
  python -m mathematics_dataset.evaluate \
    --input_json=output_json/train-easy/linear_programming__non_trivial_linear_programming.json \
    --output_json=eval_results.json
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
from absl import logging
from openai import OpenAI

FLAGS = flags.FLAGS

flags.DEFINE_string('input_json', None, 'Path to generated JSON file to evaluate')
flags.DEFINE_string('output_json', None, 'Path to write evaluation results')
flags.DEFINE_string('model', 'gpt-4o', 'OpenAI model to use')
flags.DEFINE_float('tolerance', 0.01, 'Tolerance for comparing float answers')
flags.mark_flag_as_required('input_json')
flags.mark_flag_as_required('output_json')

SYSTEM_PROMPT = (
    "You are a math solver. The user will give you an optimization problem. "
    "Respond with ONLY the optimal numeric value, nothing else. "
    "No words, no units, no explanation. Just the number."
)


def evaluate_problem(client, question, model):
  """Sends a question to the model and returns the response as a float."""
  response = client.chat.completions.create(
      model=model,
      messages=[
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": question},
      ],
      temperature=0,
  )
  answer_text = response.choices[0].message.content.strip()
  try:
    return float(answer_text)
  except ValueError:
    logging.warning('Could not parse model response as float: %s', answer_text)
    return None


def main(unused_argv):
  api_key = os.environ.get('OPENAI_API_KEY')
  if not api_key:
    logging.fatal('OPENAI_API_KEY environment variable not set')
    return

  client = OpenAI(api_key=api_key)

  with open(FLAGS.input_json, 'r') as f:
    problems = json.load(f)

  results = []
  correct_count = 0

  for i, problem in enumerate(problems):
    question = problem['question']
    expected = float(problem['answer'])

    logging.info('Evaluating problem %d/%d', i + 1, len(problems))
    model_answer = evaluate_problem(client, question, FLAGS.model)

    is_correct = (
        model_answer is not None
        and abs(expected - model_answer) <= FLAGS.tolerance
    )
    if is_correct:
      correct_count += 1

    results.append({
        'question': question,
        'expected_answer': expected,
        'model_answer': model_answer,
        'correct': is_correct,
    })

  output = {
      'model': FLAGS.model,
      'total': len(problems),
      'correct': correct_count,
      'accuracy': correct_count / len(problems) if problems else 0,
      'tolerance': FLAGS.tolerance,
      'results': results,
  }

  with open(FLAGS.output_json, 'w') as f:
    json.dump(output, f, indent=2)

  logging.info('Results written to %s', FLAGS.output_json)
  logging.info('Accuracy: %d/%d (%.1f%%)',
               correct_count, len(problems),
               100 * output['accuracy'])


if __name__ == '__main__':
  app.run(main)
