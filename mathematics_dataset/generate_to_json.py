"""Write generated questions to JSON files.

Given an output directory, this will create the following subdirectories:

*   train-easy
*   train-medium
*   train-hard
*   interpolate
*   extrapolate

and populate each with a JSON file per module, where each file contains a list
of {"question": ..., "answer": ...} dictionaries.

Passing --train_split=False will create a single output directory 'train' for
training data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil

# Dependency imports
from absl import app
from absl import flags
from absl import logging
from mathematics_dataset import generate
import six
from six.moves import range

FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', None, 'Where to write output JSON')
flags.DEFINE_boolean('train_split', True,
                     'Whether to split training data by difficulty')
flags.DEFINE_string('levels', None, 'Levels to generate, e.g. "1-4" or "2,5,7"')
flags.mark_flag_as_required('output_dir')


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


def main(unused_argv):
  generate.init_modules(FLAGS.train_split)

  allowed_levels = _parse_levels(FLAGS.levels) if FLAGS.levels else None

  output_dir = os.path.expanduser(FLAGS.output_dir)
  if os.path.exists(output_dir):
    logging.info('Removing existing output dir %s', output_dir)
    shutil.rmtree(output_dir)
  logging.info('Writing to %s', output_dir)
  os.makedirs(output_dir)

  for regime, flat_modules in six.iteritems(generate.filtered_modules):
    if allowed_levels is not None:
      try:
        level_num = int(regime.split('-')[-1])
      except ValueError:
        continue
      if level_num not in allowed_levels:
        continue
    regime_dir = os.path.join(output_dir, regime)
    os.mkdir(regime_dir)
    per_module = generate.counts[regime]
    #writing loop
    for module_name, module in six.iteritems(flat_modules):
      problems = []
      for _ in range(per_module):
        problem, _ = generate.sample_from_module(module)
        problems.append({
            'question': str(problem.question),
            'answer': str(problem.answer),
            'level': regime,
        })
      path = os.path.join(regime_dir, module_name + '.json')
      with open(path, 'w') as json_file:
        json.dump(problems, json_file, indent=2)
      logging.info('Written %s', path)


if __name__ == '__main__':
  app.run(main)
