# Copyright 2018 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prints to stdout different curriculum questions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import textwrap

# Dependency imports
from absl import app
from absl import flags
from absl import logging
from mathematics_dataset import generate_settings
from mathematics_dataset.modules import modules
import six
from six.moves import range


FLAGS = flags.FLAGS

flags.DEFINE_string('filter', '', 'restrict to matching module names')
flags.DEFINE_integer('num_problems', 25, 'Number of problems per module per level')
flags.DEFINE_bool('show_dropped', False, 'Whether to print dropped questions')


NUM_LEVELS = 8

filtered_modules = collections.OrderedDict([])
counts = {}


def _filter_and_flatten(modules_):
  """Returns flattened dict, filtered according to FLAGS."""
  flat = collections.OrderedDict()

  def add(submodules, prefix=None):
    for key, module_or_function in six.iteritems(submodules):
      full_name = prefix + '__' + key if prefix is not None else key
      if isinstance(module_or_function, dict):
        add(module_or_function, full_name)
      else:
        if FLAGS.filter not in full_name:
          continue
        flat[full_name] = module_or_function

  add(modules_)

  # Make sure list of modules are in deterministic order. This is important when
  # generating across multiple machines.
  flat = collections.OrderedDict(
      [(key, flat[key]) for key in sorted(six.iterkeys(flat))])

  return flat


def init_modules():
  """Inits the dicts containing functions for generating modules by level."""
  if filtered_modules:
    return  # already initialized

  for i in range(NUM_LEVELS):
    level_name = 'level-{}'.format(i + 1)
    filtered_modules[level_name] = _filter_and_flatten(modules.train(i))
    counts[level_name] = FLAGS.num_problems


def sample_from_module(module):
  """Samples a problem, ignoring samples with overly long questions / answers.

  Args:
    module: Callable returning a `Problem`.

  Returns:
    Pair `(problem, num_dropped)`, where `problem` is an instance of `Problem`
    and `num_dropped` is an integer >= 0 indicating the number of samples that
    were dropped.
  """
  num_dropped = 0
  while True:
    problem = module()
    question = str(problem.question)
    if len(question) > generate_settings.MAX_QUESTION_LENGTH:
      num_dropped += 1
      if FLAGS.show_dropped:
        logging.warning('Dropping question: %s', question)
      continue
    answer = str(problem.answer)
    if len(answer) > generate_settings.MAX_ANSWER_LENGTH:
      num_dropped += 1
      if FLAGS.show_dropped:
        logging.warning('Dropping question with answer: %s', answer)
      continue
    return problem, num_dropped


def main(unused_argv):
  """Prints Q&As from modules according to FLAGS.filter."""
  init_modules()

  text_wrapper = textwrap.TextWrapper(
      width=80, initial_indent=' ', subsequent_indent='  ')

  for regime, flat_modules in six.iteritems(filtered_modules):
    per_module = counts[regime]
    for module_name, module in six.iteritems(flat_modules):
      # These magic print constants make the header bold.
      print('\033[1m{}/{}\033[0m'.format(regime, module_name))
      num_dropped = 0
      for _ in range(per_module):
        problem, extra_dropped = sample_from_module(module)
        num_dropped += extra_dropped
        text = text_wrapper.fill(
            '{}  \033[92m{}\033[0m'.format(problem.question, problem.answer))
        print(text)
      if num_dropped > 0:
        logging.warning('Dropped %d examples', num_dropped)


if __name__ == '__main__':
  app.run(main)
