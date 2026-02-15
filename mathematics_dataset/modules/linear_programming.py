"""Linear programming questions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random

from mathematics_dataset import example
from mathematics_dataset.util import composition


_ENTROPY_TRAIN = (3, 10)
_ENTROPY_INTERPOLATE = (8, 8)
_ENTROPY_EXTRAPOLATE = (12, 12)


def _make_modules(entropy):
  return {
      'placeholder': functools.partial(placeholder_problem, *entropy),
  }


def train(entropy_fn):
  return _make_modules(entropy_fn(_ENTROPY_TRAIN))


def test():
  return _make_modules(_ENTROPY_INTERPOLATE)


def test_extra():
  return _make_modules(_ENTROPY_EXTRAPOLATE)


def placeholder_problem(min_entropy, max_entropy):
  """A placeholder linear programming problem that returns a fixed answer."""
  context = composition.Context()

  template = random.choice([
      'What is the maximum value of the objective function? (placeholder)',
  ])

  return example.Problem(
      question=example.question(context, template),
      answer=42)


# def sequence_next_term(min_entropy, max_entropy):
#   """E.g., "What is the next term in the sequence 1, 2, 3?"."""
#   entropy = random.uniform(min_entropy, max_entropy)
#   context = composition.Context()
#   variable = sympy.Symbol(context.pop())

#   sequence = _PolynomialSequence(variable, entropy)
#   min_num_terms = sequence.min_num_terms
#   num_terms = random.randint(min_num_terms, min_num_terms + 3)
#   sequence_sample = [sequence.term(n + 1) for n in range(num_terms)]
#   sequence_sample = display.NumberList(sequence_sample)

#   template = random.choice([
#       'What is next in {sequence}?',
#       'What comes next: {sequence}?',
#       'What is the next term in {sequence}?',
#   ])
#   answer = sequence.term(num_terms + 1)

#   return example.Problem(
#       question=example.question(context, template, sequence=sequence_sample),
#       answer=answer)