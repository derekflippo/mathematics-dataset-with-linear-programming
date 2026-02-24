"""Linear programming questions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random
import cvxpy as cp
import numpy as np

from mathematics_dataset import example
from mathematics_dataset.util import composition
from mathematics_dataset.sample import number

_ENTROPY_TRAIN = (2, 4)
_ENTROPY_INTERPOLATE = (8, 8)
_ENTROPY_EXTRAPOLATE = (12, 12)


def _make_modules(entropy):
  return {
      'basic_linear_programming': functools.partial(basic_linear_programming, *entropy),
      'non_trivial_linear_programming': functools.partial(non_trivial_linear_programming, *entropy),
  }


def train(entropy_fn):
  return _make_modules(entropy_fn(_ENTROPY_TRAIN))


def test():
  return _make_modules(_ENTROPY_INTERPOLATE)


def test_extra():
  return _make_modules(_ENTROPY_EXTRAPOLATE)


#this returns example object
def basic_linear_programming(min_entropy, max_entropy):
  entropy = random.uniform(min_entropy, max_entropy)
  context = composition.Context()

  c1 = number.integer(entropy / 3, signed=False, min_abs=1)
  c2 = number.integer(entropy / 3, signed=False, min_abs=1)
  c3 = number.integer(entropy / 3, signed=False, min_abs=2)
  #(x,y) 
  corners = [(0,0), (c3,0), (0,c3)]
  answer = max(c1*x + c2*y for x, y in corners)
  template = random.choice([
      'What is the maximum value of the objective function: {c1}x + {c2}y \n Given: x+y<={c3} , x,y>=0',
  ])

  return example.Problem(
      question=example.question(context, template, c1 = c1, c2=c2,c3=c3), 
      answer=answer)

def non_trivial_linear_programming(min_entropy, max_entropy):
  entropy = random.uniform(min_entropy, max_entropy)
  context = composition.Context()

  m = 10
  n = 10
  # m = number.integer(entropy/2, signed=False, min_abs=1)
  # n = number.integer(entropy/2, signed=False, min_abs=1)
  s0 = np.random.randn(m)
  lamb0 = np.maximum(-s0, 0)
  s0 = np.maximum(s0, 0)
  x0 = np.random.randn(n)
  A = np.random.randn(m, n)
  b = A @ x0 + s0
  c = -A.T @ lamb0

  # Define and solve the CVXPY problem.
  x = cp.Variable(n)
  prob = cp.Problem(cp.Minimize(c.T@x),
                  [A @ x <= b])
  prob.solve()

  template = random.choice([
      'Minimize {c}' ,
  ])
  return example.Problem(
      question=example.question(context, template, c=c), 
      answer=prob.value)


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