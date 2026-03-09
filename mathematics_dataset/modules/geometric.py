"""Geometric programming questions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random
import cvxpy as cp

from mathematics_dataset import example
from mathematics_dataset.util import composition
from mathematics_dataset.sample import number


_ENTROPY_TRAIN = (1, 4)
_ENTROPY_INTERPOLATE = (5, 6)
_ENTROPY_EXTRAPOLATE = (7, 8)


def _make_modules(entropy):
  return {
      'basic_geometric_programming':
          functools.partial(basic_geometric_programming, *entropy)
  }


def train(entropy_fn):
  return _make_modules(entropy_fn(_ENTROPY_TRAIN))


def test():
  return _make_modules(_ENTROPY_INTERPOLATE)


def test_extra():
  return _make_modules(_ENTROPY_EXTRAPOLATE)


def basic_geometric_programming(min_entropy, max_entropy):
  entropy = random.uniform(min_entropy, max_entropy)
  context = composition.Context()

  # coefficients
  a = number.integer(entropy/3, signed=False, min_abs=1)
  b = number.integer(entropy/3, signed=False, min_abs=1)
  c = number.integer(entropy/3, signed=False, min_abs=1)

  e = number.integer(entropy/3, signed=False, min_abs=1)
  f = number.integer(entropy/3, signed=False, min_abs=1)

  # guarantee feasibility
  d = b + c + number.integer(entropy/3, signed=False, min_abs=2)
  g = e + f + number.integer(entropy/3, signed=False, min_abs=2)

  # exponents for harder objective
  p = random.randint(1, 3)
  q = random.randint(1, 3)

  x = cp.Variable(pos=True)
  y = cp.Variable(pos=True)

  objective = cp.Minimize(a * x**p * y**q)

  constraints = [
    b*x + c*y <= d,
    e*x*y <= g,
    x*y >= 1
]

  prob = cp.Problem(objective, constraints)
  prob.solve(gp=True, solver=cp.SCS)

  if prob.status not in ["optimal", "optimal_inaccurate"]:
    raise ValueError("GP solve failed with status: {}".format(prob.status))

  template = random.choice([
      ('Minimize the geometric program: {a}x^{p}y^{q} subject to '
       '{b}x + {c}y <= {d} and {e}xy <= {g}, with x,y > 0. '
       'What is the optimal value?'),

      ('Consider the geometric program: minimize f(x,y) = {a}x^{p}y^{q} '
       'subject to {b}x + {c}y <= {d} and {e}xy <= {g}. '
       'Find the minimum value.'),

      ('Find the minimum of the geometric program: minimize {a}x^{p}y^{q} '
       'under constraints {b}x + {c}y <= {d} and {e}xy <= {g}, '
       'where x,y > 0.')
  ])

  return example.Problem(
      question=example.question(
          context,
          template,
          a=a,
          b=b,
          c=c,
          d=d,
          e=e,
          g=g,
          p=p,
          q=q
      ),
      answer=round(float(prob.value), 3))