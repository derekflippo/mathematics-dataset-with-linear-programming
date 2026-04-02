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

  # exponents
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

  constraint_strings = [
      f"{b}x + {c}y <= {d}",
      f"{e}xy <= {g}",
      "xy >= 1"
  ]

  # additional constraints
  if random.random() < 0.5:
    h = number.integer(entropy/3, signed=False, min_abs=1)
    constraints.append(x <= h)
    constraint_strings.append(f"x <= {h}")

  if random.random() < 0.5:
    constraints.append(y <= g)
    constraint_strings.append(f"y <= {g}")

  if random.random() < 0.5:
    constraints.append(x**2 * y <= g + d)
    constraint_strings.append(f"x^2 y <= {g + d}")

  if random.random() < 0.5:
    constraints.append(x / y <= g)
    constraint_strings.append(f"x/y <= {g}")

  if random.random() < 0.5:
    constraints.append(b*x + c*y + e*x*y <= d + g)
    constraint_strings.append(f"{b}x + {c}y + {e}xy <= {d + g}")

  prob = cp.Problem(objective, constraints)
  prob.solve(gp=True, solver=cp.SCS, eps=1e-6, max_iters=5000)

  if prob.status not in ["optimal", "optimal_inaccurate"]:
    raise ValueError("GP solve failed with status: {}".format(prob.status))

  constraints_text = ", ".join(constraint_strings)

  template = random.choice([
      ('Find the minimum value of {a}x^{p}y^{q} subject to {constraints}, '
       'with x,y > 0.'),

      ('Minimize f(x,y) = {a}x^{p}y^{q} given that {constraints}. '
       'What is the minimum value?'),

      ('Suppose x,y > 0 satisfy {constraints}. '
       'Determine the smallest possible value of {a}x^{p}y^{q}.'),

      ('Under the constraints {constraints}, '
       'find the minimum value of {a}x^{p}y^{q}.')
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
          q=q,
          constraints=constraints_text
      ),
      answer=round(float(prob.value), 3)
  )