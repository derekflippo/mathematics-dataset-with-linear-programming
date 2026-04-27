"""Geometric programming questions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random
import numpy as np
import cvxpy as cp

from mathematics_dataset import example
from mathematics_dataset.util import composition


_ENTROPY_TRAIN = (4, 10)
_ENTROPY_INTERPOLATE = (8, 8)
_ENTROPY_EXTRAPOLATE = (12, 12)


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


def _safe_round(x, ndigits=3):
  return round(float(x), ndigits)


def _make_monomial(x, exponents, coeff):
  term = coeff
  for i in range(len(exponents)):
    if exponents[i] != 0:
      term *= cp.power(x[i], exponents[i])
  return term


def _monomial_str(exponents, coeff):
  parts = []
  for i in range(len(exponents)):
    if exponents[i] == 1:
      parts.append(f"x{i+1}")
    elif exponents[i] > 1:
      parts.append(f"x{i+1}^{exponents[i]}")
    elif exponents[i] < 0:
      parts.append(f"x{i+1}^{exponents[i]}")

  if not parts:
    return str(coeff)

  if coeff == 1:
    return "*".join(parts)

  return str(coeff) + "*" + "*".join(parts)


def _difficulty_params(entropy):
  if entropy < 6:
    return dict(n=(2, 4), obj=(2, 3), cons=(2, 3), terms=(1, 2),
                exp=(-1, 2), coeff=5, tightness=(1.05, 1.2))
  elif entropy < 10:
    return dict(n=(3, 5), obj=(3, 4), cons=(3, 4), terms=(2, 3),
                exp=(-1, 2), coeff=7, tightness=(1.02, 1.15))
  else:
    return dict(n=(4, 5), obj=(4, 5), cons=(4, 5), terms=(2, 3),
                exp=(-2, 3), coeff=10, tightness=(1.01, 1.1))


def basic_geometric_programming(min_entropy, max_entropy):
  entropy = random.uniform(min_entropy, max_entropy)
  context = composition.Context()

  params = _difficulty_params(entropy)

  # Dimension and variables
  n = random.randint(*params["n"])
  x = cp.Variable(n, pos=True)

  # Feasible reference point
  x_star = np.random.uniform(1.2, 2.0, size=n)

  # Objective construction
  obj_terms = random.randint(*params["obj"])
  objective_expr = 0
  obj_strings = []

  used = set()
  for _ in range(obj_terms):
    coeff = random.randint(1, params["coeff"])

    while True:
      exponents = tuple(np.random.randint(params["exp"][0],
                                          params["exp"][1] + 1,
                                          size=n))

      if all(e == 0 for e in exponents):
        continue

      # Ensure every objective monomial has at least one negative exponent, so the objective is not trivially minimized by lower bounds only.
      if all(e >= 0 for e in exponents):
        exponents = list(exponents)
        exponents[random.randint(0, n - 1)] = -1
        exponents = tuple(exponents)

      if exponents not in used:
        used.add(exponents)
        break

    term = _make_monomial(x, exponents, coeff)
    objective_expr += term
    obj_strings.append(_monomial_str(exponents, coeff))

  objective = cp.Minimize(objective_expr)
  obj_str = " + ".join(obj_strings)

  constraints = []
  constraint_strings = []

  # Upper-bound posynomial constraints
  num_constraints = random.randint(*params["cons"])

  for _ in range(num_constraints):
    expr = 0
    expr_val = 0
    term_strings = []

    used = set()
    for _ in range(random.randint(*params["terms"])):
      coeff = random.randint(1, min(5, params["coeff"]))

      while True:
        exponents = tuple(np.random.randint(0,
                                            params["exp"][1] + 1,
                                            size=n))
        if all(e == 0 for e in exponents):
          continue
        if exponents not in used:
          used.add(exponents)
          break

      term = _make_monomial(x, exponents, coeff)
      expr += term

      val = coeff * np.prod(x_star ** exponents)
      expr_val += val

      term_strings.append(_monomial_str(exponents, coeff))

    bound = round(expr_val * random.uniform(*params["tightness"]), 2)

    constraints.append(expr <= bound)
    constraint_strings.append(
        "(" + " + ".join(term_strings) + f") <= {bound}"
    )

  # Lower coupling constraint
  i, j = random.sample(range(n), 2)
  val = x_star[i] * x_star[j]
  c = val * random.uniform(0.8, 1.2)

  rhs = round(1 / c, 2)

  constraints.append(1 / (x[i] * x[j]) <= rhs)
  constraint_strings.append(f"1/(x{i+1}*x{j+1}) <= {rhs}")

  # Variable bounds: 0.5 <= x_i <= 5
  for i in range(n):
    constraints.append(x[i] <= 5)
    constraints.append(1 / x[i] <= 2)

  prob = cp.Problem(objective, constraints)

  try:
    prob.solve(gp=True, solver=cp.SCS, eps=1e-6, max_iters=10000)
  except cp.SolverError:
    return basic_geometric_programming(min_entropy, max_entropy)

  if prob.status in ["infeasible", "unbounded", "infeasible_inaccurate", "unbounded_inaccurate"]:
    return basic_geometric_programming(min_entropy, max_entropy)

  constraints_text = ", ".join(constraint_strings)
  bounds_text = "0.5 ≤ x_i ≤ 5"

  template = random.choice([
      (
          "Find the minimum value of {obj} subject to {constraints}, "
          "{bounds}, with x_i > 0."
      ),
      (
          "Minimize f(x) = {obj} given that {constraints}, {bounds}. "
          "What is the minimum value?"
      ),
      (
          "Suppose x_i > 0 satisfy {constraints}, {bounds}. Determine "
          "the smallest possible value of {obj}."
      ),
  ])

  return example.Problem(
      question=example.question(
          context,
          template,
          obj=obj_str,
          constraints=constraints_text,
          bounds=bounds_text,
      ),
      answer=_safe_round(prob.value, 3),
  )