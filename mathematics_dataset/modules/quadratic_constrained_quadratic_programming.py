"""Quadratically constrained quadratic programming questions.

This module generates small convex QCQP instances and labels them
by solving with CVXPY.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random
import numpy as np

from mathematics_dataset import example

import cvxpy as cp


# (n_variables, m_quadratic_constraints) per level.
_LEVEL_DIMS = [
    (2,  1),  # level 1
    (2,  2),  # level 2
    (3,  2),  # level 3
    (3,  3),  # level 4
    (4,  3),  # level 5
    (5,  4),  # level 6
    (7,  5),  # level 7
    (10, 7),  # level 8
]



def _make_modules(level):
  return {
      'basic_qcqp': functools.partial(basic_qcqp, level),
  }


def train(level):
  return _make_modules(level)


def _rand_int_matrix(n, low, high):
  return np.random.randint(low, high + 1, size=(n, n)).astype(float)


def _rand_int_vector(n, low, high):
  return np.random.randint(low, high + 1, size=n).astype(float)


def _symmetrize(B):
  return 0.5 * (B + B.T)


def _make_psd_matrix(n, low=-2, high=2, lam=1.0):
  """Return a positive semidefinite (usually positive definite) matrix."""
  M = _rand_int_matrix(n, low, high)
  return M.T @ M + lam * np.eye(n)


def _format_matrix(M):
  out = []
  for row in M:
    formatted_row = []
    for v in row:
      if abs(v - round(v)) < 1e-9:
        formatted_row.append(int(round(v)))
      else:
        formatted_row.append(round(float(v), 3))
    out.append(formatted_row)
  return str(out)


def _format_vector(v):
  out = []
  for x in v:
    if abs(x - round(x)) < 1e-9:
      out.append(int(round(x)))
    else:
      out.append(round(float(x), 3))
  return str(out)


def _safe_round(x, ndigits=3):
  if x is None:
    return None
  return round(float(x), ndigits)


# note: qcqp sometimes not convex, we only generate convex problems
def basic_qcqp(level):
  # 1) Choose dimension and number of constraints from level
  n, m = _LEVEL_DIMS[level]

  coeff_low, coeff_high = -4, 4

  # 2) Choose a known feasible point x_star
  x_star = _rand_int_vector(n, -2, 2)

  # 3) Build convex quadratic objective
  Q0 = _make_psd_matrix(n, low=-2, high=2, lam=1.0)
  c0 = _rand_int_vector(n, coeff_low, coeff_high)

  # 4) Build convex quadratic constraints
  Q_list = []
  c_list = []
  d_list = []

  for _ in range(m):
    Qi = _make_psd_matrix(n, low=-2, high=2, lam=0.5)
    ci = _rand_int_vector(n, coeff_low, coeff_high)

    # make x_star feasible with nonnegative slack
    slack = random.randint(1, 4)
    di = float(x_star.T @ Qi @ x_star + ci.T @ x_star + slack)

    Q_list.append(Qi)
    c_list.append(ci)
    d_list.append(di)

  # 5) Add an explicit norm bound to avoid unboundedness / instability
  R = max(3.0, float(np.linalg.norm(x_star) + 3.0))

  # 6) Build and solve the CVXPY problem
  x = cp.Variable(n)

  constraints = [cp.sum_squares(x) <= R**2]
  for i in range(m):
    constraints.append(cp.quad_form(x, Q_list[i]) + c_list[i] @ x <= d_list[i])

  objective = cp.Minimize(0.5 * cp.quad_form(x, Q0) + c0 @ x)
  prob = cp.Problem(objective, constraints)

  prob.solve(solver=cp.CLARABEL, verbose=False)

  retries = 3
  while prob.status not in ["optimal", "optimal_inaccurate"] and retries > 0:
    # resample everything except dimension
    Q0 = _make_psd_matrix(n, low=-2, high=2, lam=1.0)
    c0 = _rand_int_vector(n, coeff_low, coeff_high)

    Q_list = []
    c_list = []
    d_list = []

    for _ in range(m):
      Qi = _make_psd_matrix(n, low=-2, high=2, lam=0.5)
      ci = _rand_int_vector(n, coeff_low, coeff_high)
      slack = random.randint(1, 4)
      di = float(x_star.T @ Qi @ x_star + ci.T @ x_star + slack)
      Q_list.append(Qi)
      c_list.append(ci)
      d_list.append(di)

    constraints = [cp.sum_squares(x) <= R**2]
    for i in range(m):
      constraints.append(cp.quad_form(x, Q_list[i]) + c_list[i] @ x <= d_list[i])

    objective = cp.Minimize(0.5 * cp.quad_form(x, Q0) + c0 @ x)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    retries -= 1

  answer = prob.value

  # 7) Build question text
  constraint_lines = []
  for i in range(m):
    constraint_lines.append(
        "Q_{0} = {1}, c_{0} = {2}, d_{0} = {3}".format(
            i + 1,
            _format_matrix(Q_list[i]),
            _format_vector(c_list[i]),
            _safe_round(d_list[i], 3)
        )
    )

  constraints_text = "\n".join(constraint_lines)

  template = random.choice([
      "Consider the convex quadratically constrained quadratic program over x in R^{n}:\n"
      "Minimize (1/2) x^T Q_0 x + c_0^T x\n"
      "subject to x^T Q_i x + c_i^T x <= d_i for i=1..{m},\n"
      "and ||x||_2^2 <= {R2}.\n\n"
      "Q_0 = {Q0}\n"
      "c_0 = {c0}\n"
      "{constraints_text}\n\n"
      "What is the minimum value of the objective?",
  ])

  question = example.question(
      template,
      n=n,
      m=m,
      R2=_safe_round(R**2, 3),
      Q0=_format_matrix(Q0),
      c0=_format_vector(c0),
      constraints_text=constraints_text
  )

  return example.Problem(question=question, answer=answer)
