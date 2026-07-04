"""Quadratic programming questions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random
import cvxpy as cp
import numpy as np

from mathematics_dataset import example

# (n_variables, m_inequality_constraints) per level.
_LEVEL_DIMS = [
    (2,  2),   # level 1
    (2,  3),   # level 2
    (3,  3),   # level 3
    (3,  4),   # level 4
    (4,  5),   # level 5
    (5,  6),   # level 6
    (7,  8),   # level 7
    (10, 12),  # level 8
]



def _make_modules(level):
  return {
      'quadratic_programming': functools.partial(
          quadratic_programming, level
      ),
  }


def train(level):
  return _make_modules(level)


def quadratic_programming(level):
  n, m = _LEVEL_DIMS[level]
  p = max(1, n // 2)

  coeff_low, coeff_high = -4, 4

  # Pick a feasible point x0 (non-negative integers)
  x0 = np.random.randint(1, coeff_high + 1, size=(n,)).astype(float)

  # Build PSD matrix P = M^T M from integer M
  M = np.random.randint(coeff_low, coeff_high + 1, size=(n, n)).astype(float)
  P = M.T @ M

  q = np.random.randint(coeff_low, coeff_high + 1, size=(n,)).astype(float)

  # Inequality constraints: G x <= h with integer slack
  G = np.random.randint(coeff_low, coeff_high + 1, size=(m, n)).astype(float)
  slack = np.random.randint(1, coeff_high + 1, size=(m,)).astype(float)
  h = G @ x0 + slack

  # Equality constraints: A x = b satisfied at x0 by construction
  A = np.random.randint(coeff_low, coeff_high + 1, size=(p, n)).astype(float)
  b = A @ x0

  # Define and solve the CVXPY problem
  x = cp.Variable(n)
  prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                   [G @ x <= h,
                    A @ x == b])
  prob.solve(solver=cp.CLARABEL, verbose=False)

  retries = 3
  while prob.status not in ["optimal", "optimal_inaccurate"] and retries > 0:
    q = np.random.randint(coeff_low, coeff_high + 1, size=(n,)).astype(float)
    G = np.random.randint(coeff_low, coeff_high + 1, size=(m, n)).astype(float)
    slack = np.random.randint(1, coeff_high + 1, size=(m,)).astype(float)
    h = G @ x0 + slack
    A = np.random.randint(coeff_low, coeff_high + 1, size=(p, n)).astype(float)
    b = A @ x0
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                     [G @ x <= h,
                      A @ x == b])
    prob.solve(solver=cp.CLARABEL, verbose=False)
    retries -= 1

  # Format arrays for readable output
  P_str = np.array2string(P, threshold=np.inf)
  q_str = np.array2string(q, threshold=np.inf)
  G_str = np.array2string(G, threshold=np.inf)
  h_str = np.array2string(h, threshold=np.inf)
  A_str = np.array2string(A, threshold=np.inf)
  b_str = np.array2string(b, threshold=np.inf)
  # answer = round(prob.value, 2)
  answer = prob.value
  template = random.choice([
      'Minimize the quadratic objective (1/2) x^T P x + q^T x where\n'
      'P = {P} and q = {q},\nsubject to G x <= h and A x = b, where\n'
      'G = {G}, h = {h},\nA = {A}, b = {b}.\n'
      'What is the optimal value?\n'
      'You must solve it using only mental mathematical reasoning. '
      'Do NOT write or execute any code. Do NOT use Python, MATLAB, Julia, or any programming language. '
      'Do NOT use CVXPY, scipy, numpy, or any solver library.',
  ])

  return example.Problem(
      question=example.question(
          template, P=P_str, q=q_str, G=G_str, h=h_str, A=A_str, b=b_str),
      answer=answer)
