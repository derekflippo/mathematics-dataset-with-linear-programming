"""Quadratic programming questions."""

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

_ENTROPY_TRAIN = (1, 4)
_ENTROPY_INTERPOLATE = (5, 6)
_ENTROPY_EXTRAPOLATE = (7, 8)


def _make_modules(entropy):
  return {
      'quadratic_programming': functools.partial(
          quadratic_programming, *entropy
      ),
  }


def train(entropy_fn):
  return _make_modules(entropy_fn(_ENTROPY_TRAIN))


def test():
  return _make_modules(_ENTROPY_INTERPOLATE)


def test_extra():
  return _make_modules(_ENTROPY_EXTRAPOLATE)


def quadratic_programming(min_entropy, max_entropy):
  entropy = random.uniform(min_entropy, max_entropy)
  context = composition.Context()

  n = int(number.integer(entropy / 2, signed=False, min_abs=2))
  m = int(number.integer(entropy / 2, signed=False, min_abs=1))
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
  prob.solve(solver=cp.SCS, verbose=False)

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
      'What is the optimal value?',
      'Find the minimum value of the quadratic program: minimize (1/2) y^T Q y + r^T y\n'
      'where Q = {P} and r = {q},\nsubject to C y <= d and E y = f, where\n'
      'C = {G}, d = {h},\nE = {A}, f = {b}.\n'
      'What is the optimal objective value?',
      'Consider the optimization problem: minimize (1/2) w^T H w + g^T w\n'
      'with H = {P} and g = {q},\nunder the constraints F w <= e and D w = c, where\n'
      'F = {G}, e = {h},\nD = {A}, c = {b}.\n'
      'What is the minimum value achieved?',
  ])

  return example.Problem(
      question=example.question(
          context, template, P=P_str, q=q_str, G=G_str, h=h_str, A=A_str, b=b_str),
      answer=answer)
