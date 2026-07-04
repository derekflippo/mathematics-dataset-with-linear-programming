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

# (n_variables, m_constraints) per level.
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
      'non_trivial_linear_programming': functools.partial(non_trivial_linear_programming, level),
  }


def train(level):
  return _make_modules(level)


def non_trivial_linear_programming(level):
  context = composition.Context()
  n, m = _LEVEL_DIMS[level]
  # Use integer inputs so displayed values are exact (no precision loss)
  coeff_low, coeff_high = -4, 4
  # Resample until optimal value (c^T @ x0) is nonzero
  optimal_value = 0
  while optimal_value == 0:
    s0 = np.random.randint(coeff_low, coeff_high + 1, size=(m,)).astype(float)
    while np.all(s0 >= 0):
      s0 = np.random.randint(coeff_low, coeff_high + 1, size=(m,)).astype(float)
    lamb0 = np.maximum(-s0, 0)
    s0 = np.maximum(s0, 0)
    x0 = np.random.randint(1, coeff_high + 1, size=(n,)).astype(float)
    A = np.random.randint(coeff_low, coeff_high + 1, size=(m, n)).astype(float)
    b = A @ x0 + s0
    c = -A.T @ lamb0
    optimal_value = c @ x0
  
  
  #c is the weights
  #A is the constraints matrix

  # Define and solve the CVXPY problem.
  x = cp.Variable(n)

  #guarantee feasibility and boundedness with x >= 0
  prob = cp.Problem(cp.Minimize(c.T@x),
                  [A @ x <= b, x >= 0])
  prob.solve(solver=cp.CLARABEL, verbose=False)

  # Format arrays for readable output
  A_str = np.array2string(A, precision=2, suppress_small=True, threshold=np.inf)
  b_str = np.array2string(b, precision=2, suppress_small=True, threshold=np.inf)
  c_str = np.array2string(c, precision=2, suppress_small=True, threshold=np.inf)
  answer = round(prob.value)

  template = random.choice([
      'Minimize the objective function c^T * x where \nc = {c}, subject to the constraints \nA * x <= b, where \nA = {A} and \nb = {b}. What is the optimal value?\n'
      'You must solve it using only mental mathematical reasoning. '
      'Do NOT write or execute any code. Do NOT use Python, MATLAB, Julia, or any programming language. '
      'Do NOT use CVXPY, scipy, numpy, or any solver library.',
  ])
  return example.Problem(
      question=example.question(context, template, c=c_str, b=b_str, A=A_str),
      answer=answer)