"""Semidefinite programming questions.

This module defines a generator that produces *small* SDP instances
and labels them by solving with CVXPY (so we can verify correctness).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random
import numpy as np

from mathematics_dataset import example
from mathematics_dataset.util import composition

# CVXPY: we use this to solve the SDP to verify
import cvxpy as cp



# Entropy ranges control "difficulty" / "size" selection for train/test.
# The entropy_fn provided by the dataset framework chooses a value in
# these ranges; we then map that to k (matrix size) and m (#constraints).
_ENTROPY_TRAIN = (3, 10)
_ENTROPY_INTERPOLATE = (8, 8)
_ENTROPY_EXTRAPOLATE = (12, 12)


# Module registry: dataset framework expects a dict of named generators.
# functools.partial(basic_semidefinite_programming, *entropy) means:
#   later, calling the module runs basic_semidefinite_programming(minE, maxE)
def _make_modules(entropy):
  return {
      'basic_semidefinite_programming': functools.partial(
          basic_semidefinite_programming, *entropy
      ),
  }


# Entry points used by the dataset framework to select train/test ranges
def train(entropy_fn):
  return _make_modules(entropy_fn(_ENTROPY_TRAIN))


def test():
  return _make_modules(_ENTROPY_INTERPOLATE)


def test_extra():
  return _make_modules(_ENTROPY_EXTRAPOLATE)


# Helper functions

# symmetrize: turn arbitrary matrix into symmetric matrix (B+B^T)/2
def _symmetrize(B):
  """Return the symmetric part of B: (B + B^T)/2."""
  return 0.5 * (B + B.T)

# rand_int_matrix: small integer matrices (keeps numbers readable)
def _rand_int_matrix(k, low, high):
  """Random kxk integer matrix in [low, high], returned as float array."""
  return np.random.randint(low, high + 1, size=(k, k)).astype(float)

# format_matrix: make matrices printable inside question strings
def _format_matrix(M):
  """Pretty-format a small matrix for text questions."""
  out = []
  for row in M:
    formatted_row = []
    for v in row:
      # Show as int if it's essentially an integer; else show rounded float.
      if abs(v - round(v)) < 1e-9:
        formatted_row.append(int(round(v)))
      else:
        formatted_row.append(round(float(v), 3))
    out.append(formatted_row)
  return str(out)

# safe_round: stable rounding for numeric solver outputs
def _safe_round(x, ndigits=3):
  """Round numeric values from solvers (which may be numpy floats)."""
  if x is None:
    return None
  return round(float(x), ndigits)

# Problem generator: builds an SDP instance and solves it for the label.

# SDP form generated:
#   minimize   <C, X> = trace(C^T X)
#   subject to <A_i, X> = b_i   for i=1..m
#              X ⪰ 0 (PSD)

# Feasibility trick:
#   - Construct a known PSD matrix X* = M M^T
#   - Set b_i = <A_i, X*>
# Then X* satisfies all constraints by construction.
def basic_semidefinite_programming(min_entropy, max_entropy):
  entropy = random.uniform(min_entropy, max_entropy)
  context = composition.Context()  # provides consistent formatting hooks

  # Complexity selection (controls how big/complex the SDP is).
  # We keep k small so the question is readable and solver stable.
  if entropy < 6:
    k = 2
  else:
    k = 3  # 3x3 is still readable and is a "real" SDP

  # Choose the number of linear constraints m (besides PSD).
  # Note: total constraints are: m equalities + 1 PSD constraint.
  m = 2 if k == 3 else 1
  if entropy > 9:
    m = 3

  # Coefficient magnitude: keep small integers to avoid huge numbers in the statement and reduce numerical instability
  coeff_low, coeff_high = -3, 3

  # 1) Generate symmetric constraint matrices A_1...A_m
  A_list = []
  for _ in range(m):
    B = _rand_int_matrix(k, coeff_low, coeff_high)
    A = _symmetrize(B)
    A_list.append(A)

  # 2) Generate a PSD "witness" matrix X_star = M M^T so that it's always PSD 
  M = _rand_int_matrix(k, -2, 2)
  X_star = M @ M.T  # PSD by construction

  # 3) Compute right-hand sides b_i using the inner product
  b_list = []
  for A in A_list:
    b_i = float(np.trace(A.T @ X_star))
    b_list.append(b_i)

  # 4) Generate symmetric objective matrix C
  # Objective is: minimize <C, X> = trace(C^T X)
  Bc = _rand_int_matrix(k, coeff_low, coeff_high)
  C = _symmetrize(Bc)

  # 5) Build and solve the CVXPY problem for verification/label
  # we add trace(X) == trace(X_star) to reduce unboundedness and scale issues. To maximize robustness, uncomment
  X = cp.Variable((k, k), symmetric=True)

  constraints = [X >> 0]

  # Optional normalization (recommended for stability):
  # constraints.append(cp.trace(X) == float(np.trace(X_star)))

  for i in range(m):
    constraints.append(cp.trace(A_list[i] @ X) == b_list[i])

  objective = cp.Minimize(cp.trace(C @ X))
  prob = cp.Problem(objective, constraints)

  # Solve with SCS (common default for cone problems like SDP).
  prob.solve(solver=cp.SCS, verbose=False)

  # 6) Retry logic: If solver doesn't return optimal, we resample the *objective* C.
  # (If the failure is due to constraints/conditioning, resampling A/X* is better; this is just a lightweight retry.)
  retries = 3
  while prob.status not in ["optimal", "optimal_inaccurate"] and retries > 0:
    Bc = _rand_int_matrix(k, coeff_low, coeff_high)
    C = _symmetrize(Bc)

    objective = cp.Minimize(cp.trace(C @ X))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    retries -= 1

  if prob.status not in ["optimal", "optimal_inaccurate"]:
    raise ValueError("SDP solve failed with status: {}".format(prob.status))

  # 7) Extract the numeric optimal value as the "answer" (rounded)
  answer = _safe_round(prob.value, ndigits=3)

  # 8) Build natural-language question
  A_text = "\n".join(
      ["A_{} = {}".format(i + 1, _format_matrix(A_list[i])) for i in range(m)]
  )
  b_text = ", ".join(
      ["b_{} = {}".format(i + 1, _safe_round(b_list[i], 3)) for i in range(m)]
  )

  template = random.choice([
      "Consider the semidefinite program over a symmetric {k}x{k} matrix X:\n"
      "Minimize <C, X> subject to <A_i, X> = b_i for i=1..{m}, and X is positive semidefinite (X ⪰ 0).\n\n"
      "C = {C}\n"
      "{A_text}\n"
      "{b_text}\n\n"
      "What is the minimum value of <C, X> (rounded to 3 decimals)?"
  ])

  question = example.question(
      context,
      template,
      k=k,
      m=m,
      C=_format_matrix(C),
      A_text=A_text,
      b_text=b_text
  )

  # 9) Return an example.Problem object the dataset framework expects
  # question: formatted string
  # answer: numeric label
  return example.Problem(question=question, answer=answer)
