# Copyright 2018 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The various mathematics modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mathematics_dataset.modules import geometric
from mathematics_dataset.modules import linear_programming
from mathematics_dataset.modules import quadratic_programming
from mathematics_dataset.modules import quadratic_constrained_quadratic_programming
from mathematics_dataset.modules import semidefinite_programming
import six


all_ = {
    'geometric': geometric,
    'linear_programming': linear_programming,
    'quadratic_constrained_quadratic_programming': quadratic_constrained_quadratic_programming,
    'quadratic_programming': quadratic_programming,
    'semidefinite_programming': semidefinite_programming,
}


def train(level):
  """Returns dict of modules generating problems at the given difficulty level.

  Args:
    level: Integer level index in [0, 7].

  Returns a dict mapping each module name to its dict of generator callables.
  """
  return {
      name: module.train(level) for name, module in six.iteritems(all_)
  }


