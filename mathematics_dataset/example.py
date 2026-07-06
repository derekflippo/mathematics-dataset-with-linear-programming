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

"""Containers for "[example] problems" (i.e., question/answer) pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


def question(template, **kwargs):
  """Makes a question by formatting `template` with the given fields.

  Example:

  ```
  question('What is {p} over {q}?', p=3, q=4)
  ```

  Arguments:
    template: A string, like "Calculate the value of {exp}.".
    **kwargs: A dictionary mapping template fields to values.

  Returns:
    String.
  """
  assert isinstance(template, str)
  return template.format(**kwargs)


Problem = collections.namedtuple('Problem', ('question', 'answer'))


