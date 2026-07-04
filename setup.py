# Copyright 2019 DeepMind Technologies Limited.
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

"""Module setuptools script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup

description = """A benchmark for evaluating LLMs on constrained optimization.

This code procedurally generates constrained mathematical optimization
problems across five problem types (geometric, linear, quadratic, quadratically
constrained quadratic, and semidefinite programming) and five difficulty
levels, each with a verified numerical answer. It also provides tooling to
evaluate large language models on these problems and to judge their reasoning.

Built on the problem generation framework of "Analysing Mathematical Reasoning
Abilities of Neural Models" (Saxton, Grefenstette, Hill, Kohli)
(https://openreview.net/pdf?id=H1gR5iR5FX).
"""

setup(
    name='mathematics_dataset',
    version='2.0.0',
    description='A benchmark for evaluating LLMs on constrained optimization',
    long_description=description,
    license='Apache License, Version 2.0',
    keywords='constrained optimization benchmark llm evaluation',
    packages=find_packages(),
    install_requires=[
        'absl-py>=0.1.0',
        'sympy>=1.2',
        'six',
        'openai>=1.0',
        'anthropic>=0.30',
        'google-genai>=0.3',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
