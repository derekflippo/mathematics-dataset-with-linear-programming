# Constrained Optimization Benchmark for LLMs

A benchmark for evaluating LLM performance on constrained mathematical optimization problems. Problems are procedurally generated across five problem types and five difficulty levels, with verified numerical answers.

## Problem Types

| ID | Type | Description |
|----|------|-------------|
| GP | Geometric Programming | Posynomial objective and constraint functions |
| LP | Linear Programming | Linear objective with linear inequality constraints |
| QCQP | Quadratically Constrained Quadratic Programming | Quadratic objective with quadratic constraints |
| QP | Quadratic Programming | Quadratic objective with linear constraints |
| SDP | Semidefinite Programming | Linear objective with positive semidefinite matrix constraints |

## Difficulty Levels

Levels 1–5, with increasing problem size, number of constraints, and complexity of the optimal solution structure.

## Setup

```bash
pip install -e .
```

## Generating Problems

```bash
python -m mathematics_dataset.generate_to_file \
  --output_dir=output_json \
  --num_problems=25
```

## Evaluating Models

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
export DEEPSEEK_API_KEY=sk-...

python -m mathematics_dataset.evaluate \
  --input_dir=output_json \
  --output_dir=eval_results
```

Supported models are configured in `mathematics_dataset/evaluate.py` under `ENGINES`.

## LLM Judge

Run the reasoning and final-answer judge on evaluation results:

```bash
export OPENAI_API_KEY=sk-...

python mathematics_dataset/judge_errors.py \
  --input_json eval_results/level-1/quadratic_programming__quadratic_programming__claude-sonnet-4-6.json \
  --output_json judged.json \
  --judge all
```

## License

Apache 2.0 — derived from the [DeepMind Mathematics Dataset](https://github.com/deepmind/mathematics_dataset).
