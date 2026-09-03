# Agentic Eval Benchmark Harness

A pure Python production-grade multi-dimensional evaluation and benchmarking framework for autonomous AI agents implementing:
- Multi-metric scenario evaluation:
  - **Exact Match & Normalized String Matching:** Case/whitespace normalized verification.
  - **ROUGE-L F1 & BLEU Scoring:** Precision, recall, and overlap n-gram metrics for open-ended generative responses.
  - **Jaccard Similarity:** Token-set intersection over union.
  - **JSON Schema Validation:** Evaluates structural conformance and key existence for structured outputs.
  - **Tool-Calling Precision, Recall & F1:** Validates correct tool selection, required arguments, and payload schema integrity.
  - **Safety & Guardrail Compliance:** Detects prompt leakage, system instruction overrides, and forbidden adversarial tokens.
- Scenario categorization (`REASONING`, `TOOL_USE`, `SAFETY`, `CODE_GENERATION`) across three difficulty tiers (`EASY`, `MEDIUM`, `HARD`).
- Latency and token economic cost tracking with 95% confidence interval estimation.
- Agent leaderboard generation in Table, Markdown, and JSON formats.
- Batch CSV evaluation and telemetry auditing across multi-agent pipelines.

Requires Python standard library only (zero external runtime dependencies).

---

## Features

- **Standard Benchmark Suites:** Pre-configured suites including `reasoning-v1`, `tool-use-v1`, `safety-v1`, and the comprehensive `agentic-core-v1`.
- **Pluggable Agent Adapters:** Out-of-the-box support for `RuleBasedAgent`, `DeterministicMockAgent`, and failure-mode testing with `FaultyAgent`.
- **Side-by-Side Agent Comparison:** Compares candidate models across identical benchmark distributions with detailed category breakdowns.
- **Batch CSV Processing:** High-throughput validation and telemetry auditing for evaluation runs.

---

## Installation & Requirements

- Python 3.10+ (tested on 3.10, 3.11, 3.12)
- Zero external runtime dependencies. `pytest` is optional for running tests.

```bash
git clone https://github.com/abusuraihsakhri/agentic-eval-benchmark-harness.git
cd agentic-eval-benchmark-harness
```

---

## CLI Usage

### 1. List Available Benchmark Suites
```bash
python cli.py list-suites
```

### 2. Inspect Scenarios in a Suite
```bash
python cli.py list-scenarios --suite agentic-core-v1
```

### 3. Run Benchmark Suite on an Agent
```bash
python cli.py run --suite reasoning-v1 --agent mock --format table
```
Or export directly to JSON:
```bash
python cli.py run --suite reasoning-v1 --agent mock --format json
```

### 4. Compare Agents Side-by-Side
```bash
python cli.py compare --suite agentic-core-v1 --agents mock rule-based faulty-injection --format markdown
```

### 5. Evaluate a Single Prompt-Response Pair
```bash
python cli.py eval-single --prompt "Solve 2+2" --response "4" --expected "4" --json
```

### 6. Single Task Telemetry Audit
```bash
python cli.py audit --task-id TASK-2026-001 --primary 29.4 --secondary 15.1 --json
```

### 7. Batch CSV Processing
Batch process records and generate evaluation triage reports:
```bash
python cli.py batch -i sample.csv -o results.csv
```

---

## Python API Quickstart

```python
from agent_eval_harness import (
    BenchmarkRunner,
    DeterministicMockAgent,
    get_standard_benchmark_suites,
    ReportGenerator,
)

# 1. Load benchmark suite
suites = get_standard_benchmark_suites()
suite = suites["reasoning-v1"]

# 2. Setup candidate agent
agent = DeterministicMockAgent("candidate-agent")
agent.add_response("What is 42 * 15?", "The answer is 630.")
agent.add_response("Alice is taller than Bob", "Alice is the tallest.")
agent.add_response("Solve for x: 3x + 15 = 45", "x is 10.")

# 3. Execute benchmark
runner = BenchmarkRunner()
results = runner.run_suite(suite, {"candidate": agent})

# 4. Generate report
leaderboard = ReportGenerator.render_leaderboard(results)
print(leaderboard)
```

---

## Running Tests

Run the test suite using standard `unittest` or `pytest`:

```bash
pytest -v
```

