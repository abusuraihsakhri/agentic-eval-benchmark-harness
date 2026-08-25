# Agentic Eval Benchmark Harness

A production-grade, multi-dimensional evaluation and benchmarking harness for autonomous LLM agents. Designed for comprehensive testing across multi-turn reasoning, tool/function calling, instruction following, structured output schemas, safety guardrails, latency, and token economics.

---

## Key Features

- **Multi-Dimensional Metrics Engine**:
  - **Exact Match**: Case/whitespace normalized string comparisons.
  - **Lexical & Semantic Overlap**: Longest Common Subsequence (ROUGE-L precision/recall/F-measure), BLEU-1/BLEU-2 with brevity penalty, Jaccard token set similarity, Levenshtein edit distance.
  - **JSON Schema & Structural Validator**: Recursive structural validation with type matching, missing key penalty, and numeric floating-point tolerance.
  - **Tool & Function Calling Evaluator**: Evaluates tool selection accuracy, argument precision/recall/F1, and dispatch sequence fidelity.
  - **Safety & Guardrail Compliance**: Automatic detection of prompt injection patterns (`SYSTEM OVERRIDE`, jailbreak instructions) and forbidden token/PII leakage.
  - **Token & Cost Accounting**: Standard token pricing models ($/1M input & output tokens for GPT-4o, Claude 3.5 Sonnet, Llama-3-70B, etc.), p50/p95 latency profiling.
- **Statistical Aggregation**:
  - Bootstrap 95% confidence intervals (empirical resampling) for overall score reliability.
  - Granular breakdown across task categories and difficulty tiers (Easy, Medium, Hard, Expert).
- **Curated Standard Suites**:
  - `reasoning-v1`: Multi-step logic, math, and analytical deduction.
  - `tool-use-v1`: Function calling and API invocation benchmarks.
  - `safety-v1`: Adversarial robustness and sensitive information guardrails.
  - `agentic-core-v1`: Comprehensive full-spectrum agent evaluation suite.
- **Pluggable Agent Adapters**:
  - Standard abstract `BaseAgent` and `CallableAgent` wrapper for arbitrary agent callbacks.
  - Built-in `DeterministicMockAgent`, `RuleBasedAgent`, and `FaultyAgent` for testing and calibration.
- **Reporting & Leaderboards**:
  - Interactive CLI, terminal leaderboard tables, structured JSON dossiers, and GitHub-ready Markdown reports.

---

## Metric Formulas & Mathematical Formulation

### 1. ROUGE-L (Longest Common Subsequence)
Given candidate token sequence $C$ of length $m$ and reference token sequence $R$ of length $n$:
$$\text{LCS}(C, R) = \text{length of longest common subsequence}$$
$$P_{\text{LCS}} = \frac{\text{LCS}(C, R)}{m}, \quad R_{\text{LCS}} = \frac{\text{LCS}(C, R)}{n}$$
$$F_{\text{LCS}} = \frac{(1 + \beta^2) P_{\text{LCS}} R_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 P_{\text{LCS}}} \quad (\beta = 1.2)$$

### 2. BLEU Score with Brevity Penalty
$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \le r \end{cases}$$
$$\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^N w_n \ln p_n \right)$$

### 3. Tool Calling F1 Score
Given expected tool invocations $E$ and actual invocations $A$:
$$\text{Precision} = \frac{\sum_{e \in E} \max_{a \in A} \text{Match}(e, a)}{|A|}, \quad \text{Recall} = \frac{\sum_{e \in E} \max_{a \in A} \text{Match}(e, a)}{|E|}$$
$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 4. Bootstrap 95% Confidence Interval
For $N$ scenario evaluation scores $S = [s_1, s_2, \dots, s_N]$, generate $B = 500$ bootstrap resamples $S_b^*$, compute sample means $\bar{s}_b^*$, sort them, and select percentiles:
$$\text{CI}_{95\%} = \left[ \bar{s}^*_{(2.5\%)}, \, \bar{s}^*_{(97.5\%)} \right]$$

---

## CLI Usage

### 1. List Available Benchmark Suites
```bash
python cli.py list-suites
```

### 2. List Scenarios in a Suite
```bash
python cli.py list-scenarios --suite agentic-core-v1
```

### 3. Run Benchmark Suite on an Agent
```bash
python cli.py run --suite agentic-core-v1 --agent mock --format table
python cli.py run --suite reasoning-v1 --agent rule-based --format markdown
python cli.py run --suite tool-use-v1 --agent mock --format json --output results.json
```

### 4. Compare Multiple Agents Side-by-Side
```bash
python cli.py compare --suite agentic-core-v1 --agents mock rule-based faulty-injection
```

### 5. Evaluate Single Prompt/Response Pair
```bash
python cli.py eval-single \
  --prompt "What is the capital of France?" \
  --response "The capital of France is Paris." \
  --expected "Paris"
```

### 6. Interactive Mode
```bash
python cli.py interactive
```

---

## Python API Example

```python
from agent_eval_harness import (
    BenchmarkRunner,
    get_standard_benchmark_suites,
    CallableAgent,
    AgentInput,
    AgentOutput,
    ReportGenerator,
)

# 1. Define custom agent function
def my_custom_agent(inp: AgentInput) -> AgentOutput:
    if "42 * 15" in inp.prompt:
        return AgentOutput(response_text="The answer is 630.")
    return AgentOutput(response_text=f"Processed: {inp.prompt}")

# 2. Wrap as agent
agent = CallableAgent(my_custom_agent, agent_id="my-custom-llm-agent")

# 3. Load benchmark suite and execute
suites = get_standard_benchmark_suites()
suite = suites["reasoning-v1"]

runner = BenchmarkRunner()
result = runner.run_suite(suite, {"my-agent": agent})

# 4. Print formatted leaderboard
print(ReportGenerator.render_leaderboard(result))
```

---

## Running the Unit Test Suite

```bash
python -m unittest test_agent_eval_harness.py
```

All 35+ unit tests verify:
- Exact match, ROUGE-L, BLEU, Jaccard, and Levenshtein metrics.
- JSON structure and schema tolerance validations.
- Function/tool calling precision, recall, and argument matching.
- Safety policies, prompt injection detection, and forbidden token filtering.
- Latency profiling, token cost accounting, and bootstrap confidence intervals.
- CLI subcommands, JSON serialization, and Markdown report rendering.

---

## License

MIT License. Developed by Dr. Abu Suraih Sakhri.
