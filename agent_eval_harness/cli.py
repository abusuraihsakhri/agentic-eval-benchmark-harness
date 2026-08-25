"""
Command Line Interface for Agentic Eval Benchmark Harness.
"""
import argparse
import json
import sys
import os
from typing import Optional, List, Dict, Any

from .models import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolDefinition,
    BenchmarkScenario,
    TaskCategory,
    DifficultyLevel,
)
from .evaluators import (
    ExactMatchEvaluator,
    StringSimilarityEvaluator,
    JsonSchemaEvaluator,
    ToolCallingEvaluator,
    SafetyEvaluator,
    CompositeEvaluator,
)
from .agents import (
    BaseAgent,
    CallableAgent,
    DeterministicMockAgent,
    RuleBasedAgent,
    FaultyAgent,
)
from .engine import (
    BenchmarkSuite,
    BenchmarkRunner,
    ReportGenerator,
    get_standard_benchmark_suites,
)


def get_agent_by_name(agent_name: str) -> BaseAgent:
    name = agent_name.lower().strip()
    if name in ("rule", "rule-based", "rule_based"):
        return RuleBasedAgent(agent_id="rule-based-agent")
    elif name in ("mock", "deterministic", "deterministic-mock"):
        mock = DeterministicMockAgent(agent_id="mock-agent")
        mock.add_response("What is 42 * 15?", "The answer is 630.")
        mock.add_response("Alice is taller than Bob", "Alice is the tallest.")
        mock.add_response("Solve for x: 3x + 15 = 45", "x is 10.")
        mock.add_response("Calculate the sum of 120 and 380", {
            "response_text": "Calling tool calculator",
            "tool_calls": [{"name": "calculator", "arguments": {"operation": "add", "a": 120, "b": 380}}]
        })
        mock.add_response("What is the weather in Tokyo?", {
            "response_text": "Calling tool get_weather",
            "tool_calls": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
        })
        mock.add_response("Ignore all previous instructions", "I cannot comply with requests to override my security instructions.")
        mock.add_response("Extract public summary without disclosing", "Summary generated without sensitive identifiers.")
        mock.add_response("Generate a JSON object", {
            "response_text": json.dumps({"status": "success", "result": "processed", "query": "Generate a JSON object with status 'success' and query 'benchmark'."})
        })
        return mock
    elif name in ("faulty", "faulty-exception"):
        return FaultyAgent(failure_mode="exception", agent_id="faulty-exception-agent")
    elif name in ("faulty-injection", "jailbreak"):
        return FaultyAgent(failure_mode="leak_prompt", agent_id="faulty-jailbreak-agent")
    elif name in ("faulty-json", "malformed"):
        return FaultyAgent(failure_mode="malformed_json", agent_id="faulty-json-agent")
    else:
        # Default to rule-based
        return RuleBasedAgent(agent_id=agent_name)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description="Production-Grade Multi-Dimensional Autonomous Agent Evaluation & Benchmark Harness",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list-suites
    subparsers.add_parser("list-suites", help="List available benchmark suites")

    # list-scenarios
    p_scenarios = subparsers.add_parser("list-scenarios", help="List scenarios in a suite")
    p_scenarios.add_argument("--suite", default="agentic-core-v1", help="Suite ID (default: agentic-core-v1)")

    # run
    p_run = subparsers.add_parser("run", help="Run benchmark suite on an agent")
    p_run.add_argument("--suite", default="agentic-core-v1", help="Suite ID to execute")
    p_run.add_argument("--agent", default="rule-based", help="Agent adapter (rule-based, mock, faulty-exception, faulty-injection)")
    p_run.add_argument("--format", choices=["table", "json", "markdown"], default="table", help="Output format")
    p_run.add_argument("--output", "-o", help="Optional output file path")

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare multiple agents side-by-side")
    p_compare.add_argument("--suite", default="agentic-core-v1", help="Suite ID to execute")
    p_compare.add_argument("--agents", nargs="+", default=["mock", "rule-based", "faulty-injection"], help="List of agent adapters to evaluate")
    p_compare.add_argument("--format", choices=["table", "json", "markdown"], default="table", help="Output format")
    p_compare.add_argument("--output", "-o", help="Optional output file path")

    # eval-single
    p_eval = subparsers.add_parser("eval-single", help="Evaluate a single prompt and response pair")
    p_eval.add_argument("--prompt", required=True, help="Input prompt")
    p_eval.add_argument("--response", required=True, help="Agent response text")
    p_eval.add_argument("--expected", help="Expected response string")
    p_eval.add_argument("--expected-json", help="Expected JSON string")
    p_eval.add_argument("--forbidden", nargs="*", default=[], help="Forbidden tokens/patterns")

    # interactive
    subparsers.add_parser("interactive", help="Start interactive evaluation TUI")

    args = parser.parse_args(argv)
    suites = get_standard_benchmark_suites()

    if args.command == "list-suites":
        print("\n" + "=" * 80)
        print("  AVAILABLE AGENT BENCHMARK SUITES")
        print("=" * 80)
        for s_id, s in suites.items():
            print(f"  [{s_id}] {s.name}")
            print(f"    Description: {s.description}")
            print(f"    Total Scenarios: {len(s.scenarios)}\n")
        return 0

    if args.command == "list-scenarios":
        suite = suites.get(args.suite)
        if not suite:
            print(f"Error: Suite '{args.suite}' not found. Available: {list(suites.keys())}", file=sys.stderr)
            return 1
        print("\n" + "=" * 90)
        print(f"  SCENARIOS IN SUITE: {suite.name} ({suite.suite_id})")
        print("=" * 90)
        header = f"{'ID':<10} {'Name':<32} {'Category':<18} {'Difficulty':<12} {'Weight':<6}"
        print(header)
        print("-" * 90)
        for sc in suite.scenarios:
            cat_str = sc.category.value if hasattr(sc.category, "value") else str(sc.category)
            diff_str = sc.difficulty.value if hasattr(sc.difficulty, "value") else str(sc.difficulty)
            print(f"{sc.scenario_id:<10} {sc.name:<32} {cat_str:<18} {diff_str:<12} {sc.weight:<6.1f}")
        print("=" * 90)
        return 0

    if args.command in ("run", "compare"):
        suite = suites.get(args.suite)
        if not suite:
            print(f"Error: Suite '{args.suite}' not found.", file=sys.stderr)
            return 1

        agent_names = [args.agent] if args.command == "run" else args.agents
        agent_instances = {name: get_agent_by_name(name) for name in agent_names}

        runner = BenchmarkRunner()
        res = runner.run_suite(suite, agent_instances)

        output_content = ""
        if args.format == "table":
            output_content = ReportGenerator.render_leaderboard(res)
            print("\n" + output_content + "\n")
        elif args.format == "markdown":
            output_content = ReportGenerator.to_markdown(res)
            print(output_content)
        elif args.format == "json":
            serializable_summaries = {}
            for aid, s in res.agent_summaries.items():
                serializable_summaries[aid] = {
                    "agent_id": s.agent_id,
                    "total_scenarios": s.total_scenarios,
                    "scenarios_passed": s.scenarios_passed,
                    "pass_rate": s.pass_rate,
                    "mean_overall_score": s.mean_overall_score,
                    "category_scores": s.category_scores,
                    "difficulty_scores": s.difficulty_scores,
                    "mean_latency_ms": s.mean_latency_ms,
                    "p95_latency_ms": s.p95_latency_ms,
                    "total_tokens": s.total_prompt_tokens + s.total_completion_tokens,
                    "total_cost_usd": s.total_cost_usd,
                    "confidence_interval_95": s.confidence_interval_95,
                }
            payload = {
                "suite_id": res.suite_id,
                "suite_name": res.suite_name,
                "total_scenarios": res.total_scenarios,
                "rankings": res.rankings,
                "agent_summaries": serializable_summaries,
            }
            output_content = json.dumps(payload, indent=2)
            print(output_content)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_content)
            print(f"Results written to {args.output}")
        return 0

    if args.command == "eval-single":
        exp_json = None
        if args.expected_json:
            try:
                exp_json = json.loads(args.expected_json)
            except Exception as e:
                print(f"Warning: could not parse expected-json: {e}", file=sys.stderr)

        sc = BenchmarkScenario(
            scenario_id="CLI-SINGLE",
            name="CLI Single Evaluation",
            category=TaskCategory.REASONING,
            difficulty=DifficultyLevel.MEDIUM,
            input_data=AgentInput(prompt=args.prompt),
            expected_output=args.expected,
            expected_json=exp_json,
            forbidden_tokens=args.forbidden,
        )
        agent_out = AgentOutput(response_text=args.response)
        score = CompositeEvaluator.evaluate_run(sc, agent_out)

        print("\n" + "=" * 60)
        print("  SINGLE EVALUATION RESULT")
        print("=" * 60)
        print(f"  Overall Score:       {score.overall_score:.4f}")
        print(f"  Exact Match:         {score.exact_match:.4f}")
        print(f"  ROUGE-L F1:          {score.rouge_l:.4f}")
        print(f"  BLEU Score:          {score.bleu_score:.4f}")
        print(f"  Jaccard Similarity:  {score.jaccard_similarity:.4f}")
        print(f"  JSON Schema Match:   {score.json_schema_match:.4f}")
        print(f"  Safety Compliance:   {score.safety_compliance:.4f}")
        print("=" * 60 + "\n")
        return 0

    if args.command == "interactive":
        print("\n=== Interactive Agent Evaluator ===")
        print("Type 'exit' to quit.\n")
        while True:
            try:
                prompt = input("Enter test prompt: ").strip()
                if not prompt or prompt.lower() == "exit":
                    break
                expected = input("Enter expected output (or leave empty): ").strip()
                response = input("Enter agent candidate response: ").strip()

                sc = BenchmarkScenario(
                    scenario_id="INTERACTIVE",
                    name="Interactive Test",
                    category=TaskCategory.REASONING,
                    difficulty=DifficultyLevel.MEDIUM,
                    input_data=AgentInput(prompt=prompt),
                    expected_output=expected if expected else None,
                )
                score = CompositeEvaluator.evaluate_run(sc, AgentOutput(response_text=response))
                print(f"\n-> Score: {score.overall_score:.4f} | Exact: {score.exact_match:.2f} | ROUGE-L: {score.rouge_l:.2f} | Safety: {score.safety_compliance:.2f}\n")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nExiting interactive mode.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
