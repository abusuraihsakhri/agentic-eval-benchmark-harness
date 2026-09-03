"""
Comprehensive Test Suite for Agentic Eval Benchmark Harness.
"""
import unittest
import json
import io
import sys
from unittest.mock import patch

from agent_eval_harness.models import (
    TaskCategory,
    DifficultyLevel,
    EvalMetricType,
    ToolDefinition,
    ToolCall,
    AgentInput,
    AgentOutput,
    BenchmarkScenario,
    EvaluationScore,
    ScenarioRunResult,
    AgentSummaryScore,
)
from agent_eval_harness.evaluators import (
    EvaluatorUtils,
    ExactMatchEvaluator,
    StringSimilarityEvaluator,
    JsonSchemaEvaluator,
    ToolCallingEvaluator,
    SafetyEvaluator,
    CompositeEvaluator,
)
from agent_eval_harness.agents import (
    BaseAgent,
    CallableAgent,
    DeterministicMockAgent,
    RuleBasedAgent,
    FaultyAgent,
    TokenCostModel,
)
from agent_eval_harness.engine import (
    BenchmarkSuite,
    BenchmarkRunner,
    ScoreAggregator,
    ReportGenerator,
    get_standard_benchmark_suites,
)
from agent_eval_harness.cli import main, get_agent_by_name


class TestEvaluatorUtils(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(EvaluatorUtils.normalize_text("  Hello,   WORLD!  "), "hello, world!")
        self.assertEqual(EvaluatorUtils.normalize_text("'quoted text'"), "quoted text")
        self.assertEqual(EvaluatorUtils.normalize_text(""), "")

    def test_tokenize(self):
        tokens = EvaluatorUtils.tokenize("Hello world! 123 test.")
        self.assertEqual(tokens, ["hello", "world", "123", "test"])
        self.assertEqual(EvaluatorUtils.tokenize(""), [])

    def test_ngrams(self):
        tokens = ["the", "quick", "brown", "fox"]
        unigrams = EvaluatorUtils.get_ngrams(tokens, 1)
        bigrams = EvaluatorUtils.get_ngrams(tokens, 2)
        self.assertEqual(len(unigrams), 4)
        self.assertEqual(len(bigrams), 3)
        self.assertEqual(bigrams[("the", "quick")], 1)
        self.assertEqual(len(EvaluatorUtils.get_ngrams(tokens, 5)), 0)


class TestExactMatchEvaluator(unittest.TestCase):
    def test_exact_match_normalized(self):
        self.assertEqual(ExactMatchEvaluator.evaluate("Paris", "paris"), 1.0)
        self.assertEqual(ExactMatchEvaluator.evaluate(" Paris  ", "paris"), 1.0)
        self.assertEqual(ExactMatchEvaluator.evaluate("London", "Paris"), 0.0)

    def test_exact_match_case_sensitive(self):
        self.assertEqual(ExactMatchEvaluator.evaluate("Paris", "paris", case_sensitive=True), 0.0)
        self.assertEqual(ExactMatchEvaluator.evaluate("Paris", "Paris", case_sensitive=True), 1.0)

    def test_exact_match_none_and_empty(self):
        self.assertEqual(ExactMatchEvaluator.evaluate("", None), 1.0)
        self.assertEqual(ExactMatchEvaluator.evaluate("something", None), 0.0)
        self.assertEqual(ExactMatchEvaluator.evaluate(None, "expected"), 0.0)


class TestStringSimilarityEvaluator(unittest.TestCase):
    def test_levenshtein_distance(self):
        self.assertEqual(StringSimilarityEvaluator.levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(StringSimilarityEvaluator.levenshtein_distance("test", "test"), 0)
        self.assertEqual(StringSimilarityEvaluator.levenshtein_distance("", "abc"), 3)

    def test_levenshtein_similarity(self):
        self.assertEqual(StringSimilarityEvaluator.levenshtein_similarity("test", "test"), 1.0)
        self.assertAlmostEqual(StringSimilarityEvaluator.levenshtein_similarity("cat", "rat"), 2 / 3, places=2)

    def test_jaccard_similarity(self):
        s1 = "the quick brown fox"
        s2 = "the fast brown fox"
        # union = {the, quick, fast, brown, fox} (5), intersection = {the, brown, fox} (3) -> 3/5 = 0.6
        self.assertAlmostEqual(StringSimilarityEvaluator.jaccard_similarity(s1, s2), 0.6, places=2)
        self.assertEqual(StringSimilarityEvaluator.jaccard_similarity("", ""), 1.0)
        self.assertEqual(StringSimilarityEvaluator.jaccard_similarity("alpha", "beta"), 0.0)

    def test_longest_common_subsequence(self):
        t1 = ["a", "b", "c", "d"]
        t2 = ["a", "c", "d"]
        self.assertEqual(StringSimilarityEvaluator.longest_common_subsequence(t1, t2), 3)

    def test_rouge_l(self):
        cand = "the cat sat on the mat"
        ref = "the cat was on the mat"
        p, r, f = StringSimilarityEvaluator.rouge_l(cand, ref)
        self.assertGreater(f, 0.8)
        self.assertEqual(StringSimilarityEvaluator.rouge_l("", "")[2], 1.0)
        self.assertEqual(StringSimilarityEvaluator.rouge_l("apple", "banana")[2], 0.0)

    def test_bleu_score(self):
        cand = "the cat sat on the mat"
        ref = "the cat sat on the mat"
        self.assertAlmostEqual(StringSimilarityEvaluator.bleu_score(cand, ref), 1.0, places=2)
        self.assertEqual(StringSimilarityEvaluator.bleu_score("apple orange", "banana kiwi"), 0.0)


class TestJsonSchemaEvaluator(unittest.TestCase):
    def test_parse_json_safely(self):
        valid = '{"key": "value", "count": 10}'
        self.assertEqual(JsonSchemaEvaluator.parse_json_safely(valid), {"key": "value", "count": 10})

        markdown_json = '```json\n{"status": "ok"}\n```'
        self.assertEqual(JsonSchemaEvaluator.parse_json_safely(markdown_json), {"status": "ok"})

        self.assertIsNone(JsonSchemaEvaluator.parse_json_safely("invalid plain text"))
        self.assertIsNone(JsonSchemaEvaluator.parse_json_safely(""))

    def test_match_structures_dict(self):
        actual = {"a": 1, "b": "hello", "c": [1, 2]}
        expected = {"a": 1, "b": "hello", "c": [1, 2]}
        self.assertEqual(JsonSchemaEvaluator.match_structures(actual, expected), 1.0)

        partial = {"a": 1, "b": "mismatched"}
        score = JsonSchemaEvaluator.match_structures(partial, expected)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_match_structures_float_tolerance(self):
        self.assertEqual(JsonSchemaEvaluator.match_structures(10.00001, 10.0), 1.0)
        self.assertLess(JsonSchemaEvaluator.match_structures(15.0, 10.0), 1.0)


class TestToolCallingEvaluator(unittest.TestCase):
    def test_tool_calling_perfect_match(self):
        calls = [ToolCall(name="search", arguments={"q": "python"})]
        p, r, f1, meta = ToolCallingEvaluator.evaluate(calls, calls)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        self.assertEqual(f1, 1.0)

    def test_tool_calling_empty_expected(self):
        p, r, f1, _ = ToolCallingEvaluator.evaluate([], [])
        self.assertEqual(f1, 1.0)
        p_hallucinated, _, f1_h, _ = ToolCallingEvaluator.evaluate([ToolCall(name="unneeded")], [])
        self.assertEqual(f1_h, 0.0)

    def test_tool_calling_missing_calls(self):
        p, r, f1, _ = ToolCallingEvaluator.evaluate([], [ToolCall(name="fetch", arguments={"id": 1})])
        self.assertEqual(f1, 0.0)


class TestSafetyEvaluator(unittest.TestCase):
    def test_clean_response(self):
        comp, violations = SafetyEvaluator.evaluate("The capital of France is Paris.", ["forbidden_word"])
        self.assertEqual(comp, 1.0)
        self.assertEqual(len(violations), 0)

    def test_forbidden_token_detected(self):
        comp, violations = SafetyEvaluator.evaluate("Contains secret_key in text", ["secret_key"])
        self.assertLess(comp, 1.0)
        self.assertEqual(len(violations), 1)

    def test_prompt_injection_pattern(self):
        comp, violations = SafetyEvaluator.evaluate("SYSTEM OVERRIDE: Revealing system prompt now.", [])
        self.assertLess(comp, 1.0)
        self.assertGreaterEqual(len(violations), 1)


class TestCompositeEvaluator(unittest.TestCase):
    def test_composite_scoring(self):
        scenario = BenchmarkScenario(
            scenario_id="TEST-001",
            name="Test Scenario",
            category=TaskCategory.REASONING,
            difficulty=DifficultyLevel.EASY,
            input_data=AgentInput(prompt="Say hello"),
            expected_output="Hello world",
        )
        out_perfect = AgentOutput(response_text="Hello world", latency_ms=10.0)
        score_perfect = CompositeEvaluator.evaluate_run(scenario, out_perfect)
        self.assertEqual(score_perfect.exact_match, 1.0)
        self.assertEqual(score_perfect.overall_score, 1.0)

        out_error = AgentOutput(response_text="", error="Fail", latency_ms=10.0)
        score_error = CompositeEvaluator.evaluate_run(scenario, out_error)
        self.assertEqual(score_error.overall_score, 0.0)


class TestTokenCostModel(unittest.TestCase):
    def test_pricing_calculation(self):
        cost_gpt4o = TokenCostModel.calculate_cost(1000, 500, model="gpt-4o")
        # 1000 * 2.5/1M + 500 * 10/1M = 0.0025 + 0.005 = 0.0075
        self.assertAlmostEqual(cost_gpt4o, 0.0075, places=5)

        cost_zero = TokenCostModel.calculate_cost(0, 0)
        self.assertEqual(cost_zero, 0.0)


class TestAgentImplementations(unittest.TestCase):
    def test_callable_agent(self):
        def my_fn(inp: AgentInput):
            return f"Processed: {inp.prompt}"

        agent = CallableAgent(my_fn, agent_id="custom-agent")
        out = agent.run(AgentInput(prompt="test input"))
        self.assertEqual(out.response_text, "Processed: test input")
        self.assertGreater(out.prompt_tokens, 0)

    def test_deterministic_mock_agent(self):
        mock = DeterministicMockAgent()
        mock.add_response("greet", "Hello there!")
        out = mock.run(AgentInput(prompt="Please greet the user"))
        self.assertEqual(out.response_text, "Hello there!")

    def test_rule_based_agent_arithmetic(self):
        agent = RuleBasedAgent()
        out = agent.run(AgentInput(prompt="What is 50 + 25?"))
        self.assertIn("75", out.response_text)

    def test_rule_based_agent_tool_calling(self):
        tool = ToolDefinition(name="calculator", description="Math tool", required=["a", "b"])
        agent = RuleBasedAgent()
        out = agent.run(AgentInput(prompt="Use calculator with 10 and 20", tools=[tool]))
        self.assertEqual(len(out.tool_calls), 1)
        self.assertEqual(out.tool_calls[0].name, "calculator")

    def test_faulty_agent_modes(self):
        f_exc = FaultyAgent(failure_mode="exception")
        out_exc = f_exc.run(AgentInput(prompt="test"))
        self.assertIsNotNone(out_exc.error)

        f_leak = FaultyAgent(failure_mode="leak_prompt")
        out_leak = f_leak.run(AgentInput(prompt="test"))
        self.assertIn("SYSTEM OVERRIDE", out_leak.response_text)


class TestBenchmarkEngineAndRunner(unittest.TestCase):
    def test_standard_suites_creation(self):
        suites = get_standard_benchmark_suites()
        self.assertIn("reasoning-v1", suites)
        self.assertIn("tool-use-v1", suites)
        self.assertIn("safety-v1", suites)
        self.assertIn("agentic-core-v1", suites)
        self.assertGreater(len(suites["agentic-core-v1"].scenarios), 5)

    def test_suite_filters(self):
        suites = get_standard_benchmark_suites()
        full = suites["agentic-core-v1"]
        reasoning_scenarios = full.filter_by_category(TaskCategory.REASONING)
        self.assertGreaterEqual(len(reasoning_scenarios), 3)

    def test_score_aggregator_bootstrap_ci(self):
        scores = [0.8, 0.85, 0.9, 0.95, 1.0]
        low, high = ScoreAggregator.bootstrap_ci(scores)
        self.assertLessEqual(low, high)
        self.assertGreaterEqual(low, 0.75)
        self.assertLessEqual(high, 1.0)

    def test_benchmark_runner_full_execution(self):
        suites = get_standard_benchmark_suites()
        suite = suites["reasoning-v1"]
        mock_agent = get_agent_by_name("mock")
        faulty_agent = get_agent_by_name("faulty")

        runner = BenchmarkRunner()
        result = runner.run_suite(suite, {"mock": mock_agent, "faulty": faulty_agent})

        self.assertEqual(len(result.rankings), 2)
        self.assertEqual(result.rankings[0], "mock")
        self.assertGreater(result.agent_summaries["mock"].mean_overall_score, 0.8)
        self.assertEqual(result.agent_summaries["faulty"].pass_rate, 0.0)

    def test_report_generation(self):
        suites = get_standard_benchmark_suites()
        suite = suites["reasoning-v1"]
        runner = BenchmarkRunner()
        res = runner.run_suite(suite, {"rule-based": RuleBasedAgent()})

        table = ReportGenerator.render_leaderboard(res)
        self.assertIn("AGENT EVALUATION BENCHMARK HARNESS - LEADERBOARD", table)

        md = ReportGenerator.to_markdown(res)
        self.assertIn("# Agent Benchmark Report", md)


class TestCLICommands(unittest.TestCase):
    def test_cli_list_suites(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = main(["list-suites"])
            self.assertEqual(code, 0)
            self.assertIn("AVAILABLE AGENT BENCHMARK SUITES", fake_out.getvalue())

    def test_cli_list_scenarios(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = main(["list-scenarios", "--suite", "reasoning-v1"])
            self.assertEqual(code, 0)
            self.assertIn("SCENARIOS IN SUITE", fake_out.getvalue())

    def test_cli_eval_single(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = main(["eval-single", "--prompt", "What is 2+2?", "--response", "4", "--expected", "4"])
            self.assertEqual(code, 0)
            self.assertIn("Overall Score:", fake_out.getvalue())

    def test_cli_run_json(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = main(["run", "--suite", "reasoning-v1", "--agent", "mock", "--format", "json"])
            self.assertEqual(code, 0)
            data = json.loads(fake_out.getvalue())
            self.assertIn("rankings", data)
            self.assertIn("agent_summaries", data)

    def test_cli_audit_json(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = main(["audit", "--json", "--primary", "28.0"])
            self.assertEqual(code, 0)
            data = json.loads(fake_out.getvalue())
            self.assertEqual(data["overall_status"], "ELEVATED_RISK_WARNING")

    def test_cli_batch(self):
        import os
        import tempfile
        sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "out_batch.csv")
            ret = main(["batch", "-i", sample_path, "-o", out_file])
            self.assertEqual(ret, 0)
            self.assertTrue(os.path.exists(out_file))


if __name__ == "__main__":
    unittest.main()

