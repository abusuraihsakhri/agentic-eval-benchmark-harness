"""
Agent Evaluation Benchmark Harness: Production-Grade Multi-Dimensional Agent Benchmarking.
"""

from .models import (
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
    BenchmarkSuiteResult,
)
from .evaluators import (
    EvaluatorUtils,
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
    TokenCostModel,
)
from .engine import (
    BenchmarkSuite,
    BenchmarkRunner,
    ScoreAggregator,
    ReportGenerator,
    get_standard_benchmark_suites,
)

__version__ = "1.0.0"

__all__ = [
    "TaskCategory",
    "DifficultyLevel",
    "EvalMetricType",
    "ToolDefinition",
    "ToolCall",
    "AgentInput",
    "AgentOutput",
    "BenchmarkScenario",
    "EvaluationScore",
    "ScenarioRunResult",
    "AgentSummaryScore",
    "BenchmarkSuiteResult",
    "EvaluatorUtils",
    "ExactMatchEvaluator",
    "StringSimilarityEvaluator",
    "JsonSchemaEvaluator",
    "ToolCallingEvaluator",
    "SafetyEvaluator",
    "CompositeEvaluator",
    "BaseAgent",
    "CallableAgent",
    "DeterministicMockAgent",
    "RuleBasedAgent",
    "FaultyAgent",
    "TokenCostModel",
    "BenchmarkSuite",
    "BenchmarkRunner",
    "ScoreAggregator",
    "ReportGenerator",
    "get_standard_benchmark_suites",
]
