"""
Data Models and Enums for Agent Evaluation Benchmark Harness.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import time


class TaskCategory(str, Enum):
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    CODE_GENERATION = "code_generation"
    INSTRUCTION_FOLLOWING = "instruction_following"
    RETRIEVAL_QA = "retrieval_qa"
    SAFETY = "safety"
    MULTI_STEP = "multi_step"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class EvalMetricType(str, Enum):
    EXACT_MATCH = "exact_match"
    ROUGE_L = "rouge_l"
    BLEU_SCORE = "bleu_score"
    JACCARD_SIMILARITY = "jaccard_similarity"
    JSON_SCHEMA_MATCH = "json_schema_match"
    TOOL_CALL_F1 = "tool_call_f1"
    SAFETY_COMPLIANCE = "safety_compliance"
    LATENCY_SCORE = "latency_score"
    COST_EFFICIENCY = "cost_efficiency"
    COMPLETENESS = "completeness"
    OVERALL_SCORE = "overall_score"


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: Optional[str] = None


@dataclass
class AgentInput:
    prompt: str
    system_prompt: Optional[str] = None
    tools: List[ToolDefinition] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class AgentOutput:
    response_text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: Optional[str] = None
    raw_response: Any = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class BenchmarkScenario:
    scenario_id: str
    name: str
    category: TaskCategory
    difficulty: DifficultyLevel
    input_data: AgentInput
    expected_output: Optional[str] = None
    expected_json: Optional[Dict[str, Any]] = None
    expected_tool_calls: List[ToolCall] = field(default_factory=list)
    forbidden_tokens: List[str] = field(default_factory=list)
    rubric: Dict[str, float] = field(default_factory=dict)
    weight: float = 1.0
    tags: List[str] = field(default_factory=list)
    max_latency_ms: float = 5000.0
    max_cost_usd: float = 0.10


@dataclass
class EvaluationScore:
    exact_match: float = 0.0
    rouge_l: float = 0.0
    bleu_score: float = 0.0
    jaccard_similarity: float = 0.0
    json_schema_match: float = 0.0
    tool_call_precision: float = 0.0
    tool_call_recall: float = 0.0
    tool_call_f1: float = 0.0
    safety_compliance: float = 1.0
    latency_score: float = 1.0
    cost_score: float = 1.0
    completeness: float = 0.0
    overall_score: float = 0.0
    metric_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioRunResult:
    scenario_id: str
    scenario_name: str
    category: TaskCategory
    difficulty: DifficultyLevel
    agent_id: str
    passed: bool
    score: EvaluationScore
    agent_output: AgentOutput
    execution_time_ms: float
    timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None


@dataclass
class AgentSummaryScore:
    agent_id: str
    total_scenarios: int
    scenarios_passed: int
    pass_rate: float
    mean_overall_score: float
    category_scores: Dict[str, float]
    difficulty_scores: Dict[str, float]
    mean_latency_ms: float
    p95_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost_usd: float
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)


@dataclass
class BenchmarkSuiteResult:
    suite_id: str
    suite_name: str
    total_scenarios: int
    timestamp: float
    agent_summaries: Dict[str, AgentSummaryScore]
    run_records: Dict[str, List[ScenarioRunResult]]
    rankings: List[str]
    summary_notes: str
