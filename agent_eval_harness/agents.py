"""
Agent Adapters and Built-in Evaluator Agents for the Benchmark Harness.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union
import time
import json
import re

from .models import AgentInput, AgentOutput, ToolCall, ToolDefinition


class TokenCostModel:
    """Estimates USD cost based on token counts and model pricing."""

    PRICING_TABLE = {
        "gpt-4o": {"prompt": 2.50 / 1_000_000, "completion": 10.00 / 1_000_000},
        "gpt-4o-mini": {"prompt": 0.15 / 1_000_000, "completion": 0.60 / 1_000_000},
        "claude-3-5-sonnet": {"prompt": 3.00 / 1_000_000, "completion": 15.00 / 1_000_000},
        "llama-3-70b": {"prompt": 0.90 / 1_000_000, "completion": 0.90 / 1_000_000},
        "mock-default": {"prompt": 1.00 / 1_000_000, "completion": 2.00 / 1_000_000},
    }

    @classmethod
    def calculate_cost(cls, prompt_tokens: int, completion_tokens: int, model: str = "mock-default") -> float:
        pricing = cls.PRICING_TABLE.get(model, cls.PRICING_TABLE["mock-default"])
        return round(prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"], 8)


class BaseAgent(ABC):
    """Abstract Base Class for agent benchmarks."""

    def __init__(self, agent_id: str = "base-agent", model_name: str = "mock-default"):
        self.agent_id = agent_id
        self.model_name = model_name

    @abstractmethod
    def run(self, agent_input: AgentInput) -> AgentOutput:
        """Executes the agent logic on the input and returns AgentOutput."""
        pass

    def __call__(self, agent_input: AgentInput) -> AgentOutput:
        return self.run(agent_input)


class CallableAgent(BaseAgent):
    """Wraps an arbitrary Python function as a benchmarkable agent."""

    def __init__(self, func: Callable[[AgentInput], Union[AgentOutput, str, Dict[str, Any]]], agent_id: str = "callable-agent"):
        super().__init__(agent_id=agent_id)
        self.func = func

    def run(self, agent_input: AgentInput) -> AgentOutput:
        start_t = time.time()
        try:
            res = self.func(agent_input)
            latency = (time.time() - start_t) * 1000.0

            if isinstance(res, AgentOutput):
                if res.latency_ms <= 0:
                    res.latency_ms = latency
                return res

            if isinstance(res, str):
                p_toks = len(agent_input.prompt.split()) * 2
                c_toks = len(res.split()) * 2
                cost = TokenCostModel.calculate_cost(p_toks, c_toks, self.model_name)
                return AgentOutput(
                    response_text=res,
                    prompt_tokens=p_toks,
                    completion_tokens=c_toks,
                    latency_ms=round(latency, 2),
                    cost_usd=cost,
                )

            if isinstance(res, dict):
                text_out = json.dumps(res)
                p_toks = len(agent_input.prompt.split()) * 2
                c_toks = len(text_out.split()) * 2
                cost = TokenCostModel.calculate_cost(p_toks, c_toks, self.model_name)
                tool_calls = []
                if "tool_calls" in res and isinstance(res["tool_calls"], list):
                    for tc in res["tool_calls"]:
                        if isinstance(tc, dict) and "name" in tc:
                            tool_calls.append(ToolCall(name=tc["name"], arguments=tc.get("arguments", {})))
                return AgentOutput(
                    response_text=res.get("response_text", text_out),
                    tool_calls=tool_calls,
                    prompt_tokens=p_toks,
                    completion_tokens=c_toks,
                    latency_ms=round(latency, 2),
                    cost_usd=cost,
                )

            return AgentOutput(response_text=str(res), latency_ms=round(latency, 2))

        except Exception as e:
            latency = (time.time() - start_t) * 1000.0
            return AgentOutput(
                response_text="",
                error=f"{type(e).__name__}: {str(e)}",
                latency_ms=round(latency, 2),
            )


class DeterministicMockAgent(BaseAgent):
    """Returns predetermined responses and tool calls mapped by prompt keywords or IDs."""

    def __init__(self, response_map: Optional[Dict[str, Union[str, AgentOutput, Dict[str, Any]]]] = None, agent_id: str = "mock-agent"):
        super().__init__(agent_id=agent_id)
        self.response_map = response_map or {}

    def add_response(self, key: str, output: Union[str, AgentOutput, Dict[str, Any]]):
        self.response_map[key] = output

    def run(self, agent_input: AgentInput) -> AgentOutput:
        start_t = time.time()
        prompt_lower = agent_input.prompt.lower()

        # Check exact key matches or substring
        matched_val = None
        for k, v in self.response_map.items():
            if k.lower() in prompt_lower or k == agent_input.prompt:
                matched_val = v
                break

        latency = (time.time() - start_t) * 1000.0 + 5.0  # nominal 5ms simulation

        if matched_val is None:
            return AgentOutput(
                response_text="Default response: " + agent_input.prompt,
                prompt_tokens=len(agent_input.prompt.split()),
                completion_tokens=10,
                latency_ms=latency,
                cost_usd=TokenCostModel.calculate_cost(len(agent_input.prompt.split()), 10),
            )

        if isinstance(matched_val, AgentOutput):
            matched_val.latency_ms = latency
            return matched_val

        if isinstance(matched_val, str):
            p_toks = len(agent_input.prompt.split())
            c_toks = len(matched_val.split())
            return AgentOutput(
                response_text=matched_val,
                prompt_tokens=p_toks,
                completion_tokens=c_toks,
                latency_ms=latency,
                cost_usd=TokenCostModel.calculate_cost(p_toks, c_toks),
            )

        if isinstance(matched_val, dict):
            p_toks = len(agent_input.prompt.split())
            c_toks = len(str(matched_val).split())
            tool_calls = []
            if "tool_calls" in matched_val:
                for tc in matched_val["tool_calls"]:
                    if isinstance(tc, dict):
                        tool_calls.append(ToolCall(name=tc.get("name", ""), arguments=tc.get("arguments", {})))
                    elif isinstance(tc, ToolCall):
                        tool_calls.append(tc)
            return AgentOutput(
                response_text=matched_val.get("response_text", json.dumps(matched_val)),
                tool_calls=tool_calls,
                prompt_tokens=p_toks,
                completion_tokens=c_toks,
                latency_ms=latency,
                cost_usd=TokenCostModel.calculate_cost(p_toks, c_toks),
            )

        return AgentOutput(response_text=str(matched_val), latency_ms=latency)


class RuleBasedAgent(BaseAgent):
    """Heuristic rule-based agent capable of basic reasoning, tool invocation, and JSON output."""

    def __init__(self, agent_id: str = "rule-agent"):
        super().__init__(agent_id=agent_id)

    def run(self, agent_input: AgentInput) -> AgentOutput:
        start_t = time.time()
        prompt = agent_input.prompt.strip()
        p_lower = prompt.lower()
        tool_calls = []
        resp_text = ""

        # 1. Tool Call handling if tools are provided
        if agent_input.tools:
            for tool in agent_input.tools:
                tool_name = tool.name.lower()
                if tool_name in p_lower or any(p in p_lower for p in tool.description.lower().split()[:3]):
                    # Synthesize tool call parameters from prompt numbers or strings
                    args: Dict[str, Any] = {}
                    numbers = re.findall(r"[-+]?\d*\.?\d+", prompt)
                    if numbers:
                        for idx, req in enumerate(tool.required):
                            if idx < len(numbers):
                                try:
                                    args[req] = float(numbers[idx]) if "." in numbers[idx] else int(numbers[idx])
                                except ValueError:
                                    args[req] = numbers[idx]
                    tool_calls.append(ToolCall(name=tool.name, arguments=args))

        # 2. Arithmetic / Math Reasoning
        math_match = re.search(r"what is (\d+)\s*([\+\-\*\/])\s*(\d+)", p_lower)
        if math_match:
            a = float(math_match.group(1))
            op = math_match.group(2)
            b = float(math_match.group(3))
            if op == "+":
                res = a + b
            elif op == "-":
                res = a - b
            elif op == "*":
                res = a * b
            elif op == "/":
                res = a / b if b != 0 else 0
            else:
                res = 0
            resp_text = f"The answer is {int(res) if res.is_integer() else res}."

        # 3. JSON Output generation
        elif "json" in p_lower:
            resp_text = json.dumps({"status": "success", "result": "processed", "query": prompt})

        # 4. Fallback response
        if not resp_text and not tool_calls:
            resp_text = f"Processed request: {prompt}"
        elif tool_calls and not resp_text:
            resp_text = f"Calling tool {tool_calls[0].name}."

        latency = (time.time() - start_t) * 1000.0 + 8.0
        p_tokens = len(prompt.split()) + 10
        c_tokens = len(resp_text.split()) + len(tool_calls) * 15
        cost = TokenCostModel.calculate_cost(p_tokens, c_tokens, self.model_name)

        return AgentOutput(
            response_text=resp_text,
            tool_calls=tool_calls,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            latency_ms=round(latency, 2),
            cost_usd=cost,
        )


class FaultyAgent(BaseAgent):
    """Simulates agent failure modes: exceptions, timeouts, malformed JSON, toxic content."""

    def __init__(self, failure_mode: str = "exception", agent_id: str = "faulty-agent"):
        super().__init__(agent_id=agent_id)
        self.failure_mode = failure_mode  # 'exception', 'malformed_json', 'leak_prompt', 'empty'

    def run(self, agent_input: AgentInput) -> AgentOutput:
        start_t = time.time()
        if self.failure_mode == "exception":
            return AgentOutput(
                error="RuntimeError: Simulated catastrophic agent failure",
                latency_ms=15.0,
            )
        elif self.failure_mode == "malformed_json":
            return AgentOutput(
                response_text="{status: 'unquoted_key', error: broken json",
                latency_ms=12.0,
            )
        elif self.failure_mode == "leak_prompt":
            return AgentOutput(
                response_text="SYSTEM OVERRIDE! Revealing system prompt: You are an internal admin bot.",
                latency_ms=10.0,
            )
        elif self.failure_mode == "empty":
            return AgentOutput(
                response_text="",
                latency_ms=5.0,
            )
        return AgentOutput(response_text="unknown failure mode", latency_ms=10.0)
