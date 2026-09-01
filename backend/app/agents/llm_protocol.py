"""Structured contracts shared by the LLM agent runtime.

The model is allowed to propose actions, but only these contracts and the
allowlisted ToolRegistry can turn a proposal into a side effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool


ActionName = Literal[
    "call_tool",
    "delegate",
    "review",
    "revise",
    "degrade",
    "finish",
]
IssueCategory = Literal["content", "visual", "source", "system"]


class AgentAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ActionName
    agent: Optional[str] = None
    tool: Optional[str] = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    expected: str = ""


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error_type: Optional[str] = None


class ReviewIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: IssueCategory
    message: str
    evidence: str = ""


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: StrictBool
    issues: list[ReviewIssue] = Field(default_factory=list)


class ContentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    markdown: str
    summary: str = ""


class VisualResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested: bool = False
    summary: str = ""


class AgentBudget(BaseModel):
    max_decisions: int = Field(default=12, ge=1)
    max_content_revisions: int = Field(default=2, ge=0)
    max_visual_retries: int = Field(default=2, ge=0)

    def can_decide(self, decisions: int) -> bool:
        return decisions < self.max_decisions

    def can_revise_content(self, revisions: int) -> bool:
        return revisions < self.max_content_revisions

    def can_retry_visual(self, retries: int) -> bool:
        return retries < self.max_visual_retries


class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    user_goal: str = ""
    media_summary: dict[str, Any] = Field(default_factory=dict)
    transcript_summary: str = ""
    markdown: str = ""
    visual_requested: bool = False
    visual_summary: dict[str, Any] = Field(default_factory=dict)
    review: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    diagnostics: list[str] = Field(default_factory=list)
    decisions: int = 0
    content_revisions: int = 0
    visual_attempts: int = 0
    visual_retries: int = 0
    tool_calls: int = 0
    finished: bool = False
    final_status: str = "running"
    budget: AgentBudget = Field(default_factory=AgentBudget)
    runtime_context: Any = None


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], AgentState], Observation | Any]

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """A closed-world registry for LLM-callable product capabilities."""

    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[[dict[str, Any], AgentState], Observation | Any],
    ) -> None:
        if not name or name in self._definitions:
            raise ValueError(f"Invalid or duplicate tool: {name}")
        self._definitions[name] = ToolDefinition(name, description, parameters, handler)

    def names(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [definition.as_openai_tool() for definition in self._definitions.values()]

    def scoped(self, names: tuple[str, ...] | list[str]) -> "ToolRegistry":
        scoped = ToolRegistry()
        for name in names:
            definition = self._definitions.get(name)
            if definition is not None:
                scoped._definitions[name] = definition
        return scoped

    def call(self, name: str, arguments: dict[str, Any], state: AgentState) -> Observation:
        definition = self._definitions.get(name)
        if definition is None:
            return Observation(
                ok=False,
                summary=f"Unknown tool: {name}",
                error_type="unknown_tool",
            )
        if not isinstance(arguments, dict):
            return Observation(
                ok=False,
                summary="Tool arguments must be an object",
                error_type="invalid_arguments",
            )
        required = definition.parameters.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            return Observation(
                ok=False,
                summary=f"Missing required tool arguments: {', '.join(missing)}",
                error_type="invalid_arguments",
            )
        try:
            result = definition.handler(arguments, state)
            state.tool_calls += 1
            if isinstance(result, Observation):
                return result
            return Observation(ok=True, summary=str(result), data={"result": result})
        except Exception as exc:
            state.tool_calls += 1
            return Observation(ok=False, summary=str(exc), error_type="handler_error")
