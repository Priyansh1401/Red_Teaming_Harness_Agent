from typing import Any, TypedDict


class Attempt(TypedDict):
    strategy_category: str
    prompt: str
    response: str
    success: bool
    severity: float
    justification: str


class HarnessState(TypedDict):
    target_spec: str
    attempt_history: list[Attempt]
    strategies_tried: list[str]
    budget_remaining: int
    max_severity_found: float
    final_report: str
    current_strategy_category: str
    current_prompt: str
    current_response: str
    current_target_error: bool
    current_success: bool
    current_severity: float
    current_justification: str
    debug_output_path: str
