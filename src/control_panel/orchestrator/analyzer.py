"""Task Analyzer — converts raw natural-language input to a structured TaskSpec.

Uses the Inference Runtime (any available backend).
Falls back to RuleBasedAdapter if inference fails.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from control_panel.common.logging import get_logger
from control_panel.common.models import InferenceRequest, TaskSpec, new_id, utc_now
from control_panel.inference.runtime import InferenceRuntime

__all__ = ["TaskAnalyzer"]

log = get_logger(__name__)

_ANALYZE_PROMPT_TEMPLATE = """
Analyze the following user request and extract structured information.

User request: "{raw_input}"

Return a JSON object with these fields:
- objective: string — what the user wants to accomplish (one clear sentence)
- subtasks: array of strings — ordered list of sub-operations needed
- category: string — task category using dot notation (e.g. "file_management.organization")
- novelty_score: float 0.0–1.0 — how novel/unusual this task is (0=routine, 1=unprecedented)

Respond ONLY with valid JSON. No explanation.
""".strip()

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "objective": {"type": "string"},
        "subtasks": {"type": "array", "items": {"type": "string"}},
        "category": {"type": "string"},
        "novelty_score": {"type": "number"},
    },
    "required": ["objective", "subtasks", "category"],
}

# Pattern-based category detection (used as fallback and verification)
_CATEGORY_PATTERNS = [
    (re.compile(r"organiz|sort|arrang|categori|group.*file", re.I), "file_management.organization"),
    (re.compile(r"clean|tidy|declutter", re.I), "file_management.cleanup"),
    (re.compile(r"rename|batch.?rename", re.I), "file_management.rename"),
    (re.compile(r"duplicat|find.?duplic", re.I), "file_management.deduplication"),
    (re.compile(r"mov.*file|file.*mov", re.I), "file_management.organization"),
]


def _extract_task_spec_json(text: str) -> Optional[dict]:
    """Extract a valid TaskSpec JSON payload from raw model output using raw_decode."""
    decoder = json.JSONDecoder()
    start = 0
    while True:
        idx = text.find('{', start)
        if idx == -1:
            break
        try:
            candidate, _ = decoder.raw_decode(text, idx)
            if isinstance(candidate, dict):
                if all(k in candidate for k in ("objective", "subtasks", "category")):
                    return candidate
        except json.JSONDecodeError:
            pass
        start = idx + 1
    return None


class TaskAnalyzer:
    """Converts raw user input to a structured TaskSpec via inference."""

    def __init__(self, inference_runtime: InferenceRuntime) -> None:
        self._runtime = inference_runtime

    def analyze(self, raw_input: str) -> TaskSpec:
        """Analyze raw input and return a TaskSpec.

        Tries inference first, falls back to rule-based parsing.
        """
        log.info("analyzer.start", input_len=len(raw_input))

        # Try inference
        structured = self._try_inference(raw_input)

        if structured:
            task_spec = TaskSpec(
                task_id=new_id(),
                raw_input=raw_input,
                objective=structured.get("objective", raw_input),
                subtasks=structured.get("subtasks", []),
                category=structured.get("category", self._detect_category(raw_input)),
                novelty_score=float(structured.get("novelty_score", 0.5)),
                created_at=utc_now(),
            )
        else:
            # Rule-based fallback
            task_spec = self._rule_based_analysis(raw_input)

        log.info(
            "analyzer.complete",
            category=task_spec.category,
            subtasks=len(task_spec.subtasks),
            novelty=task_spec.novelty_score,
        )
        return task_spec

    # ------------------------------------------------------------------

    def _try_inference(self, raw_input: str) -> Optional[dict]:
        """Attempt to analyze via inference runtime. Returns None on failure."""
        try:
            request = InferenceRequest(
                prompt=_ANALYZE_PROMPT_TEMPLATE.format(raw_input=raw_input),
                task_type="analyze",
                response_schema=_RESPONSE_SCHEMA,
                temperature=0.2,
                max_tokens=512,
            )
            result = self._runtime.infer(request)

            if result.structured:
                return result.structured

            # Try to extract the required JSON payload
            parsed = _extract_task_spec_json(result.content)
            if parsed is not None:
                return parsed

            raise ValueError("No valid TaskSpec JSON payload found in output")

        except Exception as e:
            log.warning("analyzer.inference_failed", error=str(e))

        return None

    def _rule_based_analysis(self, raw_input: str) -> TaskSpec:
        """Fallback: pattern-based task analysis."""
        category = self._detect_category(raw_input)

        if "file_management" in category:
            subtasks = ["scan_directory", "classify_files", "organize_files"]
            objective = f"Organize files in the specified directory by type/category"
        else:
            subtasks = ["analyze_request", "execute_task"]
            objective = raw_input

        return TaskSpec(
            task_id=new_id(),
            raw_input=raw_input,
            objective=objective,
            subtasks=subtasks,
            category=category,
            novelty_score=0.1,
            created_at=utc_now(),
        )

    def _detect_category(self, text: str) -> str:
        """Detect task category from text using patterns."""
        for pattern, category in _CATEGORY_PATTERNS:
            if pattern.search(text):
                return category
        return "general.unknown"
