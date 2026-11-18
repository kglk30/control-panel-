"""Rule-Based Adapter — always-available, zero-dependency inference fallback.

Handles recognized file-organization patterns without any model.
Reports backend="rule_based", device="none".

This adapter exists for reliability and graceful degradation.
It is NOT the primary hackathon demo path.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from control_panel.common.models import BackendInfo, InferenceRequest, InferenceResult
from control_panel.common.logging import get_logger
from control_panel.inference.models import ModelProfile

__all__ = ["RuleBasedAdapter"]

log = get_logger(__name__)

# Recognized task patterns mapped to structured responses
_FILE_ORG_PATTERNS = [
    (re.compile(r"organiz|sort|arrang|categori|group", re.I), "file_management.organization"),
    (re.compile(r"clean|tidy|declutter", re.I), "file_management.cleanup"),
    (re.compile(r"rename|batch.?rename", re.I), "file_management.rename"),
    (re.compile(r"duplicat|find.?duplic", re.I), "file_management.deduplication"),
]

_TASK_TYPE_RESPONSES: dict[str, str] = {
    "analyze": (
        "Based on pattern analysis: This appears to be a file organization task. "
        "Objective: organize files by type/category. "
        "Suggested subtasks: scan directory, classify files, create category folders, move files."
    ),
    "classify": "FILE_ORGANIZATION",
    "plan": (
        "Execution plan: "
        "1. Scan the target directory to inventory all files. "
        "2. Classify each file by extension/type. "
        "3. Create category subdirectories (Images, Documents, Videos, Audio, Archives, Other). "
        "4. Move files into their respective category directories."
    ),
    "reason": "Pattern-based reasoning: File organization by extension is the safest strategy.",
    "generate": "Generated response: Task accepted. Will organize files by type into subdirectories.",
    "embed": "",  # Embeddings require a real model
}

_DEFAULT_MODEL_ID = "rule_based_v1"


class RuleBasedAdapter:
    """Pattern-matching inference adapter. Always available.

    Used as the last fallback when all other backends are unavailable.
    Handles file organization patterns with deterministic responses.
    """

    @property
    def name(self) -> str:
        return "rule_based"

    @property
    def available(self) -> bool:
        return True  # Always available — no external dependencies

    def check_availability(self) -> BackendInfo:
        """Return current availability status."""
        return BackendInfo(
            name=self.name,
            available=True,
            device="none",
            reason=None,
        )

    def execute(
        self,
        request: InferenceRequest,
        model: Optional[ModelProfile] = None,
        runtime: str = "rule_based",
    ) -> InferenceResult:
        """Execute rule-based pattern matching on the request prompt."""
        t0 = time.perf_counter_ns()

        content = self._match(request.prompt, request.task_type)

        # Attempt to parse structured output if schema requested
        structured = None
        if request.response_schema and request.task_type == "analyze":
            structured = self._build_task_spec(request.prompt)

        latency_ms = int((time.perf_counter_ns() - t0) / 1_000_000)

        log.debug(
            "rule_based.executed",
            task_type=request.task_type,
            latency_ms=latency_ms,
        )

        return InferenceResult(
            request_id=request.request_id,
            content=content,
            structured=structured,
            backend_used="rule_based",
            device_used="none",
            runtime_type="rule_based",
            model_id=_DEFAULT_MODEL_ID,
            tokens_used=0,
            latency_ms=latency_ms,
            source="rule_based",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match(self, prompt: str, task_type: str) -> str:
        """Match the prompt against known patterns and return a response."""
        # Check for file organization patterns
        for pattern, _category in _FILE_ORG_PATTERNS:
            if pattern.search(prompt):
                response = _TASK_TYPE_RESPONSES.get(task_type, "")
                if response:
                    return response

        # Generic fallback
        return (
            f"Rule-based response for task_type='{task_type}': "
            f"Prompt understood. No specific pattern matched — "
            f"defaulting to file organization workflow."
        )

    def _build_task_spec(self, prompt: str) -> dict:
        """Build a structured TaskSpec dict from pattern matching."""
        category = "file_management.organization"
        for pattern, cat in _FILE_ORG_PATTERNS:
            if pattern.search(prompt):
                category = cat
                break

        return {
            "objective": "Organize files in the target directory",
            "subtasks": ["scan_directory", "classify_files", "organize_files"],
            "category": category,
            "novelty_score": 0.1,
        }
