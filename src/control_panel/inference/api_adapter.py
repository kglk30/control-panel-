"""API Adapter — OpenAI-compatible remote inference endpoint.

Calls any OpenAI-compatible chat completions endpoint.
Reports backend="api", device="remote".

Configuration via environment: CONTROL_PANEL_API_KEY
Configuration via config: inference.api_base_url, inference.api_model
"""

from __future__ import annotations

import time
from typing import Optional

from control_panel.common.models import BackendInfo, InferenceRequest, InferenceResult
from control_panel.common.config import get_api_key, get_config
from control_panel.common.logging import get_logger
from control_panel.inference.models import ModelProfile

__all__ = ["APIAdapter"]

log = get_logger(__name__)


class APIAdapter:
    """Calls an OpenAI-compatible chat completions endpoint."""

    def __init__(self) -> None:
        cfg = get_config().inference
        self._base_url = cfg.api_base_url
        self._model = cfg.api_model
        self._timeout = cfg.api_timeout_seconds
        self._max_retries = cfg.api_max_retries

    @property
    def name(self) -> str:
        return "api"

    @property
    def available(self) -> bool:
        """Available if an API key is configured."""
        return get_api_key() is not None

    def check_availability(self) -> BackendInfo:
        if not get_api_key():
            return BackendInfo(
                name=self.name,
                available=False,
                device="remote",
                reason="CONTROL_PANEL_API_KEY env var not set",
            )
        return BackendInfo(name=self.name, available=True, device="remote")

    def execute(
        self,
        request: InferenceRequest,
        model: Optional[ModelProfile] = None,
        runtime: str = "api",
    ) -> InferenceResult:
        """Call the remote API and return the response."""
        t0 = time.perf_counter_ns()
        api_key = get_api_key()

        if not api_key:
            return self._error_result(request, t0, "API key not configured")

        try:
            import httpx

            messages = [{"role": "user", "content": request.prompt}]
            if request.context:
                system_msg = str(request.context.get("system", ""))
                if system_msg:
                    messages.insert(0, {"role": "system", "content": system_msg})

            payload: dict = {
                "model": self._model,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            if request.response_schema:
                payload["response_format"] = {"type": "json_object"}

            last_error: Optional[str] = None
            for attempt in range(self._max_retries + 1):
                try:
                    with httpx.Client(timeout=self._timeout) as client:
                        resp = client.post(
                            f"{self._base_url}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                            },
                            json=payload,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        content = data["choices"][0]["message"]["content"]
                        usage = data.get("usage", {})
                        tokens_used = usage.get("total_tokens", 0)
                        latency_ms = int((time.perf_counter_ns() - t0) / 1_000_000)

                        structured = None
                        if request.response_schema:
                            import json
                            try:
                                structured = json.loads(content)
                            except Exception:
                                pass

                        log.info(
                            "api.completed",
                            model=self._model,
                            tokens=tokens_used,
                            latency_ms=latency_ms,
                            attempt=attempt,
                        )

                        return InferenceResult(
                            request_id=request.request_id,
                            content=content,
                            structured=structured,
                            backend_used="api",
                            device_used="remote",
                            runtime_type="api",
                            model_id=self._model,
                            tokens_used=tokens_used,
                            latency_ms=latency_ms,
                            source="remote",
                        )

                except httpx.HTTPStatusError as e:
                    last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                    if e.response.status_code < 500:
                        break  # 4xx errors won't be fixed by retry
                except httpx.TimeoutException:
                    last_error = f"Timeout after {self._timeout}s (attempt {attempt + 1})"
                except Exception as e:
                    last_error = str(e)

                if attempt < self._max_retries:
                    import asyncio
                    time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s

            return self._error_result(request, t0, last_error or "Unknown API error")

        except ImportError:
            return self._error_result(request, t0, "httpx not installed")
        except Exception as e:
            return self._error_result(request, t0, str(e))

    def _error_result(self, request: InferenceRequest, t0: int, error: str) -> InferenceResult:
        latency_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
        log.warning("api.failed", error=error)
        return InferenceResult(
            request_id=request.request_id,
            content=f"[API error: {error}]",
            backend_used="api",
            device_used="remote",
            runtime_type="api",
            model_id=self._model,
            latency_ms=latency_ms,
            source="remote",
        )
