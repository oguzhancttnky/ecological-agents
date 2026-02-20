from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from lab_sim.config.settings import OllamaSettings


class OllamaAdapter:
    _request_semaphore = threading.Semaphore(3)
    # Shared thread pool — sync requests run here so the async loop stays free
    _thread_pool = ThreadPoolExecutor(max_workers=10, thread_name_prefix="ollama")

    def __init__(self, settings: OllamaSettings) -> None:
        self._settings = settings
        self._logger = logging.getLogger("lab_sim.ollama")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_with_retry(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._settings.host}{endpoint}"
        attempts = self._settings.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                with self._request_semaphore:
                    response = requests.post(
                        url,
                        json=payload,
                        timeout=self._settings.timeout_seconds,
                    )
                    response.raise_for_status()
                    return response.json()
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                self._logger.warning(
                    "Ollama request retrying endpoint=%s attempt=%d/%d error=%s",
                    endpoint,
                    attempt,
                    attempts,
                    exc.__class__.__name__,
                )
                time.sleep(self._settings.retry_backoff_seconds * attempt)
        raise RuntimeError(f"Ollama request failed for {endpoint}") from last_error

    def _sync_generate_timed(self, prompt: str, timeout_s: float) -> tuple[str, float]:
        """Blocking HTTP call to Ollama — uses requests, no aiohttp.

        Returns (response_text, latency_ms).
        Raises RuntimeError on timeout or connection failure.
        """
        url = f"{self._settings.host}/api/generate"
        payload = {
            "model": self._settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self._settings.llm_temperature},
        }
        t0 = time.perf_counter()
        # timeout=None means wait indefinitely — correct for a local single-GPU Ollama
        req_timeout = timeout_s if timeout_s and timeout_s > 0 else None
        
        self._logger.info("OLLAMA request  model=%s prompt_len=%d", self._settings.llm_model, len(prompt))
        self._logger.info("OLLAMA prompt:\n%s", prompt)
        
        try:
            resp = requests.post(url, json=payload, timeout=req_timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout as exc:
            self._logger.warning("OLLAMA timeout after %.0fs", time.perf_counter() - t0)
            raise RuntimeError("LLM timeout") from exc
        except requests.exceptions.ConnectionError as exc:
            self._logger.warning("OLLAMA connection error after %.0fs: %s",
                                 time.perf_counter() - t0, exc)
            raise RuntimeError("LLM connection error") from exc
            
        latency_ms = (time.perf_counter() - t0) * 1000.0
        response_text = str(data.get("response", "")).strip()
        
        self._logger.info("OLLAMA response latency=%.0fms tokens~%d", latency_ms, len(response_text) // 4)
        self._logger.info("OLLAMA response_text:\n%s", response_text)
        
        return response_text, latency_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Synchronous LLM generation (retained for backward compatibility)."""
        payload = self._post_with_retry(
            "/api/generate",
            {
                "model": self._settings.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self._settings.llm_temperature},
            },
        )
        return str(payload.get("response", "")).strip()

    async def async_generate(
        self,
        prompt: str,
        timeout_s: float | None = None,
        semaphore: asyncio.Semaphore | None = None,
        session: Any = None,  # unused; kept for API compat
    ) -> tuple[str, float]:
        """Async-compatible LLM call via thread executor.

        Runs _sync_generate_timed in a thread pool so the asyncio event loop
        is never blocked.  Uses the same requests-based TCP connection that
        Ollama handles reliably — no aiohttp, no ServerDisconnectedError.

        Returns:
            (response_text, latency_ms)
        """
        effective_timeout = (
            timeout_s if timeout_s is not None else float(self._settings.timeout_seconds)
        )
        loop = asyncio.get_running_loop()

        async def _run() -> tuple[str, float]:
            return await loop.run_in_executor(
                self._thread_pool,
                self._sync_generate_timed,
                prompt,
                effective_timeout,
            )

        if semaphore is not None:
            async with semaphore:
                return await _run()
        return await _run()

    def embed(self, text: str) -> list[float]:
        """Synchronous embedding generation."""
        payload = self._post_with_retry(
            "/api/embeddings",
            {
                "model": self._settings.embedding_model,
                "prompt": text,
            },
        )
        return payload.get("embedding", [])
