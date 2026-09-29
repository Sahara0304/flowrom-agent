from __future__ import annotations

import os
import time
from typing import Any

import requests


DEFAULT_BASE_URL = "https://api.minimaxi.com"
DEFAULT_MODEL = "MiniMax-M3"


class MiniMaxError(RuntimeError):
    """Raised when a MiniMax request fails or returns an invalid payload."""


class MiniMaxClient:
    """Small native HTTP client for MiniMax text/chatcompletion_v2."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        connect_timeout: float = 15.0,
        read_timeout: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("MINIMAX_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or os.getenv("MINIMAX_MODEL") or DEFAULT_MODEL
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
        self.max_retries = max(1, int(max_retries))

        if not self.api_key:
            raise MiniMaxError(
                "MINIMAX_API_KEY is not set."
            )

        # Important: base_url is the host root, not /v1.
        self.endpoint = f"{self.base_url}/v1/text/chatcompletion_v2"
        self.session = requests.Session()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                print(
                    f"[MiniMax] POST {self.endpoint} "
                    f"(attempt {attempt}/{self.max_retries})"
                )
                response = self.session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=(self.connect_timeout, self.read_timeout),
                )
                print(f"[MiniMax] HTTP {response.status_code}")
                response.raise_for_status()

                try:
                    data = response.json()
                except ValueError as exc:
                    raise MiniMaxError(
                        "MiniMax returned a non-JSON response."
                    ) from exc

                base_resp = data.get("base_resp") or {}
                base_status = base_resp.get("status_code", 0)
                if base_status not in (0, "0", None):
                    raise MiniMaxError(
                        f"MiniMax API error: {base_resp}"
                    )

                choices = data.get("choices")
                if not choices:
                    raise MiniMaxError(
                        "MiniMax response contains no choices."
                    )

                message = choices[0].get("message") or {}
                content = message.get("content")
                if content is None:
                    raise MiniMaxError(
                        "MiniMax response contains no message.content."
                    )

                return str(content)

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = exc
            except requests.exceptions.HTTPError as exc:
                last_error = exc
                # 4xx is normally not transient; do not waste retries.
                if response.status_code < 500:
                    break
            except MiniMaxError as exc:
                last_error = exc
                # Payload errors are not normally transient.
                break

            if attempt < self.max_retries:
                wait_seconds = min(5 * (2 ** (attempt - 1)), 30)
                print(f"[MiniMax] Request failed: {last_error}")
                print(f"[MiniMax] Retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)

        raise MiniMaxError(
            f"MiniMax request failed after {self.max_retries} attempts: {last_error}"
        )


def minimax_chat(
    messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Convenience wrapper used by the three agents."""
    client = MiniMaxClient(model=model)
    return client.chat(messages, temperature=temperature)
