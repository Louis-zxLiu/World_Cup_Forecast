from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from .schemas import LLMSettings, PublicLLMSettings, PublicSearchSettings, SearchSettings


def mask_key(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}...{api_key[-4:]}"


def public_llm_settings(settings: LLMSettings) -> PublicLLMSettings:
    return PublicLLMSettings(
        base_url=settings.base_url,
        api_key_masked=mask_key(settings.api_key),
        api_key_saved=bool(settings.api_key),
        model=settings.model,
        temperature=settings.temperature,
        timeout_seconds=settings.timeout_seconds,
        enabled=settings.enabled,
    )


def public_search_settings(settings: SearchSettings) -> PublicSearchSettings:
    return PublicSearchSettings(
        provider=settings.provider,
        base_url=settings.base_url,
        api_key_masked=mask_key(settings.api_key),
        api_key_saved=bool(settings.api_key),
        timeout_seconds=settings.timeout_seconds,
        max_results=settings.max_results,
        enabled=settings.enabled,
    )


class OpenAICompatibleClient:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self.settings.enabled or not self.settings.api_key:
            raise RuntimeError("LLM is disabled or API key is empty")
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def test_connection(self) -> dict[str, str | bool]:
        text = await self.complete("Reply with exactly OK.", "Health check")
        return {"ok": True, "message": text[:200]}

    async def stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        """Yield response text incrementally via the OpenAI streaming API."""
        if not self.settings.enabled or not self.settings.api_key:
            raise RuntimeError("LLM is disabled or API key is empty")
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "stream": True,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
