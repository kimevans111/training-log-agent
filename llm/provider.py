"""LLM provider abstraction with a mock fallback."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseLLMProvider(ABC):
    """Abstract interface for future OpenAI-compatible providers."""

    @abstractmethod
    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate an answer from prompt and structured context."""


class MockLLMProvider(BaseLLMProvider):
    """Deterministic fallback provider used when no API key is available."""

    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        summary = context.get("summary", {})
        diagnoses = context.get("diagnoses", {}).get("diagnoses", [])
        suggestions = context.get("suggestions", {}).get("priority_suggestions", [])

        headline = summary.get("headline", "No summary is available.")
        main_issue = diagnoses[0].get("type") if diagnoses else "no major issue"
        first_suggestion = suggestions[0] if suggestions else "Use the run as a baseline and change one variable at a time."
        return (
            f"Question: {prompt}\n\n"
            f"Based on the parsed log, {headline} The most important detected issue is "
            f"`{main_issue}`. Recommended next action: {first_suggestion}"
        )


class OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI-compatible chat completion client using urllib."""

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.model = model or "gpt-4o-mini"

    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        system_text = (
            "You are a deep learning training log analyst for 3D plant point cloud segmentation experiments. "
            "Answer concisely based on the provided context."
        )
        context_text = json.dumps(context, ensure_ascii=False, indent=2)
        user_message = f"Context:\n{context_text}\n\nQuestion: {prompt}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.2,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            return str(body["choices"][0]["message"]["content"])
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            return MockLLMProvider().generate(f"Provider request failed: {exc}\n\n{prompt}", context)
        except Exception:
            return MockLLMProvider().generate(prompt, context)


def get_llm_provider() -> BaseLLMProvider:
    """Create an LLM provider from environment variables, falling back to mock."""

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("QWEN_API_KEY")
    use_real_llm = os.getenv("USE_REAL_LLM", "false").lower() in {"1", "true", "yes"}
    if api_key and use_real_llm:
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("OPENAI_MODEL"),
        )
    return MockLLMProvider()

