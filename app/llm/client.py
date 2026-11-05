"""Provider-agnostic LLM client.

LLM_PROVIDER = rules      -> no LLM; every agent uses its deterministic fallback (default, needs nothing)
LLM_PROVIDER = ollama     -> local Ollama server (free, private)  [best for local development]
LLM_PROVIDER = openai     -> any OpenAI-compatible API (Groq / OpenRouter / OpenAI ...)  [good for public hosting]
LLM_PROVIDER = anthropic  -> Anthropic Messages API

The client NEVER decides business facts. Agents use it for language understanding
and wording; facts come from tools and are enforced by ``app.llm.guard``.
"""
from __future__ import annotations

import json
import re
import time

import requests

from app.config import Settings


class LLMError(RuntimeError):
    """Any failure to obtain a usable completion from the provider."""


class Provider:
    name = "base"
    model = ""

    def chat(self, system: str, prompt: str, json_mode: bool = False) -> str:  # pragma: no cover
        raise NotImplementedError

    def health(self) -> tuple[bool, str]:  # pragma: no cover
        return True, "ok"


class RulesProvider(Provider):
    """Placeholder meaning 'no LLM configured' - agents use deterministic logic."""
    name = "rules"
    model = "deterministic"

    def chat(self, system, prompt, json_mode=False):
        raise LLMError("no LLM provider configured (rule-based mode)")

    def health(self):
        return True, "Rule-based mode (no LLM configured)"


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url, self.model, self.timeout = base_url, model, timeout

    def chat(self, system, prompt, json_mode=False):
        payload = {"model": self.model, "stream": False,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                   "options": {"temperature": 0.1}}
        if json_mode:
            payload["format"] = "json"
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc

    def health(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
            if not any(m.split(":")[0] == self.model.split(":")[0] for m in models):
                return False, f"Ollama is running but model '{self.model}' is not pulled"
            return True, f"Ollama · {self.model}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Ollama unreachable ({type(exc).__name__})"


class OpenAICompatProvider(Provider):
    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float):
        self.base_url, self.api_key, self.model, self.timeout = base_url, api_key, model, timeout

    def chat(self, system, prompt, json_mode=False):
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions", timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "temperature": 0.1,
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"openai-compatible request failed: {exc}") from exc

    def health(self):
        return (True, f"OpenAI-compatible · {self.model}") if self.api_key else (False, "API key not set")


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str, timeout: float):
        self.api_key, self.model, self.timeout = api_key, model, timeout

    def chat(self, system, prompt, json_mode=False):
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages", timeout=self.timeout,
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                json={"model": self.model, "max_tokens": 800, "temperature": 0.1, "system": system,
                      "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"])
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise LLMError(f"anthropic request failed: {exc}") from exc

    def health(self):
        return (True, f"Anthropic · {self.model}") if self.api_key else (False, "API key not set")


def extract_json(text: str) -> dict:
    """Leniently parse a JSON object from model output (handles ``` fences and chatter)."""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    try:
        obj = json.loads(cleaned)
    except ValueError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("model did not return JSON")
        try:
            obj = json.loads(cleaned[start:end + 1])
        except ValueError as exc:
            raise LLMError("model returned malformed JSON") from exc
    if not isinstance(obj, dict):
        raise LLMError("model JSON is not an object")
    return obj


class LLMClient:
    """Thin wrapper adding uniform error handling and usage statistics."""

    def __init__(self, provider: Provider | None = None):
        self.provider = provider or RulesProvider()
        self.calls = 0
        self.failures = 0
        self.total_ms = 0.0

    @property
    def enabled(self) -> bool:
        return not isinstance(self.provider, RulesProvider)

    @property
    def label(self) -> str:
        return "rule-based" if not self.enabled else f"{self.provider.name}:{self.provider.model}"

    def complete(self, system: str, prompt: str, json_mode: bool = False) -> str:
        start = time.perf_counter()
        self.calls += 1
        try:
            text = self.provider.chat(system, prompt, json_mode)
            if not text or not text.strip():
                raise LLMError("empty completion")
            return text
        except LLMError:
            self.failures += 1
            raise
        except Exception as exc:  # noqa: BLE001 - normalise any provider bug
            self.failures += 1
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            self.total_ms += (time.perf_counter() - start) * 1000

    def complete_json(self, system: str, prompt: str) -> dict:
        return extract_json(self.complete(system, prompt, json_mode=True))

    def health(self) -> tuple[bool, str]:
        return self.provider.health()


def create_llm(settings: Settings | None = None) -> LLMClient:
    s = settings or Settings.load()
    if s.llm_provider == "ollama":
        return LLMClient(OllamaProvider(s.ollama_base_url, s.ollama_model, s.llm_timeout))
    if s.llm_provider in ("openai", "openai_compatible", "groq", "openrouter"):
        return LLMClient(OpenAICompatProvider(s.openai_base_url, s.openai_api_key, s.openai_model, s.llm_timeout))
    if s.llm_provider == "anthropic":
        return LLMClient(AnthropicProvider(s.anthropic_api_key, s.anthropic_model, s.llm_timeout))
    return LLMClient(RulesProvider())
