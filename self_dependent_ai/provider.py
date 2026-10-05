"""OpenAI-compatible chat-completions adapter using only the standard library."""

from __future__ import annotations

import json
import os
from typing import Iterable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ModelProvider(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant's message content."""


class DemoProvider:
    """A deterministic mock provider for local demos and validation without external credentials."""

    def __init__(
        self,
        name: str,
        capabilities: Iterable[str] | None = None,
        summary: str = "demo provider",
    ):
        self.name = name
        self.capabilities = set(capabilities or [])
        self.summary = summary
        self._explicit_name = True

    def generate(self, messages: list[dict[str, str]]) -> str:
        prompt = "\n".join(item.get("content", "") for item in messages)
        capability = next(iter(self.capabilities), "research")
        if capability == "review":
            payload = {
                "provider": self.name,
                "capability": capability,
                "status": "ready",
                "summary": "reviewed and approved the demo provider output",
                "tool": "read_file",
                "params": {"path": "README.md"},
                "decision": "accept",
            }
        else:
            payload = {
                "provider": self.name,
                "capability": capability,
                "status": "ready",
                "summary": f"demo {capability} output for: {prompt[:120]}",
                "tool": "search_files" if capability == "research" else "write_file",
                "params": {"query": "demo"} if capability == "research" else {"path": "README.md", "content": "demo output"},
                "decision": "continue",
            }
        return json.dumps(payload)


class OpenAICompatibleProvider:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 60,
        name: str | None = None,
        capabilities: Iterable[str] | None = None,
        demo: bool = False,
    ):
        if not demo and not api_key:
            raise ValueError("An API key is required. Set OPENAI_API_KEY.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.model = model
        self.timeout = timeout
        self._explicit_name = name is not None
        self.name = name or model or "demo-provider"
        self.capabilities = set(capabilities or [])
        self.demo = demo

    @classmethod
    def from_environment(cls) -> OpenAICompatibleProvider:
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> OpenAICompatibleProvider | DemoProvider:
        definition = dict(config)
        if bool(definition.get("demo")) or (not str(definition.get("api_key", "")).strip() and not str(definition.get("base_url", "")).strip() and not str(definition.get("model", "")).strip()):
            return DemoProvider(
                name=str(definition.get("name") or definition.get("model") or "demo-provider"),
                capabilities=definition.get("capabilities", []),
                summary=str(definition.get("summary") or "demo provider"),
            )
        return cls(
            api_key=str(definition.get("api_key", "")),
            base_url=str(definition.get("base_url", "https://api.openai.com/v1")),
            model=str(definition.get("model", "gpt-4o-mini")),
            name=definition.get("name"),
            capabilities=definition.get("capabilities", []),
        )

    @staticmethod
    def _default_registry_file() -> str | None:
        candidates = [
            os.path.join(os.getcwd(), "demo_providers.json"),
            os.path.join(os.getcwd(), ".self_dependent_ai", "providers.json"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo_providers.json"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), ".self_dependent_ai", "providers.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    @classmethod
    def from_environment_registry(cls) -> dict[str, OpenAICompatibleProvider]:
        raw = os.environ.get("SELF_DEPENDENT_PROVIDERS", "")
        if not raw.strip():
            demo_path = cls._default_registry_file()
            if demo_path is None:
                return {}
            with open(demo_path, "r", encoding="utf-8") as handle:
                raw = handle.read()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("SELF_DEPENDENT_PROVIDERS must be valid JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("SELF_DEPENDENT_PROVIDERS must contain a JSON object mapping provider names to configs.")

        providers: dict[str, OpenAICompatibleProvider] = {}
        for name, config in payload.items():
            if isinstance(config, cls):
                config.name = name
                providers[name] = config
                continue
            if not isinstance(config, dict):
                raise ValueError(f"Provider config for '{name}' must be an object.")
            config_payload = {**config, "name": name}
            if not str(config_payload.get("api_key") or "").strip() and os.environ.get("OPENAI_API_KEY"):
                config_payload["api_key"] = os.environ["OPENAI_API_KEY"]
            if not str(config_payload.get("base_url") or "").strip() and os.environ.get("OPENAI_BASE_URL"):
                config_payload["base_url"] = os.environ["OPENAI_BASE_URL"]
            if not str(config_payload.get("model") or "").strip() and os.environ.get("OPENAI_MODEL"):
                config_payload["model"] = os.environ["OPENAI_MODEL"]
            provider = cls.from_config(config_payload)
            providers[name] = provider
        return providers

    def generate(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps({"model": self.model, "messages": messages}).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Model API returned HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(f"Could not reach the model API: {error}") from error

        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Model API response did not contain a chat message.") from error
