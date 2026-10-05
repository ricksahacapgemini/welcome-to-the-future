"""Safe multi-model routing and handoff coordination for a bounded agent."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

from self_dependent_ai.provider import OpenAICompatibleProvider


class ModelCapabilityProvider(Protocol):
    """A model adapter that exposes one or more specialist capabilities."""

    name: str
    capabilities: set[str]

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Return the model's assistant response as a string."""


@dataclass(frozen=True)
class Handoff:
    capability: str
    provider_name: str
    summary: str


class MemoryStore:
    """A simple JSON-backed memory store for persistent task context."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
        return payload if isinstance(payload, dict) else {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class ExecutionState:
    goal: str
    status: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def mark_updated(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    kind: str
    requires_approval: bool = True


@dataclass(frozen=True)
class ToolCall:
    name: str
    params: dict[str, Any]
    reason: str


@dataclass(frozen=True)
class ModelHandoff:
    from_provider: str
    to_provider: str
    capability: str
    request: str
    result: dict[str, Any]
    status: str = "completed"


@dataclass
class ModelContract:
    provider: str
    capability: str
    status: str
    summary: str
    tool: str
    params: dict[str, Any]
    decision: str = "continue"
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "provider": self.provider,
            "capability": self.capability,
            "status": self.status,
            "summary": self.summary,
            "tool": self.tool,
            "params": self.params,
            "decision": self.decision,
        }
        if self.result is not None:
            payload["result"] = self.result
        return payload


@dataclass
class RunReport:
    goal: str
    workspace: str
    status: str
    winner: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    session_key: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "workspace": self.workspace,
            "status": self.status,
            "winner": self.winner,
            "candidates": self.candidates,
            "scores": self.scores,
            "decisions": self.decisions,
            "outputs": self.outputs,
            "steps": self.steps,
            "summary": self.summary,
            "session_key": self.session_key,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunReport":
        return cls(
            goal=str(payload.get("goal", "")),
            workspace=str(payload.get("workspace", "")),
            status=str(payload.get("status", "pending")),
            winner=dict(payload.get("winner") or {}),
            candidates=list(payload.get("candidates") or []),
            scores=[float(score) for score in (payload.get("scores") or [])],
            decisions=[str(decision) for decision in (payload.get("decisions") or [])],
            outputs=[str(output) for output in (payload.get("outputs") or [])],
            steps=list(payload.get("steps") or []),
            summary=str(payload.get("summary") or ""),
            session_key=payload.get("session_key"),
            created_at=str(payload.get("created_at") or datetime.now(timezone.utc).isoformat()),
            updated_at=str(payload.get("updated_at") or datetime.now(timezone.utc).isoformat()),
        )


def validate_model_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"valid": False, "error": "Contract payload must be a mapping."}

    required_keys = {"provider", "capability", "status", "summary", "tool", "params", "decision"}
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        return {"valid": False, "error": f"Missing required contract fields: {', '.join(missing)}"}

    if not isinstance(payload["provider"], str) or not payload["provider"].strip():
        return {"valid": False, "error": "Contract provider must be a non-empty string."}
    if not isinstance(payload["capability"], str) or not payload["capability"].strip():
        return {"valid": False, "error": "Contract capability must be a non-empty string."}
    if not isinstance(payload["summary"], str) or not payload["summary"].strip():
        return {"valid": False, "error": "Contract summary must be a non-empty string."}
    if not isinstance(payload["tool"], str) or not payload["tool"].strip():
        return {"valid": False, "error": "Contract tool must be a non-empty string."}
    if not isinstance(payload["params"], dict):
        return {"valid": False, "error": "Contract params must be a dictionary."}
    if not isinstance(payload["decision"], str) or not payload["decision"].strip():
        return {"valid": False, "error": "Contract decision must be a non-empty string."}

    status = str(payload["status"]).lower()
    if status not in {"ready", "running", "completed", "blocked", "failed", "recovered"}:
        return {"valid": False, "error": f"Unsupported contract status: {payload['status']}"}

    return {"valid": True, "contract": dict(payload)}


class ToolRegistry:
    """Maps capabilities to safe execution tools that can be chosen by the agent."""

    def __init__(self, tools: Mapping[str, ToolSpec] | None = None):
        self.tools = dict(tools or {})
        if not self.tools:
            self.tools = {
                "list_files": ToolSpec("list_files", "List files in the workspace", "read"),
                "search_files": ToolSpec("search_files", "Search for files or content in the workspace", "read"),
                "read_file": ToolSpec("read_file", "Read a file from the workspace", "read"),
                "patch_file": ToolSpec("patch_file", "Patch an existing file with a minimal change", "write"),
                "write_file": ToolSpec("write_file", "Write a file in the workspace", "write"),
                "run_tests": ToolSpec("run_tests", "Run the project test suite", "verify"),
            }

    def select_for_goal(self, goal: str, capability: str) -> ToolSpec:
        normalized = goal.casefold()
        if capability == "research":
            if any(token in normalized for token in ("search", "find", "grep", "lookup", "scan")):
                return self.tools.get("search_files", next(iter(self.tools.values())))
            if any(token in normalized for token in ("read", "inspect", "api", "review", "check")):
                return self.tools.get("read_file", next(iter(self.tools.values())))
            return self.tools.get("list_files", next(iter(self.tools.values())))
        if capability == "coding":
            if any(token in normalized for token in ("patch", "fix", "update", "modify", "refactor")):
                return self.tools.get("patch_file", self.tools.get("write_file", next(iter(self.tools.values()))))
            return self.tools.get("write_file", next(iter(self.tools.values())))
        if capability == "verification":
            return self.tools.get("run_tests", next(iter(self.tools.values())))
        return next(iter(self.tools.values()))

    def available(self) -> list[str]:
        return sorted(self.tools)


class ModelRouter:
    """Routes work to the provider best suited for a given capability."""

    def __init__(self, providers: Mapping[str, ModelCapabilityProvider]):
        normalized: dict[str, ModelCapabilityProvider] = {}
        for name, provider in providers.items():
            if not hasattr(provider, "name"):
                provider.name = name
            if not hasattr(provider, "capabilities"):
                provider.capabilities = set()
            provider.capabilities = set(getattr(provider, "capabilities", set()))
            explicit_name = getattr(provider, "_explicit_name", False)
            if not explicit_name and getattr(provider, "name", name) == getattr(provider, "model", ""):
                provider.name = name
            provider.name = str(getattr(provider, "name", name) or name)
            normalized[name] = provider
        self.providers = normalized
        if not self.providers:
            raise ValueError("At least one provider is required.")

    def capabilities(self) -> set[str]:
        roles: set[str] = set()
        for provider in self.providers.values():
            roles.update(getattr(provider, "capabilities", set()))
        return roles

    def route(self, capability: str) -> ModelCapabilityProvider:
        matches = [
            provider
            for provider in self.providers.values()
            if capability in getattr(provider, "capabilities", set())
        ]
        if not matches:
            available = ", ".join(sorted(self.capabilities())) or "none"
            raise ValueError(
                f"No provider supports capability '{capability}'. Available: {available}."
            )
        return sorted(matches, key=lambda provider: provider.name)[0]

    @staticmethod
    def _demo_registry_file() -> Path | None:
        candidates = [
            Path.cwd() / "demo_providers.json",
            Path.cwd() / ".self_dependent_ai" / "providers.json",
            Path(__file__).resolve().parent.parent / "demo_providers.json",
            Path(__file__).resolve().parent.parent / ".self_dependent_ai" / "providers.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @classmethod
    def from_environment(cls) -> "ModelRouter":
        config = os.environ.get("SELF_DEPENDENT_PROVIDERS", "")
        if not config.strip():
            demo_file = cls._demo_registry_file()
            if demo_file is not None:
                try:
                    payload = json.loads(demo_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Demo config '{demo_file}' must contain valid JSON.") from error
                if not isinstance(payload, dict):
                    raise ValueError(f"Demo config '{demo_file}' must contain a JSON object keyed by provider name.")
                config = json.dumps(payload)
            else:
                raise ValueError("SELF_DEPENDENT_PROVIDERS is not configured and no demo config file was found.")
        try:
            payload = json.loads(config)
        except json.JSONDecodeError as error:
            raise ValueError("SELF_DEPENDENT_PROVIDERS must be valid JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("SELF_DEPENDENT_PROVIDERS must be a JSON object keyed by provider name.")

        providers: dict[str, ModelCapabilityProvider] = {}
        for name, raw in payload.items():
            if isinstance(raw, OpenAICompatibleProvider):
                provider = raw
                provider.name = name
            else:
                if not isinstance(raw, dict):
                    raise ValueError(f"Provider '{name}' must contain a config object.")
                config_payload = {**raw, "name": name}
                if not str(config_payload.get("api_key") or "").strip() and os.environ.get("OPENAI_API_KEY"):
                    config_payload["api_key"] = os.environ["OPENAI_API_KEY"]
                if not str(config_payload.get("base_url") or "").strip() and os.environ.get("OPENAI_BASE_URL"):
                    config_payload["base_url"] = os.environ["OPENAI_BASE_URL"]
                if not str(config_payload.get("model") or "").strip() and os.environ.get("OPENAI_MODEL"):
                    config_payload["model"] = os.environ["OPENAI_MODEL"]
                provider = OpenAICompatibleProvider.from_config(config_payload)
            providers[name] = provider
        return cls(providers)


class GoalPlanner:
    """Infers the small set of capabilities needed to carry out a user's goal."""

    _CAPABILITY_KEYWORDS = {
        "research": {
            "research", "researching", "discover", "find", "lookup", "analyze", "investigate",
            "api", "contract", "schema", "specification", "requirements", "documentation",
        },
        "coding": {
            "build", "code", "implement", "develop", "fix", "create", "write", "modify",
            "patch", "client", "feature", "module", "function", "app",
        },
        "verification": {
            "verify", "validate", "test", "check", "review", "audit", "debug", "quality",
            "deployment", "production", "confirm", "ensure", "validate",
        },
    }

    _SEQUENCE_ORDER = {"research": 0, "coding": 1, "verification": 2}

    def __init__(self, router: ModelRouter):
        self.router = router

    def infer_capabilities(self, goal: str) -> list[str]:
        if not goal.strip():
            raise ValueError("A goal is required before inferring capabilities.")

        text = goal.casefold()
        scores: dict[str, int] = {cap: 0 for cap in self.router.capabilities()}

        for capability, keywords in self._CAPABILITY_KEYWORDS.items():
            if capability not in scores:
                continue
            for keyword in keywords:
                if keyword in text:
                    scores[capability] += 2

        if "before" in text and "implement" in text:
            scores["research"] += 3
        if "client" in text and "implement" in text:
            scores["coding"] += 3
        if "validate" in text or "verify" in text or "test" in text:
            scores["verification"] += 4
        if "api" in text and "contract" in text:
            scores["research"] += 4
        if "deployment" in text or "production" in text:
            scores["verification"] += 3

        candidates = [capability for capability in self.router.capabilities() if scores[capability] > 0]
        if candidates:
            positions: dict[str, int] = {}
            for capability in candidates:
                hits = []
                for keyword in self._CAPABILITY_KEYWORDS.get(capability, set()):
                    index = text.find(keyword)
                    if index >= 0:
                        hits.append(index)
                positions[capability] = min(hits) if hits else len(text) + 1
            ordered = sorted(candidates, key=lambda capability: (positions.get(capability, len(text) + 1), self._SEQUENCE_ORDER.get(capability, 99)))
            return ordered

        return [capability for capability in sorted(self.router.capabilities(), key=lambda item: self._SEQUENCE_ORDER.get(item, 99))]


class AutonomousExecutionEngine:
    """Persistent execution engine that selects tools and routes work through model specialists."""

    _DEPENDENCY_ORDER = {
        "research": [],
        "coding": ["research"],
        "verification": ["coding"],
    }

    def __init__(self, router: ModelRouter, tool_registry: ToolRegistry | None = None, memory_store: MemoryStore | None = None):
        self.router = router
        self.tool_registry = tool_registry or ToolRegistry()
        self.coordinator = CollaborativeCoordinator(router, tool_registry=self.tool_registry)
        self.memory_store = memory_store or MemoryStore(Path(".self_dependent_ai") / "memory.json")

    def _session_key(self, goal: str, workspace: Path) -> str:
        return f"session:{abs(hash(f'{goal}:{workspace.resolve()}'))}"

    def _dependency_order(self, goal: str) -> list[str]:
        plan = self.coordinator.build_plan(goal)
        ordered = [step.capability for step in plan]
        return ordered

    def _safe_workspace_path(self, workspace: Path, candidate: str | Path) -> Path:
        root = Path(workspace).resolve()
        target = (root / str(candidate)).resolve() if not Path(str(candidate)).is_absolute() else Path(str(candidate)).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Path is outside the workspace: {candidate}")
        return target

    def _list_files(self, workspace: Path, params: Mapping[str, Any]) -> list[str]:
        root = Path(workspace).resolve()
        base = params.get("path", ".")
        base_path = self._safe_workspace_path(root, base)
        if base_path.is_file():
            return [str(base_path.relative_to(root))]
        return sorted(str(path.relative_to(root)) for path in base_path.rglob("*") if path.is_file())

    def _search_files(self, workspace: Path, params: Mapping[str, Any]) -> list[str]:
        query = str(params.get("query") or params.get("pattern") or "").strip().casefold()
        if not query:
            raise ValueError("A search query is required.")
        results: list[str] = []
        for path in Path(workspace).resolve().rglob("*"):
            if not path.is_file():
                continue
            rel = str(path.relative_to(Path(workspace).resolve()))
            text = path.read_text(encoding="utf-8", errors="ignore")
            if query in rel.casefold() or query in text.casefold():
                results.append(rel)
        return sorted(results)

    def _read_file(self, workspace: Path, params: Mapping[str, Any]) -> str:
        path = self._safe_workspace_path(workspace, str(params.get("path") or params.get("file") or ""))
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_text(encoding="utf-8")

    def _write_file(self, workspace: Path, params: Mapping[str, Any]) -> str:
        rel_path = str(params.get("path") or params.get("file") or "")
        if not rel_path:
            raise ValueError("A file path is required for a write operation.")
        target = self._safe_workspace_path(workspace, rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = params.get("content", "")
        if not isinstance(content, str):
            raise ValueError("File content must be a string.")
        target.write_text(content, encoding="utf-8")
        return str(target.relative_to(Path(workspace).resolve()))

    def _patch_file(self, workspace: Path, params: Mapping[str, Any]) -> str:
        rel_path = str(params.get("path") or params.get("file") or "")
        if not rel_path:
            raise ValueError("A file path is required for a patch operation.")
        target = self._safe_workspace_path(workspace, rel_path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        content = target.read_text(encoding="utf-8")
        replacement = params.get("content")
        if isinstance(replacement, str):
            target.write_text(replacement, encoding="utf-8")
            return str(target.relative_to(Path(workspace).resolve()))
        old = params.get("old")
        new = params.get("new")
        if isinstance(old, str) and isinstance(new, str):
            target.write_text(content.replace(old, new), encoding="utf-8")
            return str(target.relative_to(Path(workspace).resolve()))
        raise ValueError("Patch requires either content or old/new text pairs.")

    def _run_tests(self, workspace: Path, params: Mapping[str, Any]) -> dict[str, Any]:
        command = params.get("command")
        if command is None:
            command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
        if isinstance(command, str):
            command = command.split()
        if not isinstance(command, list) or not command:
            raise ValueError("Verification payload must provide a command list.")

        environment = {
            name: value
            for name, value in os.environ.items()
            if not any(token in name.casefold() for token in ("key", "token", "secret", "password", "credential"))
        }
        completed = subprocess.run(
            command,
            cwd=str(workspace),
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "output": (completed.stdout + completed.stderr).strip(),
        }

    def execute_tool_call(self, tool_name: str, params: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
        if tool_name not in self.tool_registry.tools:
            raise KeyError(f"Unknown tool: {tool_name}")

        workspace_path = Path(workspace).resolve()
        try:
            if tool_name == "list_files":
                result = self._list_files(workspace_path, params)
                return {"tool": tool_name, "status": "completed", "files": result}
            if tool_name == "search_files":
                result = self._search_files(workspace_path, params)
                return {"tool": tool_name, "status": "completed", "matches": result}
            if tool_name == "read_file":
                result = self._read_file(workspace_path, params)
                return {"tool": tool_name, "status": "completed", "content": result}
            if tool_name == "write_file":
                result = self._write_file(workspace_path, params)
                return {"tool": tool_name, "status": "completed", "path": result}
            if tool_name == "patch_file":
                result = self._patch_file(workspace_path, params)
                return {"tool": tool_name, "status": "completed", "path": result}
            if tool_name == "run_tests":
                result = self._run_tests(workspace_path, params)
                return {"tool": tool_name, "status": "completed", "result": result}
            raise ValueError(f"Unsupported tool: {tool_name}")
        except Exception as error:  # pragma: no cover - defensive branch for real runtime failures
            payload = {
                "tool": tool_name,
                "status": "failed",
                "error": str(error),
                "workspace": str(workspace_path),
                "params": dict(params),
            }
            return payload

    def recover_from_failure(self, tool_name: str, params: Mapping[str, Any], error: Exception, workspace: Path | None = None) -> dict[str, Any]:
        fallback_tool = "list_files"
        if tool_name == "read_file":
            fallback_tool = "list_files"
        elif tool_name in {"patch_file", "write_file"}:
            fallback_tool = "write_file"
        elif tool_name == "run_tests":
            fallback_tool = "list_files"

        fallback_params = dict(params)
        target_workspace = Path(workspace).resolve() if workspace is not None else Path.cwd()
        fallback_result = self.execute_tool_call(fallback_tool, fallback_params, target_workspace)
        return {
            "tool": tool_name,
            "status": "recovered",
            "action": f"Retry with safer fallback: {fallback_tool}",
            "fallback_tool": fallback_tool,
            "fallback_result": fallback_result,
            "error": str(error),
        }

    def execute_step(
        self,
        tool_name: str,
        params: Mapping[str, Any],
        workspace: Path,
        goal: str,
        capability: str,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        workspace_path = Path(workspace).resolve()
        attempts = 0
        recoveries = 0
        last_error: Exception | None = None

        while attempts <= max_retries:
            attempts += 1
            try:
                result = self.execute_tool_call(tool_name, params, workspace_path)
                if result.get("status") == "failed":
                    raise ValueError(str(result.get("error") or "tool execution failed"))
                return {
                    "tool": tool_name,
                    "capability": capability,
                    "goal": goal,
                    "status": "completed",
                    "attempts": attempts,
                    "recoveries": recoveries,
                    "action": "executed",
                    **result,
                }
            except Exception as error:
                last_error = error
                if attempts > max_retries:
                    return {
                        "tool": tool_name,
                        "capability": capability,
                        "goal": goal,
                        "status": "failed",
                        "attempts": attempts,
                        "recoveries": recoveries,
                        "action": "failed after retries",
                        "error": str(error),
                    }
                recoveries += 1
                recovered = self.recover_from_failure(tool_name, params, error, workspace_path)
                if recovered.get("status") == "recovered":
                    return {
                        "tool": tool_name,
                        "capability": capability,
                        "goal": goal,
                        "status": "recovered",
                        "attempts": attempts,
                        "recoveries": recoveries,
                        "action": recovered.get("action", "Retry with fallback"),
                        "fallback_tool": recovered.get("fallback_tool"),
                        "fallback_result": recovered.get("fallback_result"),
                        "error": str(error),
                    }

        return {
            "tool": tool_name,
            "capability": capability,
            "goal": goal,
            "status": "failed",
            "attempts": attempts,
            "recoveries": recoveries,
            "action": "failed after retries",
            "error": str(last_error) if last_error is not None else "unknown tool failure",
        }

    def _session_record(self, goal: str, workspace: Path, state: ExecutionState) -> dict[str, Any]:
        ordered_steps = [step["capability"] for step in state.steps]
        history = []
        for index, step in enumerate(state.steps):
            capability = step.get("capability")
            tool = self.tool_registry.select_for_goal(goal, capability) if capability else None
            history.append(
                {
                    "index": index,
                    "capability": capability,
                    "provider_name": step.get("provider_name"),
                    "status": step.get("status", "completed"),
                    "tool": tool.name if tool else None,
                    "summary": step.get("summary"),
                }
            )

        pending_steps = [
            capability
            for capability in self._dependency_order(goal)
            if capability not in ordered_steps
        ]

        return {
            "goal": goal,
            "status": state.status,
            "workspace": str(Path(workspace).resolve()),
            "steps": state.steps,
            "history": history,
            "pending_steps": pending_steps,
            "dependencies": self._DEPENDENCY_ORDER,
            "updated_at": state.updated_at,
            "capabilities_used": ordered_steps,
            "tool_registry": self.tool_registry.available(),
        }

    def next_pending_step(self, session_key: str) -> str | None:
        state = self.resume_session(session_key)
        pending = state.get("pending_steps") or []
        return pending[0] if pending else None

    def resume_session(self, session_key: str) -> dict[str, Any]:
        state = self.memory_store.get(session_key)
        if state is None:
            raise KeyError(f"No session found for key: {session_key}")
        payload = {"memory_key": session_key, **state}
        if "history" not in payload:
            history = []
            for index, step in enumerate(payload.get("steps", [])):
                history.append(
                    {
                        "index": index,
                        "capability": step.get("capability"),
                        "provider_name": step.get("provider_name"),
                        "status": step.get("status", "completed"),
                        "summary": step.get("summary"),
                    }
                )
            payload["history"] = history
        if "pending_steps" not in payload:
            capabilities = [item.get("capability") for item in payload.get("steps", []) if item.get("capability")]
            payload["pending_steps"] = [
                capability for capability in self._DEPENDENCY_ORDER if capability not in capabilities
            ]
        return payload

    def resume_run(self, goal: str, workspace: Path, approve: Callable[[str], bool] | None = None) -> dict[str, Any]:
        session_key = self._session_key(goal, Path(workspace))
        payload = self.memory_store.get(session_key)
        if payload is None:
            raise KeyError(f"No saved autonomous run found for goal: {goal}")

        if payload.get("status") == "completed" and payload.get("report"):
            return payload

        if approve is None:
            approve = lambda summary: True
        return self.execute(goal, workspace, approve=approve)

    def execute(self, goal: str, workspace: Path, approve: Callable[[str], bool] | None = None) -> dict[str, Any]:
        if not goal.strip():
            raise ValueError("A goal is required before execution.")

        session_key = self._session_key(goal, Path(workspace))
        self.memory_store.set(session_key, {"goal": goal, "status": "running", "workspace": str(Path(workspace).resolve())})

        result = self.coordinator.execute_workflow(goal, workspace, approve=approve)
        negotiation = self.coordinator.negotiate(goal, self.coordinator.planner.infer_capabilities(goal), strategy="best_summary")
        report = self.coordinator.build_run_report(goal, str(Path(workspace).resolve()), negotiation, result.get("steps", []))
        result["report"] = report.to_dict()

        saved_state = result["state"]
        payload = self._session_record(goal, workspace, saved_state)
        payload["report"] = report.to_dict()
        payload["history"] = [
            {"index": index, "capability": step.get("capability"), "provider_name": step.get("provider_name"), "status": step.get("status", "completed"), "summary": step.get("summary")}
            for index, step in enumerate(result.get("steps", []))
        ]
        self.memory_store.set(session_key, payload)
        result["memory_key"] = session_key
        result["history"] = payload["history"]
        result["pending_steps"] = payload["pending_steps"]
        result["dependencies"] = payload["dependencies"]
        return result

    def _parse_provider_payload(self, raw_output: str) -> dict | None:
        try:
            payload = json.loads(raw_output)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def execute_workflow_loop(
        self,
        goal: str,
        workspace: Path,
        approve: Callable[[str], bool] | None = None,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        if not goal.strip():
            raise ValueError("A goal is required before workflow execution.")

        workspace_path = Path(workspace).resolve()
        if approve is None:
            approve = lambda summary: True

        session_key = self._session_key(goal, workspace_path)
        capabilities = self._dependency_order(goal)
        completed: set[str] = set()
        workflow_steps: list[dict[str, Any]] = []
        retry_events: list[dict[str, Any]] = []

        for capability in capabilities:
            required = self._DEPENDENCY_ORDER.get(capability, [])
            missing = [dependency for dependency in required if dependency not in completed]
            if missing:
                workflow_steps.append(
                    {
                        "capability": capability,
                        "status": "blocked",
                        "reason": f"Waiting for dependency(s): {', '.join(missing)}",
                        "attempts": 0,
                        "recoveries": 0,
                    }
                )
                continue

            provider = self.router.route(capability)
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a {capability} specialist. Return a JSON tool invocation that fits the goal, "
                        "keeping the call in a safe workspace-scoped payload."
                    ),
                },
                {"role": "user", "content": f"Goal: {goal}\nCapability: {capability}"},
            ]

            attempt = 0
            recoveries = 0
            tool_name: str | None = None
            params: dict[str, Any] = {}
            step_result: dict[str, Any] | None = None
            while attempt <= max_retries:
                attempt += 1
                try:
                    raw_output = str(provider.generate(messages))
                    payload = self._parse_provider_payload(raw_output)
                    if payload is None:
                        raise ValueError("Provider did not return a JSON tool payload.")
                    tool_name = str(payload.get("tool") or self.tool_registry.select_for_goal(goal, capability).name)
                    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
                    if not isinstance(params, dict):
                        params = {}

                    if tool_name in {"write_file", "patch_file", "run_tests"}:
                        approved = approve(f"Run tool '{tool_name}' for capability '{capability}'")
                        if not approved:
                            step_result = {
                                "capability": capability,
                                "status": "declined",
                                "tool": tool_name,
                                "attempts": attempt,
                                "recoveries": recoveries,
                                "reason": "User declined the tool execution.",
                            }
                            break

                    step_result = self.execute_step(tool_name, params, workspace_path, goal, capability, max_retries=max_retries)
                    if step_result["status"] in {"completed", "recovered"}:
                        break
                    retry_events.append(
                        {
                            "capability": capability,
                            "tool": tool_name,
                            "attempt": attempt,
                            "recoveries": recoveries,
                            "status": step_result["status"],
                            "error": step_result.get("error"),
                        }
                    )
                    recoveries = max(recoveries, step_result.get("recoveries", 0))
                except Exception as error:
                    retry_events.append(
                        {
                            "capability": capability,
                            "tool": tool_name,
                            "attempt": attempt,
                            "recoveries": recoveries,
                            "status": "failed",
                            "error": str(error),
                        }
                    )
                    if attempt > max_retries:
                        step_result = {
                            "capability": capability,
                            "status": "failed",
                            "tool": tool_name,
                            "attempts": attempt,
                            "recoveries": recoveries,
                            "error": str(error),
                        }
                        break
                    recoveries += 1
                    fallback = self.recover_from_failure(tool_name or self.tool_registry.select_for_goal(goal, capability).name, params, error, workspace_path)
                    retry_events[-1]["fallback"] = fallback

            if step_result is None:
                step_result = {
                    "capability": capability,
                    "status": "failed",
                    "tool": None,
                    "attempts": attempt,
                    "recoveries": recoveries,
                    "reason": "No execution result was produced.",
                }

            workflow_steps.append({
                "capability": capability,
                "status": step_result.get("status", "failed"),
                "tool": step_result.get("tool"),
                "attempts": step_result.get("attempts", 0),
                "recoveries": step_result.get("recoveries", 0),
                "action": step_result.get("action"),
                "error": step_result.get("error"),
                "result": step_result,
            })

            if step_result.get("status") in {"completed", "recovered"}:
                completed.add(capability)

        session_payload = {
            "goal": goal,
            "status": "completed" if all(step.get("status") in {"completed", "recovered"} for step in workflow_steps) else "failed",
            "workspace": str(workspace_path),
            "steps": workflow_steps,
            "history": workflow_steps,
            "pending_steps": [
                capability
                for capability in self._dependency_order(goal)
                if capability not in completed
            ],
            "retry_events": retry_events,
            "dependencies": self._DEPENDENCY_ORDER,
            "tool_registry": self.tool_registry.available(),
        }
        self.memory_store.set(session_key, session_payload)
        return {
            "status": session_payload["status"],
            "goal": goal,
            "workspace": str(workspace_path),
            "memory_key": session_key,
            "steps": workflow_steps,
            "history": workflow_steps,
            "pending_steps": session_payload["pending_steps"],
            "retry_events": retry_events,
            "dependencies": self._DEPENDENCY_ORDER,
        }


class CollaborativeCoordinator:
    """Builds a safe task plan that can be delegated to specialist providers."""

    def __init__(
        self,
        router: ModelRouter,
        planner: GoalPlanner | None = None,
        tool_registry: ToolRegistry | None = None,
        memory_store: MemoryStore | None = None,
    ):
        self.router = router
        self.planner = planner or GoalPlanner(router)
        self.tool_registry = tool_registry or ToolRegistry()
        self.memory_store = memory_store or MemoryStore(Path(".self_dependent_ai") / "provider_memory.json")

    def record_provider_outcome(
        self,
        provider_name: str,
        capability: str,
        success: bool,
        quality: float = 0.0,
        summary: str = "",
        decision: str = "continue",
        **metadata: Any,
    ) -> dict[str, Any]:
        history = self.memory_store.get("provider_history", {})
        provider_history = history.setdefault(provider_name, {})
        records = provider_history.setdefault(capability, [])
        record = {
            "success": bool(success),
            "quality": float(quality),
            "summary": str(summary),
            "decision": str(decision),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        records.append(record)
        self.memory_store.set("provider_history", history)
        return record

    def get_provider_history(self, provider_name: str, capability: str) -> list[dict[str, Any]]:
        history = self.memory_store.get("provider_history", {})
        return list(history.get(provider_name, {}).get(capability, []))

    def provider_confidence(self, provider_name: str, capability: str) -> float:
        history = self.get_provider_history(provider_name, capability)
        if not history:
            return 0.5
        weighted = []
        for record in history:
            success_score = 1.0 if bool(record.get("success")) else 0.0
            quality = float(record.get("quality", 0.0))
            weighted.append(success_score * 0.8 + quality * 0.2)
        return sum(weighted) / len(weighted)

    def _provider_run_metadata(self, capability: str) -> dict[str, dict[str, Any]]:
        metadata: dict[str, dict[str, Any]] = {}
        for provider in self.router.providers.values():
            if capability in getattr(provider, "capabilities", set()):
                provider_name = getattr(provider, "name", "unknown")
                metadata[provider_name] = {
                    "confidence": self.provider_confidence(provider_name, capability),
                    "history": self.get_provider_history(provider_name, capability),
                }
        return metadata

    def _build_handoffs(self, goal: str, capabilities: Iterable[str]) -> list[Handoff]:
        handoffs: list[Handoff] = []
        for capability in capabilities:
            provider = self.router.route(capability)
            handoffs.append(
                Handoff(
                    capability=capability,
                    provider_name=provider.name,
                    summary=f"Use {provider.name} for {capability} work on: {goal}",
                )
            )
        return handoffs

    def plan(self, goal: str, required_capabilities: Iterable[str] | None = None) -> list[Handoff]:
        if not goal.strip():
            raise ValueError("A goal is required before planning collaboration.")

        capabilities = list(required_capabilities) if required_capabilities is not None else self.planner.infer_capabilities(goal)
        return self._build_handoffs(goal, capabilities)

    def build_plan(self, goal: str) -> list[Handoff]:
        """Convenience entry point for the full automatic pipeline: infer then route."""
        return self.plan(goal)

    def _score_contract(self, contract: Mapping[str, Any], goal: str, strategy: str = "best_summary") -> float:
        if strategy == "best_summary":
            summary = str(contract.get("summary", "")).casefold()
            target_text = f"{goal} {summary}".casefold()
            score = 0.0
            for keyword in ("targeted", "specific", "clear", "minimal", "direct", "precise", "reliable"):
                if keyword in summary:
                    score += 2.5
            for keyword in ("broad", "general", "unclear", "vague"):
                if keyword in summary:
                    score -= 2.0
            for keyword in ("api", "alpha", "research", "reduce", "uncertainty", "before", "coding"):
                if keyword in target_text:
                    score += 1.5
            score += len(summary.split()) * 0.2
            return score
        return 0.0

    def negotiate(
        self,
        goal: str,
        capabilities: Iterable[str] | None = None,
        strategy: str = "best_summary",
    ) -> dict[str, Any]:
        if not goal.strip():
            raise ValueError("A goal is required before model negotiation.")

        requested = list(capabilities) if capabilities is not None else self.planner.infer_capabilities(goal)
        candidates: list[dict[str, Any]] = []
        provider_metadata = {}
        for capability in requested:
            providers = [
                provider
                for provider in self.router.providers.values()
                if capability in getattr(provider, "capabilities", set())
            ]
            if not providers:
                providers = [self.router.route(capability)]

            for provider in sorted(providers, key=lambda item: getattr(item, "name", "")):
                history = self.get_provider_history(provider.name, capability)
                confidence = self.provider_confidence(provider.name, capability)
                provider_metadata[provider.name] = {
                    "capability": capability,
                    "confidence": confidence,
                    "history": history,
                }
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a specialist model. Return a structured contract with a status, summary, tool, "
                            "params, and decision for the next step in the workflow."
                        ),
                    },
                    {"role": "user", "content": f"Goal: {goal}\nCapability: {capability}"},
                ]
                raw_output = provider.generate(messages)
                parsed = self._parse_provider_payload(raw_output)
                if parsed is None:
                    parsed = {
                        "provider": provider.name,
                        "capability": capability,
                        "status": "failed",
                        "summary": "No valid provider contract was returned.",
                        "tool": self.tool_registry.select_for_goal(goal, capability).name,
                        "params": {},
                        "decision": "halt",
                    }
                validation = validate_model_contract(parsed)
                if not validation["valid"]:
                    parsed = {
                        "provider": provider.name,
                        "capability": capability,
                        "status": "failed",
                        "summary": f"Contract validation failed: {validation['error']}",
                        "tool": self.tool_registry.select_for_goal(goal, capability).name,
                        "params": {},
                        "decision": "halt",
                    }
                contract = dict(parsed)
                base_score = self._score_contract(contract, goal, strategy)
                adjusted_score = base_score + (confidence * 5.0)
                candidate = {
                    "provider": contract.get("provider"),
                    "capability": contract.get("capability"),
                    "decision": contract.get("decision"),
                    "status": contract.get("status"),
                    "summary": contract.get("summary"),
                    "tool": contract.get("tool"),
                    "params": contract.get("params", {}),
                    "confidence": confidence,
                    "score": adjusted_score,
                }
                candidates.append(candidate)

        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
        winner = ranked[0] if ranked else {"provider": None, "decision": "halt", "status": "failed"}
        return {
            "status": "negotiated",
            "strategy": strategy,
            "goal": goal,
            "winner": winner,
            "candidates": ranked,
            "capabilities": requested,
            "providers": {
                provider_name: {
                    "capability": payload["capability"],
                    "confidence": payload["confidence"],
                    "history": payload["history"],
                }
                for provider_name, payload in provider_metadata.items()
            },
        }

    def hand_off(
        self,
        from_provider: str,
        to_provider: str,
        capability: str,
        request: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        if from_provider not in self.router.providers:
            raise ValueError(f"Unknown source provider: {from_provider}")
        if to_provider not in self.router.providers:
            raise ValueError(f"Unknown reviewer provider: {to_provider}")

        reviewer = self.router.providers[to_provider]
        validation = validate_model_contract(dict(result))
        if not validation["valid"]:
            return {
                "status": "rejected",
                "reviewer": to_provider,
                "from_provider": from_provider,
                "capability": capability,
                "request": request,
                "decision": "reject",
                "reason": validation["error"],
                "result": dict(result),
            }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a reviewer model. Validate the prior provider output, accept it when it meets the goal, "
                    "and return a structured contract with status and decision."
                ),
            },
            {"role": "user", "content": f"Goal: {request}\nCapability: {capability}\nCandidate result: {json.dumps(result, sort_keys=True)}"},
        ]
        raw_output = reviewer.generate(messages)
        payload = self._parse_provider_payload(raw_output)
        if payload is None:
            payload = {
                "provider": reviewer.name,
                "capability": capability,
                "status": "failed",
                "summary": "Review provider did not return a valid contract.",
                "tool": self.tool_registry.select_for_goal(request, "verification").name,
                "params": {},
                "decision": "reject",
            }

        review_validation = validate_model_contract(payload)
        if not review_validation["valid"]:
            return {
                "status": "rejected",
                "reviewer": to_provider,
                "from_provider": from_provider,
                "capability": capability,
                "request": request,
                "decision": "reject",
                "reason": review_validation["error"],
                "result": dict(result),
            }

        decision = str(payload.get("decision", "continue")).lower()
        status = "validated" if decision in {"accept", "continue", "approved"} else "rejected"
        return {
            "status": status,
            "reviewer": to_provider,
            "from_provider": from_provider,
            "capability": capability,
            "request": request,
            "decision": decision,
            "result": dict(result),
            "review": payload,
        }

    def select_tool(self, goal: str, capability: str) -> ToolSpec:
        return self.tool_registry.select_for_goal(goal, capability)

    def execute_goal(self, goal: str) -> list[dict[str, Any]]:
        """Infer the required capability set, route each capability to a provider, and execute it."""
        if not goal.strip():
            raise ValueError("A goal is required before execution.")

        plan = self.build_plan(goal)
        results: list[dict[str, Any]] = []
        for step in plan:
            provider = self.router.providers.get(step.provider_name)
            if provider is None:
                raise ValueError(f"Provider '{step.provider_name}' is not available in the router.")

            tool = self.select_tool(goal, step.capability)
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a {step.capability} specialist working on a controlled task. "
                        "Focus on the requested outcome, stay within the stated goal, and provide a concise result."
                    ),
                },
                {"role": "user", "content": f"Goal: {goal}\nTask: {step.summary}\nPreferred tool: {tool.name}"},
            ]
            try:
                raw_output = provider.generate(messages)
                output = str(raw_output)
                payload = self._parse_provider_payload(output)
                if payload is None:
                    payload = {"result": output, "provider": provider.name, "capability": step.capability}
                handoff = ModelHandoff(
                    from_provider=provider.name,
                    to_provider=provider.name,
                    capability=step.capability,
                    request=f"Goal: {goal} | Tool: {tool.name}",
                    result=payload,
                )
            except Exception as error:  # pragma: no cover - defensive path for real providers
                output = f"Execution failed for {step.capability}: {type(error).__name__}: {error}"
                payload = {"result": output, "provider": provider.name, "capability": step.capability, "error": str(error)}
                handoff = ModelHandoff(
                    from_provider=provider.name,
                    to_provider=provider.name,
                    capability=step.capability,
                    request=f"Goal: {goal} | Tool: {tool.name}",
                    result=payload,
                    status="failed",
                )

            results.append(
                {
                    "capability": step.capability,
                    "provider_name": step.provider_name,
                    "summary": step.summary,
                    "output": output,
                    "payload": payload,
                    "status": handoff.status,
                    "handoff": {
                        "from_provider": handoff.from_provider,
                        "to_provider": handoff.to_provider,
                        "capability": handoff.capability,
                        "request": handoff.request,
                        "result": handoff.result,
                        "status": handoff.status,
                    },
                }
            )
        return results

    def parse_contract(self, raw_output: str) -> dict[str, Any]:
        payload = self._parse_provider_payload(raw_output)
        if payload is None:
            return {"valid": False, "error": "Contract payload is not valid JSON."}

        validation = validate_model_contract(payload)
        if not validation["valid"]:
            return {"valid": False, "error": validation["error"], "raw": payload}
        return {"valid": True, "contract": validation["contract"]}

    def _parse_provider_payload(self, raw_output: str) -> dict | None:
        try:
            payload = json.loads(raw_output)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _apply_coding_output(self, workspace: Path, raw_output: str, approve: Callable[[str], bool]) -> str:
        payload = self._parse_provider_payload(raw_output)
        if payload is None:
            return "No coding edit payload was returned by the provider."

        rel_path = payload.get("path")
        content = payload.get("content")
        if not isinstance(rel_path, str) or not rel_path.strip() or not isinstance(content, str):
            return "Coding provider output did not include a workspace-safe path and content."

        target = (workspace / rel_path).resolve()
        try:
            relative = target.relative_to(workspace.resolve())
        except ValueError as error:
            raise ValueError("Coding output tried to write outside the workspace.") from error

        summary = f"Write {relative.as_posix()} ({len(content)} characters)"
        if not approve(summary):
            return f"Edit declined by user: {relative.as_posix()} was not changed."

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {relative.as_posix()}."

    def _run_verification_output(self, workspace: Path, raw_output: str, approve: Callable[[str], bool]) -> str:
        payload = self._parse_provider_payload(raw_output)
        if payload is None:
            payload = {"type": "run_tests", "command": [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]}

        if payload.get("type") not in {"run_tests", "verification", None}:
            return f"Verification output requested unsupported action: {payload.get('type')}"

        command = payload.get("command")
        if command is None:
            command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
        if not isinstance(command, list) or not command:
            raise ValueError("Verification payload must provide a command list.")

        if not approve(f"Run verification command: {' '.join(str(part) for part in command)}"):
            return "Verification declined by user; no test run started."

        environment = {
            name: value
            for name, value in os.environ.items()
            if not any(token in name.casefold() for token in ("key", "token", "secret", "password", "credential"))
        }

        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace),
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Verification command exceeded the 60-second limit.") from error

        output = (completed.stdout + completed.stderr).strip()
        return f"Exit code: {completed.returncode}\n{output}"

    def build_run_report(
        self,
        goal: str,
        workspace: str,
        negotiation: Mapping[str, Any],
        steps: Iterable[Mapping[str, Any]],
        session_key: str | None = None,
    ) -> RunReport:
        candidates = list(negotiation.get("candidates", []))
        winner = negotiation.get("winner")
        score_values = [float(item.get("score", 0.0)) for item in candidates]
        decisions = [str(item.get("decision", "continue")) for item in candidates]
        outputs = [str(step.get("output", "")) for step in steps]
        summary = (
            f"Completed {len(outputs)} step(s) with winner {winner.get('provider') if winner else 'none'} "
            f"and {len(candidates)} ranked candidates."
        )
        return RunReport(
            goal=goal,
            workspace=workspace,
            status="completed" if steps else "failed",
            winner=dict(winner or {}),
            candidates=candidates,
            scores=score_values,
            decisions=decisions,
            outputs=outputs,
            steps=[dict(step) for step in steps],
            summary=summary,
            session_key=session_key,
        )

    def execute_workflow(
        self,
        goal: str,
        workspace: Path,
        approve: Callable[[str], bool] | None = None,
    ) -> dict[str, object]:
        """Execute the inferred task plan against a real workspace, applying edits and verification steps."""
        if not goal.strip():
            raise ValueError("A goal is required before executing the workflow.")
        workspace_path = Path(workspace).resolve()
        if not workspace_path.exists() or not workspace_path.is_dir():
            raise ValueError(f"Workspace is not a directory: {workspace_path}")

        if approve is None:
            approve = lambda summary: True

        candidate_map: dict[str, dict[str, Any]] = {}
        negotiation = self.negotiate(goal, self.planner.infer_capabilities(goal), strategy="best_summary")
        for candidate in negotiation.get("candidates", []):
            capability = candidate.get("capability")
            if capability is None:
                continue
            current = candidate_map.get(capability)
            if current is None or candidate.get("score", 0.0) > current.get("score", 0.0):
                candidate_map[capability] = candidate

        plan = self.build_plan(goal)
        results: list[dict[str, Any]] = []
        capabilities_used: list[str] = []

        for step in plan:
            winner = candidate_map.get(step.capability)
            provider_name = winner.get("provider", step.provider_name) if winner else step.provider_name
            provider = self.router.providers.get(provider_name)
            if provider is None:
                provider = self.router.providers.get(step.provider_name)
            if provider is None:
                raise ValueError(f"Provider '{provider_name}' is not available in the router.")

            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a {step.capability} specialist. Return the requested result in a concise, actionable format."
                    ),
                },
                {"role": "user", "content": f"Goal: {goal}\nTask: {step.summary}\nSelected provider: {provider.name}"},
            ]
            raw_output = str(provider.generate(messages))
            output = raw_output
            payload = self._parse_provider_payload(raw_output)
            if payload is None:
                payload = {"result": raw_output, "provider": provider.name, "capability": step.capability}

            if step.capability == "coding":
                output = self._apply_coding_output(workspace_path, raw_output, approve)
            elif step.capability == "verification":
                output = self._run_verification_output(workspace_path, raw_output, approve)

            capabilities_used.append(step.capability)
            results.append(
                {
                    "capability": step.capability,
                    "provider_name": provider.name,
                    "summary": step.summary,
                    "output": output,
                    "payload": payload,
                    "status": "completed",
                }
            )

        state = ExecutionState(goal=goal, status="completed", steps=results)
        final_report = self.build_run_report(goal, str(workspace_path), negotiation, results)
        return {
            "status": "completed",
            "goal": goal,
            "workspace": str(workspace_path),
            "capabilities_used": capabilities_used,
            "steps": results,
            "state": state,
            "report": final_report.to_dict(),
        }
