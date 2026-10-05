"""Workspace-bounded agent loop with explicit approval for mutations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from self_dependent_ai.provider import ModelProvider


SYSTEM_PROMPT = """You are a coding assistant operating through a restricted local tool.
The user gives one coding goal. Decide one small next action at a time and respond
with exactly one JSON object: {"message": "brief status", "action": {...}}.
Available actions:
- {"type": "list_files"}
- {"type": "read_file", "path": "relative/path"}
- {"type": "propose_edit", "path": "relative/path", "content": "complete file content"}
- {"type": "run_tests"}
- {"type": "finish", "message": "final result"}
Only use paths within the selected workspace. Never request secrets or expose credentials.
File edits and test runs require user approval. After a tool error, revise your approach
rather than repeating the same failed action. Do not claim an action succeeded unless its
tool result says it did. Finish when the goal is achieved or cannot safely be achieved."""

_BLOCKED_DIRECTORIES = {".git", ".venv", "node_modules", "__pycache__"}
_MAX_FILE_BYTES = 100_000
_MAX_TOOL_OUTPUT = 12_000
_MAX_LISTED_FILES = 200


@dataclass(frozen=True)
class AgentResult:
    status: str
    message: str
    steps: int


class Agent:
    def __init__(
        self,
        provider: ModelProvider,
        workspace: Path,
        max_steps: int = 12,
        max_retries: int = 2,
    ):
        self.provider = provider
        self.workspace = workspace.resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"Workspace is not a directory: {self.workspace}")
        if max_steps < 1 or max_retries < 0:
            raise ValueError("max_steps must be positive and max_retries cannot be negative")
        self.max_steps = max_steps
        self.max_retries = max_retries

    def run(self, goal: str, approve: Callable[[str], bool]) -> AgentResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"goal": goal, "workspace": str(self.workspace)}, ensure_ascii=True
                ),
            },
        ]
        failures = 0

        for step in range(1, self.max_steps + 1):
            try:
                raw_response = self.provider.generate(messages)
                messages.append({"role": "assistant", "content": raw_response})
                payload = json.loads(raw_response)
                if not isinstance(payload, dict) or not isinstance(payload.get("action"), dict):
                    raise ValueError("Response must be an object with an action object.")
                action = payload["action"]
                status = payload.get("message", "")
                if not isinstance(status, str):
                    raise ValueError("Response message must be text.")
                if status:
                    print(f"Agent: {status}")
                if action.get("type") == "finish":
                    final_message = action.get("message", status or "Finished.")
                    if not isinstance(final_message, str):
                        raise ValueError("Finish message must be text.")
                    return AgentResult("completed", final_message, step)
                tool_result = self._execute(action, approve)
                failures = 0
            except Exception as error:
                failures += 1
                tool_result = f"Action failed: {type(error).__name__}: {error}"
                if failures > self.max_retries:
                    return AgentResult(
                        "failed", f"Stopped after reaching the retry limit. {tool_result}", step
                    )

            messages.append(
                {
                    "role": "user",
                    "content": f"Tool result (step {step}): {tool_result}",
                }
            )

        return AgentResult("limit_reached", "Stopped at the maximum step limit.", self.max_steps)

    def _execute(self, action: dict, approve: Callable[[str], bool]) -> str:
        action_type = action.get("type")
        if action_type == "list_files":
            return self._list_files()
        if action_type == "read_file":
            return self._read_file(action.get("path"))
        if action_type == "propose_edit":
            return self._propose_edit(action, approve)
        if action_type == "run_tests":
            return self._run_tests(approve)
        raise ValueError(f"Unsupported action type: {action_type!r}")

    def _safe_path(self, raw_path: object) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("A non-empty relative path is required.")
        target = (self.workspace / raw_path).resolve()
        try:
            relative = target.relative_to(self.workspace)
        except ValueError as error:
            raise ValueError("Path is outside the selected workspace.") from error
        parts = tuple(part.casefold() for part in relative.parts)
        if any(part in _BLOCKED_DIRECTORIES for part in parts):
            raise ValueError("Access to this directory is blocked.")
        filename = relative.name.casefold()
        if (
            any(part.startswith(".env") for part in parts)
            or any("credential" in part or "secret" in part for part in parts)
            or Path(filename).suffix in {".pem", ".key", ".p12", ".pfx"}
        ):
            raise ValueError("Access to likely secret material is blocked.")
        return target

    def _list_files(self) -> str:
        found: list[str] = []
        for current, directories, filenames in os.walk(self.workspace, followlinks=False):
            current_path = Path(current)
            directories[:] = sorted(
                directory
                for directory in directories
                if directory.casefold() not in _BLOCKED_DIRECTORIES
            )
            for filename in sorted(filenames):
                target = current_path / filename
                try:
                    self._safe_path(str(target.relative_to(self.workspace)))
                except ValueError:
                    continue
                found.append(target.relative_to(self.workspace).as_posix())
                if len(found) >= _MAX_LISTED_FILES:
                    return json.dumps(found + ["... file list truncated ..."])
        return json.dumps(found)

    def _read_file(self, raw_path: object) -> str:
        target = self._safe_path(raw_path)
        if not target.is_file():
            raise ValueError("The requested path is not a file.")
        if target.stat().st_size > _MAX_FILE_BYTES:
            raise ValueError(f"File exceeds the {_MAX_FILE_BYTES}-byte read limit.")
        return target.read_text(encoding="utf-8")[:_MAX_TOOL_OUTPUT]

    def _propose_edit(self, action: dict, approve: Callable[[str], bool]) -> str:
        target = self._safe_path(action.get("path"))
        content = action.get("content")
        if not isinstance(content, str):
            raise ValueError("Edit content must be text.")
        relative = target.relative_to(self.workspace).as_posix()
        if not approve(f"Write {relative} ({len(content)} characters)"):
            return f"Edit declined by user: {relative} was not changed."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {relative}."

    def _run_tests(self, approve: Callable[[str], bool]) -> str:
        if not approve("Run Python unittest discovery in the selected workspace"):
            return "Test run declined by user; no process was started."
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
        environment = {
            name: value
            for name, value in os.environ.items()
            if not any(token in name.casefold() for token in ("key", "token", "secret", "password", "credential"))
        }
        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Test run exceeded the 60-second limit.") from error
        output = (completed.stdout + completed.stderr).strip()
        summary = f"Exit code: {completed.returncode}\n{output}"
        return summary[:_MAX_TOOL_OUTPUT]
