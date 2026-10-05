"""A lightweight personal task tracker for the local workspace."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


class TaskTracker:
    """Store and manipulate tasks as JSON in a local file."""

    VALID_PRIORITIES = {"low", "medium", "high"}

    def __init__(self, store_path: str | Path | None = None):
        self.store_path = Path(store_path) if store_path is not None else Path.cwd() / ".task_tracker.json"
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self.store_path.write_text("[]", encoding="utf-8")

    def _load_tasks(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []

    def _save_tasks(self, tasks: list[dict[str, Any]]) -> None:
        self.store_path.write_text(json.dumps(tasks, indent=2, sort_keys=True), encoding="utf-8")

    def _normalize_due_date(self, due_date: str | None) -> str | None:
        if due_date is None:
            return None
        candidate = str(due_date).strip()
        if not candidate:
            return None
        try:
            parsed = date.fromisoformat(candidate)
        except ValueError as error:
            raise ValueError("Due date must be in YYYY-MM-DD format.") from error
        return parsed.isoformat()

    def add_task(
        self,
        title: str,
        priority: str = "medium",
        due_date: str | None = None,
        category: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        cleaned = title.strip()
        if not cleaned:
            raise ValueError("Task title cannot be empty.")

        normalized_priority = str(priority).strip().lower()
        if normalized_priority not in self.VALID_PRIORITIES:
            raise ValueError(f"Priority must be one of: {', '.join(sorted(self.VALID_PRIORITIES))}.")

        normalized_category = (str(category) if category is not None else "").strip()
        normalized_notes = (str(notes) if notes is not None else "").strip()

        tasks = self._load_tasks()
        task_id = max((task.get("id", 0) for task in tasks), default=0) + 1
        task = {
            "id": task_id,
            "title": cleaned,
            "done": False,
            "priority": normalized_priority,
            "due_date": self._normalize_due_date(due_date),
            "category": normalized_category,
            "notes": normalized_notes,
        }
        tasks.append(task)
        self._save_tasks(tasks)
        return task

    def list_tasks(self) -> list[dict[str, Any]]:
        tasks = self._load_tasks()
        return sorted(tasks, key=lambda task: (task.get("done") is False, task.get("due_date") or "9999-12-31", task.get("priority", "medium")))

    def complete_task(self, task_id: int) -> dict[str, Any]:
        tasks = self._load_tasks()
        for task in tasks:
            if int(task.get("id", -1)) == int(task_id):
                task["done"] = True
                self._save_tasks(tasks)
                return task
        raise KeyError(f"Task #{task_id} was not found.")

    def delete_task(self, task_id: int) -> bool:
        tasks = self._load_tasks()
        original = len(tasks)
        tasks = [task for task in tasks if int(task.get("id", -1)) != int(task_id)]
        if len(tasks) == original:
            return False
        self._save_tasks(tasks)
        return True


def _render_tasks(tasks: list[dict[str, Any]]) -> str:
    if not tasks:
        return "No tasks yet."

    lines = []
    for task in tasks:
        done_marker = "x" if task.get("done") else " "
        priority = str(task.get("priority", "medium")).upper()
        due = task.get("due_date")
        suffix = f" | due {due}" if due else ""
        lines.append(f"{task.get('id')}. [{done_marker}] {task.get('title')} [{priority}]{suffix}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Track personal tasks from a local JSON file.")
    parser.add_argument("--store", type=Path, default=Path.cwd() / ".task_tracker.json", help="Path to the task storage file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add a task")
    add_parser.add_argument("title", help="Task description to add")
    add_parser.add_argument("--priority", choices=["low", "medium", "high"], default="medium", help="Task priority")
    add_parser.add_argument("--due-date", dest="due_date", default=None, help="Due date in YYYY-MM-DD format")
    add_parser.add_argument("--category", default="", help="Optional task category")
    add_parser.add_argument("--notes", default="", help="Optional task notes")

    subparsers.add_parser("list", help="List all tasks")

    done_parser = subparsers.add_parser("done", help="Mark a task as complete")
    done_parser.add_argument("task_id", type=int, help="Task ID to complete")

    delete_parser = subparsers.add_parser("delete", help="Delete a task")
    delete_parser.add_argument("task_id", type=int, help="Task ID to remove")

    args = parser.parse_args()
    tracker = TaskTracker(args.store)

    if args.command == "add":
        task = tracker.add_task(
            args.title,
            priority=args.priority,
            due_date=args.due_date,
            category=args.category,
            notes=args.notes,
        )
        print(f"Added task #{task['id']}: {task['title']} [{task['priority'].upper()}]")
        if task.get("category"):
            print(f"Category: {task['category']}")
        if task.get("due_date"):
            print(f"Due: {task['due_date']}")
        if task.get("notes"):
            print(f"Notes: {task['notes']}")
        return 0

    if args.command == "list":
        print(_render_tasks(tracker.list_tasks()))
        return 0

    if args.command == "done":
        task = tracker.complete_task(args.task_id)
        print(f"Completed task #{task['id']}: {task['title']}")
        return 0

    if args.command == "delete":
        deleted = tracker.delete_task(args.task_id)
        if deleted:
            print(f"Deleted task #{args.task_id}.")
        else:
            print(f"Task #{args.task_id} was not found.")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
