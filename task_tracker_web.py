from __future__ import annotations

from datetime import date
from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for

from task_tracker import TaskTracker


PRIORITY_VALUE = {"low": 1, "medium": 2, "high": 3}


def create_app(store_path: str | Path | None = None) -> Flask:
    app = Flask(__name__)
    app.config["TASK_STORE"] = Path(store_path) if store_path is not None else Path.cwd() / ".task_tracker.json"
    app.config["TASK_STORE"].parent.mkdir(parents=True, exist_ok=True)

    def get_tracker() -> TaskTracker:
        return TaskTracker(app.config["TASK_STORE"])

    @app.route("/")
    def index():
        tracker = get_tracker()
        query = (request.args.get("q") or "").strip().lower()
        status = (request.args.get("status") or "all").strip().lower()

        tasks = tracker.list_tasks()
        if status == "active":
            tasks = [task for task in tasks if not task.get("done")]
        elif status == "done":
            tasks = [task for task in tasks if task.get("done")]

        if query:
            tasks = [
                task
                for task in tasks
                if query in " ".join(
                    [
                        str(task.get("title", "")),
                        str(task.get("category", "")),
                        str(task.get("notes", "")),
                    ]
                ).lower()
            ]

        def task_sort_key(task):
            due = task.get("due_date") or "9999-12-31"
            priority = PRIORITY_VALUE.get(str(task.get("priority", "medium")).lower(), 2)
            return (task.get("done") is False, due, -priority)

        tasks = sorted(tasks, key=task_sort_key)

        all_tasks = tracker.list_tasks()
        total_done = sum(1 for task in all_tasks if task.get("done"))
        total_active = len(all_tasks) - total_done
        overdue = 0
        today = date.today().isoformat()
        for task in all_tasks:
            due = task.get("due_date")
            if due and not task.get("done") and due < today:
                overdue += 1

        return render_template(
            "index.html",
            tasks=tasks,
            query=query,
            status=status,
            total=len(all_tasks),
            total_done=total_done,
            total_active=total_active,
            overdue=overdue,
            today=date.today().isoformat(),
        )

    @app.route("/add", methods=["POST"])
    def add_task():
        title = (request.form.get("title") or "").strip()
        if title:
            tracker = get_tracker()
            tracker.add_task(
                title,
                priority=request.form.get("priority", "medium"),
                due_date=request.form.get("due_date") or None,
                category=request.form.get("category") or None,
                notes=request.form.get("notes") or None,
            )
        return redirect(url_for("index"))

    @app.route("/delete/<int:task_id>", methods=["POST"])
    def delete_task(task_id: int):
        tracker = get_tracker()
        tracker.delete_task(task_id)
        return redirect(url_for("index"))

    @app.route("/done/<int:task_id>", methods=["POST"])
    def mark_done(task_id: int):
        tracker = get_tracker()
        try:
            tracker.complete_task(task_id)
        except KeyError:
            pass
        return redirect(url_for("index"))

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)
