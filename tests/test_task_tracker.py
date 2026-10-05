import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from task_tracker import TaskTracker


class TaskTrackerTests(unittest.TestCase):
    def test_add_and_list_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = TaskTracker(Path(tmpdir) / "tasks.json")
            tracker.add_task("Write docs")
            tracker.add_task("Ship feature")

            tasks = tracker.list_tasks()
            self.assertEqual(len(tasks), 2)
            self.assertEqual(tasks[0]["title"], "Write docs")
            self.assertFalse(tasks[0]["done"])

    def test_complete_and_delete_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = TaskTracker(Path(tmpdir) / "tasks.json")
            task = tracker.add_task("Review PR")

            updated = tracker.complete_task(task["id"])
            self.assertTrue(updated["done"])

            removed = tracker.delete_task(task["id"])
            self.assertTrue(removed)
            self.assertEqual(tracker.list_tasks(), [])

    def test_add_task_with_priority_and_due_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = TaskTracker(Path(tmpdir) / "tasks.json")
            task = tracker.add_task("Prepare demo", priority="high", due_date="2026-10-05")

            self.assertEqual(task["priority"], "high")
            self.assertEqual(task["due_date"], "2026-10-05")

    def test_cli_commands_work(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            store = Path(tmpdir) / "tasks.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "task_tracker.py"),
                    "--store",
                    str(store),
                    "add",
                    "Ship release",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0)
            self.assertIn("Ship release", completed.stdout)

            view = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "task_tracker.py"),
                    "--store",
                    str(store),
                    "list",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(view.returncode, 0)
            self.assertIn("Ship release", view.stdout)


if __name__ == "__main__":
    unittest.main()
