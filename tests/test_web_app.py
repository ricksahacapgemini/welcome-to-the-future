import tempfile
import unittest
from pathlib import Path

from task_tracker_web import create_app


class WebAppTests(unittest.TestCase):
    def test_index_lists_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(store_path=Path(tmpdir) / "tasks.json")
            client = app.test_client()

            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("Task Manager", response.get_data(as_text=True))

    def test_add_and_delete_task_via_web_routes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(store_path=Path(tmpdir) / "tasks.json")
            client = app.test_client()

            add_response = client.post(
                "/add",
                data={"title": "Ship web app", "priority": "high", "due_date": "2026-10-05"},
                follow_redirects=True,
            )
            self.assertEqual(add_response.status_code, 200)
            self.assertIn("Ship web app", add_response.get_data(as_text=True))

            delete_response = client.post("/delete/1", follow_redirects=True)
            self.assertEqual(delete_response.status_code, 200)
            self.assertNotIn("Ship web app", delete_response.get_data(as_text=True))

    def test_search_and_metadata_are_rendered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(store_path=Path(tmpdir) / "tasks.json")
            client = app.test_client()

            client.post(
                "/add",
                data={
                    "title": "Prepare launch",
                    "priority": "high",
                    "due_date": "2026-10-06",
                    "category": "Product",
                    "notes": "Finalize outreach",
                },
                follow_redirects=True,
            )
            client.post(
                "/add",
                data={
                    "title": "Inbox cleanup",
                    "priority": "low",
                    "due_date": "2026-10-07",
                    "category": "Admin",
                    "notes": "Sort incoming emails",
                },
                follow_redirects=True,
            )

            response = client.get("/?q=launch")
            self.assertEqual(response.status_code, 200)
            self.assertIn("Prepare launch", response.get_data(as_text=True))
            self.assertIn("Finalize outreach", response.get_data(as_text=True))
            self.assertNotIn("Inbox cleanup", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
