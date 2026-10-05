import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from self_dependent_ai.agent import Agent


class QueueProvider:
    def __init__(self, *responses):
        self.responses = list(responses)

    def generate(self, messages):
        return self.responses.pop(0)


class StaticProvider:
    def __init__(self, name, capabilities=None):
        self.name = name
        self.capabilities = set(capabilities or [])

    def generate(self, messages):
        return f"provider:{self.name}"


def response(action, message=""):
    return json.dumps({"message": message, "action": action})


class AgentTests(unittest.TestCase):
    def test_rejects_paths_outside_workspace_and_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = QueueProvider(
                response({"type": "read_file", "path": "../outside.txt"}),
                response({"type": "finish", "message": "Stopped safely."}),
            )
            agent = Agent(provider, Path(directory), max_steps=3, max_retries=1)

            result = agent.run("Inspect the project", approve=lambda _: True)

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.message, "Stopped safely.")
            self.assertEqual(len(provider.responses), 0)

    def test_declined_edit_does_not_write_file(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = QueueProvider(
                response({"type": "propose_edit", "path": "new.py", "content": "print('hi')"}),
                response({"type": "finish", "message": "Done."}),
            )
            agent = Agent(provider, Path(directory), max_steps=3)

            result = agent.run("Create a file", approve=lambda _: False)

            self.assertEqual(result.status, "completed")
            self.assertFalse((Path(directory) / "new.py").exists())

    def test_approved_edit_stays_in_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = QueueProvider(
                response({"type": "propose_edit", "path": "src/new.py", "content": "value = 1\n"}),
                response({"type": "finish", "message": "Done."}),
            )
            agent = Agent(provider, Path(directory), max_steps=3)

            result = agent.run("Create a file", approve=lambda _: True)

            self.assertEqual(result.status, "completed")
            self.assertEqual((Path(directory) / "src" / "new.py").read_text(), "value = 1\n")

    def test_stops_after_retry_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = QueueProvider(
                response({"type": "read_file", "path": "missing.py"}),
                response({"type": "read_file", "path": "missing.py"}),
                response({"type": "finish", "message": "Should not be reached."}),
            )
            agent = Agent(provider, Path(directory), max_steps=5, max_retries=1)

            result = agent.run("Read a file", approve=lambda _: True)

            self.assertEqual(result.status, "failed")
            self.assertIn("retry limit", result.message.lower())
            self.assertEqual(len(provider.responses), 1)

    def test_router_selects_a_provider_for_the_required_capability(self):
        from self_dependent_ai.collaboration import ModelRouter

        research = StaticProvider("research", capabilities={"research"})
        coding = StaticProvider("coding", capabilities={"coding"})
        router = ModelRouter({"research": research, "coding": coding})

        self.assertEqual(router.route("research"), research)
        self.assertEqual(router.route("coding"), coding)

    def test_router_rejects_unknown_capability(self):
        from self_dependent_ai.collaboration import ModelRouter

        router = ModelRouter({"research": StaticProvider("research", capabilities={"research"})})

        with self.assertRaises(ValueError):
            router.route("coding")

    def test_provider_registry_loads_environment_config_and_plans_handoffs(self):
        from self_dependent_ai.collaboration import CollaborativeCoordinator, ModelRouter
        from self_dependent_ai.provider import OpenAICompatibleProvider

        os.environ["SELF_DEPENDENT_PROVIDERS"] = json.dumps(
            {
                "research": {
                    "api_key": "research-key",
                    "base_url": "https://research.example/v1",
                    "model": "research-model",
                    "capabilities": ["research"],
                },
                "coding": {
                    "api_key": "coding-key",
                    "base_url": "https://coding.example/v1",
                    "model": "coding-model",
                    "capabilities": ["coding"],
                },
            }
        )
        try:
            providers = {
                name: OpenAICompatibleProvider(
                    api_key=config["api_key"],
                    base_url=config["base_url"],
                    model=config["model"],
                    name=name,
                    capabilities=config["capabilities"],
                )
                for name, config in json.loads(os.environ["SELF_DEPENDENT_PROVIDERS"]).items()
            }
            router = ModelRouter(providers)
            coordinator = CollaborativeCoordinator(router)
            plan = coordinator.plan("Ship a new feature", ["research", "coding"])
            self.assertEqual(plan[0].provider_name, "research")
            self.assertEqual(plan[1].provider_name, "coding")
        finally:
            del os.environ["SELF_DEPENDENT_PROVIDERS"]

    def test_planner_breaks_a_goal_into_capability_steps(self):
        from self_dependent_ai.collaboration import CollaborativeCoordinator, ModelRouter

        router = ModelRouter(
            {
                "research": StaticProvider("research", capabilities={"research"}),
                "coding": StaticProvider("coding", capabilities={"coding"}),
                "verification": StaticProvider("verification", capabilities={"verification"}),
            }
        )
        coordinator = CollaborativeCoordinator(router)
        plan = coordinator.plan(
            "Build a dashboard and validate it",
            ["research", "coding", "verification"],
        )

        self.assertEqual([step.capability for step in plan], ["research", "coding", "verification"])
        self.assertTrue(all(step.summary for step in plan))

    def test_goal_planner_inferrs_capabilities_from_the_goal_text(self):
        from self_dependent_ai.collaboration import GoalPlanner, ModelRouter

        router = ModelRouter(
            {
                "research": StaticProvider("research", capabilities={"research"}),
                "coding": StaticProvider("coding", capabilities={"coding"}),
                "verification": StaticProvider("verification", capabilities={"verification"}),
            }
        )
        planner = GoalPlanner(router)

        inferred = planner.infer_capabilities("Research the API, build the client, then verify tests and deployment")

        self.assertIn("research", inferred)
        self.assertIn("coding", inferred)
        self.assertIn("verification", inferred)

    def test_execution_pipeline_builds_goal_handoff_plan(self):
        from self_dependent_ai.collaboration import CollaborativeCoordinator, ModelRouter

        router = ModelRouter(
            {
                "research": StaticProvider("research", capabilities={"research"}),
                "coding": StaticProvider("coding", capabilities={"coding"}),
                "verification": StaticProvider("verification", capabilities={"verification"}),
            }
        )
        coordinator = CollaborativeCoordinator(router)

        plan = coordinator.build_plan("Research the API, code the client, and verify deployment")

        self.assertEqual([step.capability for step in plan], ["research", "coding", "verification"])
        self.assertTrue(all(step.provider_name in {"research", "coding", "verification"} for step in plan))

    def test_execution_layer_runs_each_capability_provider(self):
        from self_dependent_ai.collaboration import CollaborativeCoordinator, ModelRouter

        class RecordingProvider:
            def __init__(self, name, capability):
                self.name = name
                self.capabilities = {capability}
                self.calls = []

            def generate(self, messages):
                self.calls.append(messages)
                return f"{self.name}:{messages[-1]['content']}"

        research = RecordingProvider("research", "research")
        coding = RecordingProvider("coding", "coding")
        verification = RecordingProvider("verification", "verification")
        router = ModelRouter({"research": research, "coding": coding, "verification": verification})
        coordinator = CollaborativeCoordinator(router)

        results = coordinator.execute_goal("Research the API, code the client, and verify deployment")

        self.assertEqual([item["capability"] for item in results], ["research", "coding", "verification"])
        self.assertTrue(all(item["output"] for item in results))
        self.assertTrue(all(research.calls or coding.calls or verification.calls))

    def test_workflow_orchestrates_workspace_edit_and_test_execution(self):
        from self_dependent_ai.collaboration import CollaborativeCoordinator, ModelRouter

        class WorkflowProvider:
            def __init__(self, name, capability):
                self.name = name
                self.capabilities = {capability}

            def generate(self, messages):
                if self.name == "coding":
                    return json.dumps({"path": "notes.txt", "content": "hello from coded output\n"})
                if self.name == "verification":
                    return json.dumps({"type": "run_tests", "command": ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]})
                return "research complete"

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "tests").mkdir()
            (Path(directory) / "tests" / "__init__.py").write_text("", encoding="utf-8")
            router = ModelRouter(
                {
                    "research": WorkflowProvider("research", "research"),
                    "coding": WorkflowProvider("coding", "coding"),
                    "verification": WorkflowProvider("verification", "verification"),
                }
            )
            coordinator = CollaborativeCoordinator(router)

            result = coordinator.execute_workflow(
                "Research the API, code the client, and verify the result",
                Path(directory),
                approve=lambda _: True,
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual((Path(directory) / "notes.txt").read_text(), "hello from coded output\n")
            self.assertIn("verification", result["capabilities_used"])

    def test_cli_resume_flag_reuses_saved_session_state(self):
        import sys

        from self_dependent_ai import cli

        with mock.patch.object(cli, "AutonomousExecutionEngine") as mock_engine, \
             mock.patch.object(cli.ModelRouter, "from_environment") as mock_from_environment, \
             mock.patch.object(sys, "argv", ["self_dependent_ai.cli", "Build the feature", "--workspace", ".", "--resume"]):
            mock_from_environment.return_value = {"research": StaticProvider("research", capabilities={"research"})}
            mock_engine.return_value.resume_run.return_value = {
                "status": "completed",
                "goal": "Build the feature",
                "report": {"goal": "Build the feature", "status": "completed"},
            }

            self.assertEqual(cli.main(), 0)
            mock_engine.return_value.resume_run.assert_called_once_with("Build the feature", Path("."), approve=cli._approve)

    def test_persistent_memory_round_trip(self):
        from self_dependent_ai.collaboration import MemoryStore

        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryStore(Path(directory) / "memory.json")
            memory.set("objective", "build the dashboard")

            reloaded = MemoryStore(Path(directory) / "memory.json")
            self.assertEqual(reloaded.get("objective"), "build the dashboard")

    def test_execution_state_tracks_progress(self):
        from self_dependent_ai.collaboration import CollaborativeCoordinator, ModelRouter, ExecutionState

        class TrackingProvider:
            def __init__(self, name, capability):
                self.name = name
                self.capabilities = {capability}

            def generate(self, messages):
                return f"{self.name}:done"

        router = ModelRouter(
            {
                "research": TrackingProvider("research", "research"),
                "coding": TrackingProvider("coding", "coding"),
            }
        )
        coordinator = CollaborativeCoordinator(router)
        state = coordinator.execute_workflow(
            "Research and code the feature",
            Path(tempfile.mkdtemp()),
            approve=lambda _: True,
        )["state"]

        self.assertIsInstance(state, ExecutionState)
        self.assertEqual(state.goal, "Research and code the feature")
        self.assertEqual(state.status, "completed")
        self.assertGreaterEqual(len(state.steps), 1)

    def test_model_handoff_uses_structured_results(self):
        from self_dependent_ai.collaboration import CollaborativeCoordinator, ModelRouter

        class StructuredProvider:
            def __init__(self, name, capability):
                self.name = name
                self.capabilities = {capability}

            def generate(self, messages):
                return json.dumps({
                    "provider": self.name,
                    "capability": self.capabilities.copy().pop(),
                    "result": "structured output",
                    "summary": "completed the task",
                })

        router = ModelRouter(
            {
                "research": StructuredProvider("research", "research"),
                "coding": StructuredProvider("coding", "coding"),
            }
        )
        coordinator = CollaborativeCoordinator(router)
        results = coordinator.execute_goal("Research and code the feature")

        self.assertTrue(all(isinstance(item.get("payload"), dict) for item in results))
        self.assertTrue(all(item["payload"].get("result") for item in results))

    def test_reasoning_planner_chooses_capability_by_context_not_just_keywords(self):
        from self_dependent_ai.collaboration import CollaborativeCoordinator, ModelRouter

        class ContextProvider:
            def __init__(self, name, capability):
                self.name = name
                self.capabilities = {capability}

            def generate(self, messages):
                return "ok"

        router = ModelRouter(
            {
                "research": ContextProvider("research", "research"),
                "coding": ContextProvider("coding", "coding"),
                "verification": ContextProvider("verification", "verification"),
            }
        )
        coordinator = CollaborativeCoordinator(router)
        plan = coordinator.build_plan("I need a quick check on the API contract before implementing the client and validating it")

        self.assertIn("research", [step.capability for step in plan])
        self.assertIn("coding", [step.capability for step in plan])
        self.assertIn("verification", [step.capability for step in plan])

    def test_autonomous_execution_engine_persists_state_and_selects_tools(self):
        from self_dependent_ai.collaboration import AutonomousExecutionEngine, MemoryStore, ModelRouter, ToolRegistry

        class EngineProvider:
            def __init__(self, name, capability):
                self.name = name
                self.capabilities = {capability}

            def generate(self, messages):
                if self.name == "coding":
                    return json.dumps({"path": "artifact.txt", "content": "done\n"})
                if self.name == "verification":
                    return json.dumps({"type": "run_tests", "command": ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]})
                return "research complete"

        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryStore(Path(directory) / "session.json")
            router = ModelRouter(
                {
                    "research": EngineProvider("research", "research"),
                    "coding": EngineProvider("coding", "coding"),
                    "verification": EngineProvider("verification", "verification"),
                }
            )
            engine = AutonomousExecutionEngine(router, ToolRegistry(), memory)
            result = engine.execute("Research the API, code the result, and verify it", Path(directory), approve=lambda _: True)

            self.assertEqual(result["status"], "completed")
            self.assertTrue(memory.get(result["memory_key"]))
            self.assertEqual((Path(directory) / "artifact.txt").read_text(), "done\n")

    def test_autonomous_execution_engine_supports_resume_and_richer_tool_registry(self):
        from self_dependent_ai.collaboration import AutonomousExecutionEngine, MemoryStore, ModelRouter, ToolRegistry

        class ResumeProvider:
            def __init__(self, name, capability):
                self.name = name
                self.capabilities = {capability}

            def generate(self, messages):
                if self.name == "coding":
                    return json.dumps({"path": "resume.txt", "content": "resumed\n"})
                return "ok"

        with tempfile.TemporaryDirectory() as directory:
            router = ModelRouter(
                {
                    "research": ResumeProvider("research", "research"),
                    "coding": ResumeProvider("coding", "coding"),
                    "verification": ResumeProvider("verification", "verification"),
                }
            )
            engine = AutonomousExecutionEngine(router, ToolRegistry(), MemoryStore(Path(directory) / "resume.json"))

            result = engine.execute("Find the right file and patch it", Path(directory), approve=lambda _: True)
            resumed = engine.resume_session(result["memory_key"])

            self.assertIn("search_files", engine.tool_registry.available())
            self.assertEqual(resumed["goal"], "Find the right file and patch it")
            self.assertEqual((Path(directory) / "resume.txt").read_text(), "resumed\n")
            self.assertIn("history", resumed)
            self.assertIn("pending_steps", resumed)

    def test_execution_engine_executes_tools_and_retries_failed_steps(self):
        from self_dependent_ai.collaboration import AutonomousExecutionEngine, MemoryStore, ModelRouter, ToolRegistry

        router = ModelRouter({"research": StaticProvider("research", capabilities={"research"})})
        engine = AutonomousExecutionEngine(router, ToolRegistry(), MemoryStore(Path(tempfile.mkdtemp()) / "tool-state.json"))

        with tempfile.TemporaryDirectory() as directory:
            safe_file = Path(directory) / "safe.txt"
            safe_file.write_text("hello\n", encoding="utf-8")

            success = engine.execute_tool_call("read_file", {"path": "safe.txt"}, Path(directory))
            self.assertEqual(success["status"], "completed")
            self.assertEqual(success["content"], "hello\n")

            failed = engine.execute_tool_call("read_file", {"path": "../outside.txt"}, Path(directory))
            self.assertEqual(failed["status"], "failed")
            self.assertIn("outside the workspace", str(failed["error"]).lower())

            recovered = engine.recover_from_failure("read_file", {"path": "../outside.txt"}, ValueError("outside the workspace"), Path(directory))
            self.assertEqual(recovered["status"], "recovered")
            self.assertIn("retry", recovered["action"].lower())

    def test_execution_engine_tracks_step_attempts_and_recovers_in_order(self):
        from self_dependent_ai.collaboration import AutonomousExecutionEngine, MemoryStore, ModelRouter, ToolRegistry

        router = ModelRouter({"research": StaticProvider("research", capabilities={"research"})})
        with tempfile.TemporaryDirectory() as directory:
            engine = AutonomousExecutionEngine(router, ToolRegistry(), MemoryStore(Path(directory) / "retry.json"))
            target = Path(directory) / "payload.txt"
            target.write_text("alpha\n", encoding="utf-8")

            result = engine.execute_step(
                "read_file",
                {"path": "payload.txt"},
                Path(directory),
                goal="Inspect and recover a readable file",
                capability="research",
                max_retries=2,
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["content"], "alpha\n")
            self.assertEqual(result["attempts"], 1)
            self.assertEqual(result["recoveries"], 0)

            fail_result = engine.execute_step(
                "read_file",
                {"path": "../blocked.txt"},
                Path(directory),
                goal="Inspect and recover a readable file",
                capability="research",
                max_retries=2,
            )

            self.assertEqual(fail_result["status"], "recovered")
            self.assertEqual(fail_result["recoveries"], 1)
            self.assertIn("fallback", fail_result["action"].lower())

    def test_execution_engine_runs_a_dependency_ordered_workflow_loop_with_retries(self):
        from self_dependent_ai.collaboration import AutonomousExecutionEngine, MemoryStore, ModelRouter, ToolRegistry

        class WorkflowProvider:
            def __init__(self, name, capability):
                self.name = name
                self.capabilities = {capability}
                self.calls = 0

            def generate(self, messages):
                self.calls += 1
                if self.name == "research":
                    return json.dumps({"tool": "search_files", "params": {"query": "alpha"}})
                if self.name == "coding":
                    return json.dumps({"tool": "write_file", "params": {"path": "alpha.txt", "content": "alpha\n"}})
                return json.dumps({"tool": "run_tests", "params": {"command": ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]}})

        with tempfile.TemporaryDirectory() as directory:
            router = ModelRouter(
                {
                    "research": WorkflowProvider("research", "research"),
                    "coding": WorkflowProvider("coding", "coding"),
                    "verification": WorkflowProvider("verification", "verification"),
                }
            )
            engine = AutonomousExecutionEngine(router, ToolRegistry(), MemoryStore(Path(directory) / "workflow.json"))

            result = engine.execute_workflow_loop(
                "Research the alpha data, create the file, and verify the result",
                Path(directory),
                approve=lambda _: True,
                max_retries=2,
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual([step["capability"] for step in result["steps"]], ["research", "coding", "verification"])
            self.assertEqual((Path(directory) / "alpha.txt").read_text(), "alpha\n")
            self.assertTrue(all(step["status"] in {"completed", "recovered"} for step in result["steps"]))

    def test_model_contracts_validate_provider_handoffs(self):
        from self_dependent_ai.collaboration import CollaborativeCoordinator, ModelContract, ModelRouter, validate_model_contract

        class ContractProvider:
            def __init__(self, name, capability):
                self.name = name
                self.capabilities = {capability}

            def generate(self, messages):
                return json.dumps(
                    {
                        "provider": self.name,
                        "capability": self.capabilities.copy().pop(),
                        "status": "ready",
                        "summary": "research complete",
                        "tool": "search_files",
                        "params": {"query": "alpha"},
                        "decision": "continue",
                    }
                )

        router = ModelRouter({"research": ContractProvider("research", "research")})
        coordinator = CollaborativeCoordinator(router)
        contract = ModelContract(
            provider="research",
            capability="research",
            status="ready",
            summary="research complete",
            tool="search_files",
            params={"query": "alpha"},
            decision="continue",
        )

        validated = validate_model_contract(contract.to_dict())
        self.assertTrue(validated["valid"])
        self.assertEqual(validated["contract"]["provider"], "research")

        raw = coordinator.parse_contract(
            json.dumps({
                "provider": "research",
                "capability": "research",
                "status": "ready",
                "summary": "research complete",
                "tool": "search_files",
                "params": {"query": "alpha"},
                "decision": "continue",
            })
        )
        self.assertEqual(raw["contract"]["status"], "ready")

    def test_model_negotiation_ranks_provider_decisions(self):
        from self_dependent_ai.collaboration import CollaborativeCoordinator, ModelRouter

        class NegotiationProvider:
            def __init__(self, name, capability, decision, summary):
                self.name = name
                self.capabilities = {capability}
                self._decision = decision
                self._summary = summary

            def generate(self, messages):
                return json.dumps(
                    {
                        "provider": self.name,
                        "capability": "research",
                        "status": "ready",
                        "summary": self._summary,
                        "tool": "search_files",
                        "params": {"query": "alpha"},
                        "decision": self._decision,
                    }
                )

        router = ModelRouter(
            {
                "research_a": NegotiationProvider("research_a", "research", "continue", "broad research plan"),
                "research_b": NegotiationProvider("research_b", "research", "continue", "targeted research plan"),
            }
        )
        coordinator = CollaborativeCoordinator(router)

        result = coordinator.negotiate(
            "Research the alpha API and reduce the uncertainty before coding",
            ["research"],
            strategy="best_summary",
        )

        self.assertEqual(result["winner"]["provider"], "research_b")
        self.assertEqual(result["winner"]["decision"], "continue")
        self.assertEqual(result["status"], "negotiated")
        self.assertGreaterEqual(len(result["candidates"]), 2)

    def test_execution_workflow_uses_negotiated_winner_for_each_capability(self):
        from self_dependent_ai.collaboration import CollaborativeCoordinator, ModelRouter

        class NegotiationProvider:
            def __init__(self, name, capability, summary):
                self.name = name
                self.capabilities = {capability}
                self._summary = summary

            def generate(self, messages):
                return json.dumps(
                    {
                        "provider": self.name,
                        "capability": self.capabilities.copy().pop(),
                        "status": "ready",
                        "summary": self._summary,
                        "tool": "search_files",
                        "params": {"query": "alpha"},
                        "decision": "continue",
                    }
                )

        router = ModelRouter(
            {
                "research_a": NegotiationProvider("research_a", "research", "broad research plan"),
                "research_b": NegotiationProvider("research_b", "research", "targeted research plan"),
            }
        )
        coordinator = CollaborativeCoordinator(router)

        plan = coordinator.plan("Research the alpha API before coding", ["research"])
        self.assertEqual(plan[0].provider_name, "research_a")

        workspace = Path(".tmp-negotiation-workspace")
        workspace.mkdir(exist_ok=True)

        result = coordinator.execute_workflow(
            "Research the alpha API before coding",
            workspace,
            approve=lambda summary: True,
        )

        self.assertTrue(result["steps"])
        self.assertEqual(result["steps"][0]["provider_name"], "research_b")

    def test_demo_provider_config_file_loads_automatically(self):
        from self_dependent_ai.collaboration import ModelRouter

        payload = {
            "research_demo": {
                "demo": True,
                "capabilities": ["research"],
            },
            "review_demo": {
                "demo": True,
                "capabilities": ["review"],
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "demo_providers.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch("pathlib.Path.cwd", return_value=Path(directory)):
                router = ModelRouter.from_environment()

            self.assertIn("research_demo", router.providers)
            self.assertIn("research", router.providers["research_demo"].capabilities)
            self.assertIn("review", router.providers["review_demo"].capabilities)
            self.assertIn('"provider": "research_demo"', router.providers["research_demo"].generate([{"role": "user", "content": "Goal"}]))

    def test_cli_collaborative_review_mode_validates_workflow_output(self):
        import io
        import sys
        from contextlib import redirect_stdout

        from self_dependent_ai import cli

        class ResearchProvider:
            def __init__(self, name):
                self.name = name
                self.capabilities = {"research"}

            def generate(self, messages):
                return json.dumps(
                    {
                        "provider": self.name,
                        "capability": "research",
                        "status": "ready",
                        "summary": "targeted research plan",
                        "tool": "search_files",
                        "params": {"query": "alpha"},
                        "decision": "continue",
                    }
                )

        class ReviewProvider:
            def __init__(self, name):
                self.name = name
                self.capabilities = {"review"}

            def generate(self, messages):
                return json.dumps(
                    {
                        "provider": self.name,
                        "capability": "review",
                        "status": "ready",
                        "summary": "reviewed and approved the provider output",
                        "tool": "read_file",
                        "params": {"path": "README.md"},
                        "decision": "accept",
                    }
                )

        router = {
            "research_a": ResearchProvider("research_a"),
            "review_a": ReviewProvider("review_a"),
        }

        original_argv = sys.argv[:]
        try:
            sys.argv = ["self-dependent-ai", "Research the alpha API", "--workspace", ".", "--collaborative"]
            stdout = io.StringIO()
            with mock.patch.object(cli.ModelRouter, "from_environment", return_value=cli.ModelRouter(router)), \
                 mock.patch("builtins.input", return_value="y"), \
                 redirect_stdout(stdout):
                exit_code = cli.main()
            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertIn("Review decision", output)
            self.assertIn("accept", output.casefold())
        finally:
            sys.argv = original_argv

    def test_cli_autonomous_loop_reports_negotiation_and_execution(self):
        import sys

        from self_dependent_ai import cli

        class Provider:
            def __init__(self, name, capability, summary):
                self.name = name
                self.capabilities = {capability}
                self._summary = summary

            def generate(self, messages):
                return json.dumps(
                    {
                        "provider": self.name,
                        "capability": self.capabilities.copy().pop(),
                        "status": "ready",
                        "summary": self._summary,
                        "tool": "search_files",
                        "params": {"query": "alpha"},
                        "decision": "continue",
                    }
                )

        router = {
            "research_a": Provider("research_a", "research", "broad research plan"),
            "research_b": Provider("research_b", "research", "targeted research plan"),
        }

        original_argv = sys.argv[:]
        try:
            sys.argv = ["self-dependent-ai", "Research the alpha API", "--workspace", ".", "--autonomous"]
            with mock.patch.object(cli.ModelRouter, "from_environment", return_value=cli.ModelRouter(router)), \
                 mock.patch("builtins.input", return_value="y"):
                exit_code = cli.main()
            self.assertEqual(exit_code, 0)
        finally:
            sys.argv = original_argv

    def test_run_history_persistence_and_resume_support(self):
        from self_dependent_ai.collaboration import AutonomousExecutionEngine, ModelRouter

        class Provider:
            def __init__(self, name, capability, summary):
                self.name = name
                self.capabilities = {capability}
                self._summary = summary

            def generate(self, messages):
                return json.dumps(
                    {
                        "provider": self.name,
                        "capability": self.capabilities.copy().pop(),
                        "status": "ready",
                        "summary": self._summary,
                        "tool": "search_files",
                        "params": {"query": "alpha"},
                        "decision": "continue",
                    }
                )

        router = ModelRouter(
            {
                "research_a": Provider("research_a", "research", "broad research plan"),
                "research_b": Provider("research_b", "research", "targeted research plan"),
            }
        )
        engine = AutonomousExecutionEngine(router)

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            result = engine.execute("Research the alpha API before coding", workspace, approve=lambda _: True)

            self.assertIn("report", result)
            self.assertEqual(result["report"]["winner"]["provider"], "research_b")
            self.assertIn("scores", result["report"])
            self.assertIn("outputs", result["report"])

            session_key = engine._session_key("Research the alpha API before coding", workspace)
            persisted = engine.memory_store.get(session_key)
            self.assertIn("history", persisted)
            self.assertIn("report", persisted)

            resumed = engine.resume_run("Research the alpha API before coding", workspace)
            self.assertEqual(resumed["status"], "completed")

    def test_provider_history_biases_negotiation_and_persists_confidence(self):
        from self_dependent_ai.collaboration import CollaborativeCoordinator, MemoryStore, ModelRouter

        class Provider:
            def __init__(self, name, capability, summary):
                self.name = name
                self.capabilities = {capability}
                self._summary = summary

            def generate(self, messages):
                return json.dumps(
                    {
                        "provider": self.name,
                        "capability": self.capabilities.copy().pop(),
                        "status": "ready",
                        "summary": self._summary,
                        "tool": "search_files",
                        "params": {"query": "alpha"},
                        "decision": "continue",
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryStore(Path(directory) / "memory.json")
            router = ModelRouter(
                {
                    "research_a": Provider("research_a", "research", "broad research plan"),
                    "research_b": Provider("research_b", "research", "targeted research plan"),
                }
            )
            coordinator = CollaborativeCoordinator(router, memory_store=memory)
            coordinator.record_provider_outcome("research_a", "research", success=False, quality=0.1)
            coordinator.record_provider_outcome("research_a", "research", success=True, quality=0.2)
            coordinator.record_provider_outcome("research_b", "research", success=True, quality=0.9)
            coordinator.record_provider_outcome("research_b", "research", success=True, quality=0.95)

            result = coordinator.negotiate(
                "Research the alpha API and reduce the uncertainty before coding",
                ["research"],
                strategy="best_summary",
            )

            self.assertEqual(result["winner"]["provider"], "research_b")
            self.assertGreater(result["providers"]["research_b"]["confidence"], result["providers"]["research_a"]["confidence"])
            self.assertIn("research_b", memory.get("provider_history", {}))

    def test_model_to_model_handoff_validates_reviewer_output(self):
        from self_dependent_ai.collaboration import CollaborativeCoordinator, ModelRouter

        class ResearchProvider:
            def __init__(self, name):
                self.name = name
                self.capabilities = {"research"}

            def generate(self, messages):
                return json.dumps(
                    {
                        "provider": self.name,
                        "capability": "research",
                        "status": "ready",
                        "summary": "targeted research plan",
                        "tool": "search_files",
                        "params": {"query": "alpha"},
                        "decision": "continue",
                    }
                )

        class ReviewProvider:
            def __init__(self, name):
                self.name = name
                self.capabilities = {"review"}

            def generate(self, messages):
                return json.dumps(
                    {
                        "provider": self.name,
                        "capability": "review",
                        "status": "ready",
                        "summary": "reviewed and accepted the research outcome",
                        "tool": "read_file",
                        "params": {"path": "README.md"},
                        "decision": "accept",
                    }
                )

        router = ModelRouter({
            "research_a": ResearchProvider("research_a"),
            "review_a": ReviewProvider("review_a"),
        })
        coordinator = CollaborativeCoordinator(router)

        handoff = coordinator.hand_off(
            from_provider="research_a",
            to_provider="review_a",
            capability="research",
            request="Research the alpha API before coding",
            result={
                "provider": "research_a",
                "capability": "research",
                "status": "ready",
                "summary": "targeted research plan",
                "tool": "search_files",
                "params": {"query": "alpha"},
                "decision": "continue",
            },
        )

        self.assertEqual(handoff["status"], "validated")
        self.assertEqual(handoff["reviewer"], "review_a")
        self.assertEqual(handoff["decision"], "accept")


if __name__ == "__main__":
    unittest.main()
