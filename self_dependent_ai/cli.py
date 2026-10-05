"""Command-line entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from self_dependent_ai.agent import Agent
from self_dependent_ai.collaboration import AutonomousExecutionEngine, CollaborativeCoordinator, ModelRouter
from self_dependent_ai.provider import OpenAICompatibleProvider


def _approve(summary: str) -> bool:
    try:
        answer = input(f"\nApproval required: {summary}. Continue? [y/N] ")
    except EOFError:
        return False
    return answer.strip().casefold() in {"y", "yes"}


def _print_plan(goal: str) -> None:
    try:
        providers = ModelRouter.from_environment()
    except ValueError as error:
        raise RuntimeError(str(error)) from error

    coordinator = CollaborativeCoordinator(providers)
    plan = coordinator.build_plan(goal)
    print(f"Goal: {goal}")
    print("Execution plan:")
    for index, step in enumerate(plan, start=1):
        print(f"  {index}. [{step.capability}] {step.provider_name} - {step.summary}")


def _print_autonomous_report(goal: str, plan: list, negotiation: dict, execution: dict) -> None:
    print(f"Goal: {goal}")
    print("Provider plan:")
    for index, step in enumerate(plan, start=1):
        print(f"  {index}. [{step.capability}] {step.provider_name} - {step.summary}")

    print("Negotiated ranking:")
    for index, candidate in enumerate(negotiation.get("candidates", []), start=1):
        print(
            f"  {index}. [{candidate.get('capability')}] {candidate.get('provider')} "
            f"score={candidate.get('score', 0.0):.2f} decision={candidate.get('decision')} status={candidate.get('status')}"
        )
    print(f"Winner: {negotiation.get('winner', {}).get('provider')} -> {negotiation.get('winner', {}).get('decision')}")

    print("Execution results:")
    for index, step in enumerate(execution.get("steps", []), start=1):
        print(f"  {index}. [{step['capability']}] {step['provider_name']}")
        print(f"     {step.get('output', '')}")


def _print_collaborative_report(goal: str, plan: list, negotiation: dict, execution: dict, reviews: list[dict]) -> None:
    _print_autonomous_report(goal, plan, negotiation, execution)
    print("Review decisions:")
    for index, review in enumerate(reviews, start=1):
        print(
            f"  {index}. [{review.get('capability')}] {review.get('reviewer')} "
            f"decision={review.get('decision')} status={review.get('status')}"
        )


def _run_collaborative_loop(goal: str, workspace: Path) -> int:
    try:
        providers = ModelRouter.from_environment()
    except ValueError as error:
        raise RuntimeError(str(error)) from error

    coordinator = CollaborativeCoordinator(providers)
    plan = coordinator.build_plan(goal)
    negotiation = coordinator.negotiate(goal, [step.capability for step in plan], strategy="best_summary")
    if not negotiation.get("candidates"):
        raise RuntimeError("No model candidates were available to negotiate a valid execution plan.")

    execution = coordinator.execute_workflow(goal, workspace, approve=_approve)
    review_candidates = [
        name
        for name, provider in providers.providers.items()
        if any(capability in {"review", "verification"} for capability in getattr(provider, "capabilities", set()))
    ]
    reviewer_name = review_candidates[0] if review_candidates else negotiation.get("winner", {}).get("provider")
    reviews: list[dict] = []

    for step in execution.get("steps", []):
        source_provider = step.get("provider_name")
        result = step.get("payload") or {
            "provider": source_provider,
            "capability": step.get("capability"),
            "status": "ready",
            "summary": step.get("summary", ""),
            "tool": "unknown",
            "params": {},
            "decision": "continue",
        }
        if reviewer_name and reviewer_name != source_provider:
            review = coordinator.hand_off(
                from_provider=source_provider,
                to_provider=reviewer_name,
                capability=step.get("capability", "verification"),
                request=goal,
                result=result,
            )
        else:
            review = {
                "status": "validated",
                "reviewer": source_provider,
                "from_provider": source_provider,
                "capability": step.get("capability", "verification"),
                "request": goal,
                "decision": "accept",
                "result": result,
            }
        reviews.append(review)

    _print_collaborative_report(goal, plan, negotiation, execution, reviews)
    final_status = execution.get("status") == "completed" and all(review.get("status") != "rejected" for review in reviews)
    return 0 if final_status else 1


def _run_autonomous_loop(goal: str, workspace: Path) -> int:
    try:
        providers = ModelRouter.from_environment()
    except ValueError as error:
        raise RuntimeError(str(error)) from error

    coordinator = CollaborativeCoordinator(providers)
    plan = coordinator.build_plan(goal)
    negotiation = coordinator.negotiate(goal, [step.capability for step in plan], strategy="best_summary")

    if not negotiation.get("candidates"):
        raise RuntimeError("No model candidates were available to negotiate a valid execution plan.")

    execution = coordinator.execute_workflow(goal, workspace, approve=_approve)
    _print_autonomous_report(goal, plan, negotiation, execution)
    return 0 if execution.get("status") == "completed" else 1


def _run_execution(goal: str, workspace: Path) -> None:
    try:
        providers = ModelRouter.from_environment()
    except ValueError as error:
        raise RuntimeError(str(error)) from error

    coordinator = CollaborativeCoordinator(providers)
    result = coordinator.execute_workflow(goal, workspace, approve=_approve)
    print(f"Goal: {goal}")
    print("Execution results:")
    for index, step in enumerate(result["steps"], start=1):
        print(f"  {index}. [{step['capability']}] {step['provider_name']}")
        print(f"     {step['output']}")


def _resume_run(goal: str, workspace: Path) -> int:
    try:
        providers = ModelRouter.from_environment()
    except ValueError as error:
        raise RuntimeError(str(error)) from error

    engine = AutonomousExecutionEngine(providers)
    result = engine.resume_run(goal, workspace, approve=_approve)
    report = result.get("report") or {}

    print(f"Goal: {goal}")
    print(f"Status: {result.get('status', 'unknown')}")
    if report:
        print(f"Summary: {report.get('summary', '')}")
        if report.get("winner"):
            print(f"Winner: {report['winner'].get('provider')} -> {report['winner'].get('decision')}")
    if result.get("history"):
        print("History:")
        for index, step in enumerate(result["history"], start=1):
            print(f"  {index}. [{step.get('capability')}] {step.get('provider_name')} -> {step.get('status')}")

    return 0 if result.get("status") == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded coding agent in a workspace.")
    parser.add_argument("goal", help="The coding task to accomplish")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Project directory the agent may inspect and edit (default: current directory)",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Infer capabilities and print the provider handoff plan without executing the workspace agent.",
    )
    parser.add_argument(
        "--execute-plan",
        action="store_true",
        help="Infer capabilities, route to configured providers, and execute each model-backed task in sequence.",
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Negotiate among provider candidates, ask for explicit approval, and report the full autonomous run.",
    )
    parser.add_argument(
        "--collaborative",
        action="store_true",
        help="Run the autonomous loop and validate each provider output through a second reviewer model before closing the workflow.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the saved autonomous session for this goal and workspace, using persisted step history and final report data.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Load the bundled demo provider config when present, without requiring a pre-set SELF_DEPENDENT_PROVIDERS environment variable.",
    )
    arguments = parser.parse_args()

    if arguments.demo:
        demo_path = None
        for candidate in (
            Path.cwd() / "demo_providers.json",
            Path(__file__).resolve().parent.parent / "demo_providers.json",
        ):
            if candidate.exists():
                demo_path = candidate
                break
        if demo_path is not None:
            with demo_path.open("r", encoding="utf-8") as handle:
                os.environ["SELF_DEPENDENT_PROVIDERS"] = handle.read()
        else:
            raise RuntimeError("No demo provider config file was found.")

    if arguments.resume:
        try:
            return _resume_run(arguments.goal, arguments.workspace)
        except (KeyError, RuntimeError, ValueError) as error:
            parser.error(str(error))

    if arguments.plan:
        try:
            _print_plan(arguments.goal)
            return 0
        except RuntimeError as error:
            parser.error(str(error))

    if arguments.execute_plan or arguments.autonomous:
        try:
            return _run_autonomous_loop(arguments.goal, arguments.workspace)
        except RuntimeError as error:
            parser.error(str(error))

    if arguments.collaborative:
        try:
            return _run_collaborative_loop(arguments.goal, arguments.workspace)
        except RuntimeError as error:
            parser.error(str(error))

    try:
        provider = OpenAICompatibleProvider.from_environment()
        agent = Agent(provider, arguments.workspace)
    except ValueError as error:
        parser.error(str(error))

    print(f"Workspace: {agent.workspace}")
    result = agent.run(arguments.goal, approve=_approve)
    print(f"\n{result.status.upper()} after {result.steps} step(s): {result.message}")
    return 0 if result.status == "completed" else 1
