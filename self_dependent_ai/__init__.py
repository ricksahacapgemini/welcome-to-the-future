"""A bounded, approval-based coding-agent prototype."""

from self_dependent_ai.collaboration import (
    AutonomousExecutionEngine,
    CollaborativeCoordinator,
    ExecutionState,
    GoalPlanner,
    MemoryStore,
    ModelContract,
    ModelRouter,
    RunReport,
    ToolRegistry,
    validate_model_contract,
)

__version__ = "0.1.0"

__all__ = [
    "AutonomousExecutionEngine",
    "CollaborativeCoordinator",
    "ExecutionState",
    "GoalPlanner",
    "MemoryStore",
    "ModelContract",
    "ModelRouter",
    "RunReport",
    "ToolRegistry",
    "validate_model_contract",
]
