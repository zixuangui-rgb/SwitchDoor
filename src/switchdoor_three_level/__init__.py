"""Three-level SwitchDoor environment and deterministic data generator."""

from .config import load_config
from .generator import audit_config, generate_dataset, validate_dataset
from .live_env import LiveEnvError, PublicObservation, StepResult, ThreeLevelSwitchDoorEnv

__all__ = [
    "LiveEnvError",
    "PublicObservation",
    "StepResult",
    "ThreeLevelSwitchDoorEnv",
    "audit_config",
    "generate_dataset",
    "load_config",
    "validate_dataset",
]

__version__ = "0.2.0"
