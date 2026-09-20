from dataclasses import dataclass
from typing import Any


@dataclass
class PerceptionState:
    """
    Standardized output produced by a perception module
    and consumed by a controller.
    """

    joint_positions: Any = None
    joint_velocities: Any = None
    muscle_activations: Any = None
    foot_contacts: Any = None
    gait_phase: Any = None