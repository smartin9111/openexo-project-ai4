from src.common.types import PerceptionState
from src.perception.base import BasePerception


class DummyPerception(BasePerception):

    def process(self, observation) -> PerceptionState:
        return PerceptionState(
            joint_positions=observation["joint_positions"],
            joint_velocities=observation["joint_velocities"],
            muscle_activations=observation["muscle_activations"],
            foot_contacts=observation["foot_contacts"],
            gait_phase="stance",
        )