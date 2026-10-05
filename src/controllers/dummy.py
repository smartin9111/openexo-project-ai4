from src.common.types import PerceptionState
from src.controllers.base import BaseController


class DummyController(BaseController):

    def compute_action(self, state: PerceptionState):
        # OpenExo_L1 ankle exo command range: [-1, 0]
        if state.gait_phase == "stance":
            return -1.0

        return 0.0