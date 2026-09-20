from abc import ABC, abstractmethod

from src.common.types import PerceptionState


class BaseController(ABC):

    @abstractmethod
    def compute_action(self, state: PerceptionState):
        """
        Convert a PerceptionState into an action
        for the exoskeleton.
        """
        pass