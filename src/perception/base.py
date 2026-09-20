from abc import ABC, abstractmethod

from src.common.types import PerceptionState


class BasePerception(ABC):

    @abstractmethod
    def process(self, observation) -> PerceptionState:
        """
        Convert a raw environment observation
        into a standardized PerceptionState.
        """
        pass