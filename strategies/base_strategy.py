from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    @abstractmethod
    def evaluate(self, df):
        """Evaluate and return (signal, confidence, trend, score)."""
        pass
