from abc import ABC, abstractmethod
from typing import List

from submit.interfaces import Message

# Алиас для совместимости - будет определен позже
ModelWithMemory = None


class ModelWithMemoryBase(ABC):

    @abstractmethod
    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        pass

    @abstractmethod
    def clear_memory(self, dialogue_id: str) -> None:
        pass

    @abstractmethod
    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        pass