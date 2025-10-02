"""
Интерфейсы и классы данных для системы обработки диалогов.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Message:
    """Сообщение в диалоге."""
    content: str
    role: str  # 'user' или 'assistant'
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class Chunk:
    """Обработанный чанк текста."""
    content: str
    original_user: str
    role: str  # 'user' или 'assistant'
    session_id: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProcessedData:
    """Результат обработки данных."""
    chunks: List[Chunk]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Результат поиска."""
    chunk: Chunk
    score: float
    final_score: float


class IChunkProcessor(ABC):
    """Интерфейс для обработки чанков."""
    
    @abstractmethod
    def process_messages_batch(self, messages_list: List[List[Message]], 
                              dialogue_ids: List[str]) -> List[ProcessedData]:
        """
        Батчевая обработка сообщений.
        
        Args:
            messages_list: Список списков сообщений для каждого диалога
            dialogue_ids: Список идентификаторов диалогов
            
        Returns:
            Список обработанных данных
        """
        pass
    
    @abstractmethod
    def create_context_chunks(self, messages: List[Message]) -> List[Chunk]:
        """
        Создание чанков с контекстом.
        
        Args:
            messages: Список сообщений
            
        Returns:
            Список чанков
        """
        pass
    
    @abstractmethod
    def deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Дедупликация чанков.
        
        Args:
            chunks: Список чанков для дедупликации
            
        Returns:
            Список уникальных чанков
        """
        pass
    
    @abstractmethod
    def filter_assistant(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Фильтрация реплик ассистента.
        
        Args:
            chunks: Список чанков для фильтрации
            
        Returns:
            Отфильтрованный список чанков
        """
        pass


class IAnswerGenerator(ABC):
    """Интерфейс для генерации ответов."""
    
    @abstractmethod
    def generate_answer(self, question: str,
                       search_results: List[SearchResult],
                       dialogue_id: str) -> str:
        """
        Генерация ответа на вопрос.
        
        Args:
            question: Вопрос пользователя
            search_results: Результаты поиска по диалогу
            dialogue_id: Идентификатор диалога
            
        Returns:
            Сгенерированный ответ
        """
        pass


# Константы
NO_INFO_THRESHOLD = 0.3
