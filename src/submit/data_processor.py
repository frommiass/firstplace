"""
Препроцессор данных - БЕЗ ВЕСОВ!

Функции:
- Чанкинг с контекстом (user + assistant)
- Дедупликация через хэши
- Фильтрация ассистента
- Метаданные для temporal ordering
"""

from collections import defaultdict
from typing import List, Set
from multiprocessing.pool import ThreadPool
import re

from .interfaces import IChunkProcessor, Message, Chunk, ProcessedData


class DataProcessor(IChunkProcessor):
    """
    Препроцессор диалогов.
    """
    
    def __init__(self):
        # Хэши для дедупликации
        self.seen_hashes: Set[int] = set()
        
        # Счетчики индексов (для temporal ordering)
        self.global_indices: defaultdict[str, int] = defaultdict(int)
        
        # Минимальный набор стоп-слов (для детекции фактов)
        self.stopwords = self._get_stopwords()
    
    def _get_stopwords(self) -> Set[str]:
        """Базовый набор стоп-слов"""
        return {
            'и', 'в', 'не', 'что', 'он', 'на', 'я', 'с', 'как', 'а', 'то',
            'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
            'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот',
            'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'когда', 'даже'
        }
    
    def process_messages_batch(self, messages_list: List[List[Message]], 
                              dialogue_ids: List[str]) -> List[ProcessedData]:
        """
        Батчевая обработка сообщений.
        
        Использует ThreadPool для параллелизации (16 CPU cores).
        """
        # Задачи для параллельной обработки
        tasks = list(zip(messages_list, dialogue_ids))
        
        # Параллельная обработка
        with ThreadPool(processes=16) as pool:
            results = pool.starmap(self._process_single, tasks)
        
        return results
    
    def _process_single(self, messages: List[Message], dialogue_id: str) -> ProcessedData:
        """
        Обработка одного батча сообщений.
        """
        # 1. Создание чанков
        chunks = self.create_context_chunks(messages)
        
        # 2. Дедупликация
        chunks = self.deduplicate(chunks)
        
        # 3. Фильтрация ассистента
        chunks = self.filter_assistant(chunks)
        
        # 4. Метаданные
        chunks = self._add_metadata(chunks, dialogue_id)
        
        return ProcessedData(chunks=chunks)
    
    def create_context_chunks(self, messages: List[Message]) -> List[Chunk]:
        """
        Создание чанков с контекстом.
        
        ВАЖНО: User message + assistant подтверждение
        
        Пример:
        User: "Меня зовут Иван"
        Assistant: "Приятно познакомиться, Иван!"
        → Chunk: "Меня зовут Иван [Подтверждение: Приятно познакомиться, Иван!]"
        """
        chunks = []
        
        for i, msg in enumerate(messages):
            if msg.role == 'user':
                # Оригинальный текст
                user_text = msg.content.strip()
                
                # Ищем следующее сообщение ассистента
                assistant_confirmation = None
                if i + 1 < len(messages) and messages[i + 1].role == 'assistant':
                    assistant_confirmation = messages[i + 1].content.strip()
                
                # Создаем полный контекст
                if assistant_confirmation and len(assistant_confirmation) < 200:
                    # Добавляем подтверждение только если оно короткое
                    full_content = f"{user_text} [Подтверждение: {assistant_confirmation}]"
                else:
                    full_content = user_text
                
                chunk = Chunk(
                    content=full_content,
                    original_user=user_text,
                    role='user',
                    session_id=msg.session_id or 'unknown'
                )
                chunks.append(chunk)
                
            elif msg.role == 'assistant':
                # Реплики ассистента тоже сохраняем (потом отфильтруем)
                chunk = Chunk(
                    content=msg.content.strip(),
                    original_user='',
                    role='assistant',
                    session_id=msg.session_id or 'unknown'
                )
                chunks.append(chunk)
        
        return chunks
    
    def deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Дедупликация через хэши.
        
        Простой и быстрый метод для точных дубликатов.
        """
        unique_chunks = []
        
        for chunk in chunks:
            # Нормализация текста
            normalized = chunk.content.lower().strip()
            normalized = re.sub(r'\s+', ' ', normalized)
            
            chunk_hash = hash(normalized)
            
            if chunk_hash not in self.seen_hashes:
                self.seen_hashes.add(chunk_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def filter_assistant(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Фильтрация реплик ассистента.
        
        Оставляем только если содержит факт о пользователе.
        Убираем:
        - Общие фразы ("Как интересно!")
        - Длинные объяснения
        - Переводы/пересказы
        """
        filtered = []
        
        for chunk in chunks:
            if chunk.role == 'user':
                # User messages всегда оставляем
                filtered.append(chunk)
            elif chunk.role == 'assistant':
                # Проверяем стоит ли оставлять
                if self._contains_user_fact(chunk.content):
                    filtered.append(chunk)
        
        return filtered
    
    def _contains_user_fact(self, text: str) -> bool:
        """
        Проверка содержит ли текст ассистента факт о пользователе.
        
        Эвристики:
        1. Короткий текст (< 100 символов) + конкретика
        2. Ключевые слова подтверждения + сущности
        """
        text_lower = text.lower()
        
        # Слишком длинный - скорее всего не подтверждение
        if len(text) > 200:
            return False
        
        # Общие фразы без конкретики
        generic_phrases = [
            'как интересно', 'понял', 'хорошо', 'отлично', 'замечательно',
            'понятно', 'ясно', 'согласен', 'да', 'конечно', 'могу помочь'
        ]
        
        # Если только общие фразы - не факт
        if any(phrase in text_lower for phrase in generic_phrases) and len(text) < 50:
            words = [w for w in text_lower.split() if w not in self.stopwords]
            if len(words) < 3:
                return False
        
        # Ключевые слова подтверждения фактов
        confirmation_keywords = [
            'зовут', 'имя', 'лет', 'возраст', 'живешь', 'работа',
            'кот', 'собака', 'питомец', 'жена', 'муж', 'дети'
        ]
        
        # Если есть подтверждение + достаточно слов
        if any(kw in text_lower for kw in confirmation_keywords):
            words = [w for w in text_lower.split() if w not in self.stopwords]
            if len(words) >= 3:
                return True
        
        # Проверка на имена собственные (большая буква в середине)
        words = text.split()
        has_proper_noun = any(
            word[0].isupper() and i > 0
            for i, word in enumerate(words)
            if len(word) > 2
        )
        
        if has_proper_noun and len(text) < 100:
            return True
        
        return False
    
    def _add_metadata(self, chunks: List[Chunk], dialogue_id: str) -> List[Chunk]:
        """
        Добавление метаданных.
        
        КРИТИЧНО для temporal ordering (INFO_UPDATING)!
        """
        for i, chunk in enumerate(chunks):
            # Глобальный индекс (увеличивается с каждым вызовом)
            global_idx = self.global_indices[dialogue_id]
            self.global_indices[dialogue_id] += 1
            
            chunk.metadata = {
                'index': global_idx,  # Для temporal ordering
                'local_index': i,     # Индекс в батче
                'is_user': chunk.role == 'user',
                'has_fact': self._estimate_fact_presence(chunk),
                'session_id': chunk.session_id,
                'length': len(chunk.content),
                'dialogue_id': dialogue_id
            }
        
        return chunks
    
    def _estimate_fact_presence(self, chunk: Chunk) -> bool:
        """
        Эвристическая оценка наличия факта.
        """
        text = chunk.content.lower()
        
        # Ключевые слова фактов
        fact_keywords = [
            'зовут', 'имя', 'лет', 'возраст', 'год', 'живу', 'город',
            'работаю', 'профессия', 'учусь', 'женат', 'замужем', 'дети',
            'люблю', 'нравится', 'хобби', 'кот', 'собака', 'питомец'
        ]
        
        # Подсчет совпадений
        matches = sum(1 for kw in fact_keywords if kw in text)
        
        # Есть числа?
        has_numbers = bool(re.search(r'\d+', text))
        
        return matches >= 2 or (matches >= 1 and has_numbers)

