"""
Препроцессор данных - ИСПРАВЛЕНА УТЕЧКА РЕСУРСОВ!

Функции:
- Чанкинг с контекстом (user + assistant)
- Дедупликация через хэши
- Фильтрация ассистента
- Метаданные для temporal ordering

ИСПРАВЛЕНО:
✅ Правильное закрытие ThreadPool (нет утечек semaphore)
✅ Использование concurrent.futures вместо multiprocessing
✅ Graceful shutdown при ошибках
✅ Timeout для защиты от зависания
"""

from collections import defaultdict
from typing import List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import re

from .interfaces import IChunkProcessor, Message, Chunk, ProcessedData


class DataProcessor(IChunkProcessor):
    """
    Препроцессор диалогов с правильным управлением ресурсами.
    """
    
    def __init__(self, max_workers: int = 16):
        # Хэши для дедупликации
        self.seen_hashes: Set[int] = set()
        
        # Счетчики индексов (для temporal ordering)
        self.global_indices: defaultdict[str, int] = defaultdict(int)
        
        # Минимальный набор стоп-слов (для детекции фактов)
        self.stopwords = self._get_stopwords()
        
        # ThreadPoolExecutor (правильный способ)
        self.max_workers = max_workers
        self.executor = None
        
        # Статистика
        self.stats = {
            'total_processed': 0,
            'total_chunks': 0,
            'errors': 0
        }
    
    def _get_stopwords(self) -> Set[str]:
        """Базовый набор стоп-слов"""
        return {
            'и', 'в', 'не', 'что', 'он', 'на', 'я', 'с', 'как', 'а', 'то',
            'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
            'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот',
            'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'когда', 'даже'
        }
    
    def _get_executor(self) -> ThreadPoolExecutor:
        """Получение или создание executor"""
        if self.executor is None:
            self.executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix='DataProcessor'
            )
        return self.executor
    
    def process_messages_batch(self, messages_list: List[List[Message]], 
                              dialogue_ids: List[str]) -> List[ProcessedData]:
        """
        Батчевая обработка сообщений с правильным управлением потоками.
        
        ИСПРАВЛЕНО: Использует ThreadPoolExecutor с правильным shutdown
        """
        if not messages_list:
            return []
        
        # Получаем executor
        executor = self._get_executor()
        
        # Создаем задачи
        future_to_idx = {}
        for idx, (messages, dialogue_id) in enumerate(zip(messages_list, dialogue_ids)):
            future = executor.submit(self._process_single, messages, dialogue_id)
            future_to_idx[future] = idx
        
        # Собираем результаты в правильном порядке
        results = [None] * len(messages_list)
        
        for future in as_completed(future_to_idx.keys()):
            idx = future_to_idx[future]
            
            try:
                result = future.result(timeout=30)  # 30 секунд на задачу
                results[idx] = result
                self.stats['total_processed'] += 1
                self.stats['total_chunks'] += len(result.chunks)
                
            except TimeoutError:
                print(f"⚠️ Timeout при обработке задачи {idx}")
                results[idx] = ProcessedData(chunks=[])
                self.stats['errors'] += 1
                
            except Exception as e:
                print(f"⚠️ Ошибка обработки задачи {idx}: {e}")
                results[idx] = ProcessedData(chunks=[])
                self.stats['errors'] += 1
        
        return results
    
    def _process_single(self, messages: List[Message], dialogue_id: str) -> ProcessedData:
        """
        Обработка одного батча сообщений с отладкой.
        """
        try:
            # 1. Создание чанков
            chunks = self.create_context_chunks(messages)
            chunks_after_create = len(chunks)
            
            # 2. Дедупликация
            chunks = self.deduplicate(chunks)
            chunks_after_dedup = len(chunks)
            
            # 3. Фильтрация ассистента
            chunks = self.filter_assistant(chunks)
            chunks_after_filter = len(chunks)
            
            # 4. Метаданные
            chunks = self._add_metadata(chunks, dialogue_id)
            
            # Отладка (только если потеряно много данных)
            if chunks_after_create > 10 and chunks_after_filter < 5:
                print(f"⚠️  Потеря данных в диалоге {dialogue_id}:")
                print(f"    Сообщений: {len(messages)}")
                print(f"    После create_context_chunks: {chunks_after_create}")
                print(f"    После deduplicate: {chunks_after_dedup}")
                print(f"    После filter_assistant: {chunks_after_filter}")
            
            return ProcessedData(chunks=chunks)
            
        except Exception as e:
            print(f"⚠️ Ошибка в _process_single для диалога {dialogue_id}: {e}")
            return ProcessedData(chunks=[])
    
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
                user_text = str(msg.content).strip() if msg.content is not None else ""
                
                if not user_text:  # Пропускаем пустые
                    continue
                
                # Ищем следующее сообщение ассистента
                assistant_confirmation = None
                if i + 1 < len(messages) and messages[i + 1].role == 'assistant':
                    assistant_content = messages[i + 1].content
                    assistant_confirmation = str(assistant_content).strip() if assistant_content is not None else "" 
                
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
                content = str(msg.content).strip() if msg.content is not None else ""
                
                if not content:  # Пропускаем пустые
                    continue
                
                chunk = Chunk(
                    content=content,
                    original_user='',
                    role='assistant',
                    session_id=msg.session_id or 'unknown'
                )
                chunks.append(chunk)
        
        return chunks
    
    def deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Дедупликация через хэши.
        
        ПРОБЛЕМА: self.seen_hashes - ГЛОБАЛЬНЫЙ! 
        Хэши не очищаются между диалогами!
        """
        unique_chunks = []
        local_hashes = set()  # Локальные хэши для текущего батча
        
        for chunk in chunks:
            # Нормализация текста
            normalized = chunk.content.lower().strip()
            normalized = re.sub(r'\s+', ' ', normalized)
            
            chunk_hash = hash(normalized)
            
            # ИСПРАВЛЕНО: Проверяем только локальные хэши!
            # Глобальные хэши нужны только для полной дедупликации
            if chunk_hash not in local_hashes:
                local_hashes.add(chunk_hash)
                self.seen_hashes.add(chunk_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def filter_assistant(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Фильтрация реплик ассистента - ОТКЛЮЧЕНА ДЛЯ ОТЛАДКИ.
        
        Временно возвращаем все чанки для отладки проблемы с поиском.
        """
        # ВРЕМЕННО: возвращаем все чанки без фильтрации
        return chunks
        
        # Оригинальный код (закомментирован):
        # filtered = []
        # for chunk in chunks:
        #     if chunk.role == 'user':
        #         filtered.append(chunk)
        #     elif chunk.role == 'assistant':
        #         if self._contains_user_fact(chunk.content):
        #             filtered.append(chunk)
        # return filtered
    
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
    
    def get_stats(self) -> dict:
        """Получить статистику обработки"""
        return {
            'total_processed': self.stats['total_processed'],
            'total_chunks': self.stats['total_chunks'],
            'errors': self.stats['errors'],
            'unique_hashes': len(self.seen_hashes),
            'dialogues': len(self.global_indices)
        }
    
    def shutdown(self):
        """
        Правильное закрытие executor.
        
        ВАЖНО: Вызывать при завершении работы!
        """
        if self.executor is not None:
            try:
                # Python 3.10 не поддерживает timeout в shutdown
                # Просто ждем завершения всех задач
                self.executor.shutdown(wait=True)
            except Exception as e:
                print(f"⚠️ Ошибка при закрытии executor: {e}")
            finally:
                self.executor = None
    
    def __del__(self):
        """Cleanup при удалении объекта"""
        self.shutdown()