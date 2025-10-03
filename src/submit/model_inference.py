"""
УПРОЩЕННЫЙ главный класс - ТОЛЬКО ВЕСА!

АРХИТЕКТУРА (3 модуля):
1. DataProcessor - препроцессинг (ИСПРАВЛЕН: нет утечек ресурсов)
2. SearchEngine - поиск через ВЕСА (эмбеддинги + FAISS + reranker)
3. AnswerBuilder - генерация через ВЕСА (GigaChat)

ИСПРАВЛЕНО:
✅ Правильное закрытие DataProcessor.executor
✅ Cleanup в __del__
✅ Graceful shutdown при ошибках
"""

from collections import defaultdict
from typing import List, Dict
import time
import numpy as np
import torch

# Условные импорты
try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError:
    AutoTokenizer = None
    LLM = None
    SamplingParams = None

from .interfaces import Message

# Импорт модулей
from .data_processor import DataProcessor
from .search_engine import SearchEngine
from .answer_builder import AnswerBuilder


class SubmitModelWithMemory:
    """
    Главный класс решения - УПРОЩЕННАЯ архитектура.
    
    Модули:
    1. DataProcessor - препроцессинг
    2. SearchEngine - поиск (ВСЯ МАГИЯ через веса!)
    3. AnswerBuilder - генерация (GigaChat)
    """
    
    def __init__(self, model_path: str, weights_dir: str = "./weights") -> None:
        print("\n" + "="*80)
        print("🚀 ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ (УПРОЩЕННАЯ АРХИТЕКТУРА)")
        print("="*80)
        
        self.model_path = model_path
        self.weights_dir = weights_dir
        init_start = time.time()
        
        # Флаг для cleanup
        self._initialized = False
        
        # Проверка зависимостей (мягкая - только предупреждение)
        if AutoTokenizer is None or LLM is None:
            print("⚠️  vllm не установлен - работаем без GigaChat")
        
        # ====================================================================
        # ШАГ 1: GigaChat (опционально)
        # ====================================================================
        print("\n[1/4] Загрузка GigaChat...")
        gigachat_start = time.time()
        
        self.model = None
        self.tokenizer = None
        
        if AutoTokenizer is not None and LLM is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
                
                self.model = LLM(
                    model=self.model_path,
                    trust_remote_code=True,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.6,
                    max_model_len=131072,
                    disable_log_stats=True
                )
                
                self.model.get_tokenizer = lambda: self.tokenizer
                print(f"✓ GigaChat загружен ({time.time() - gigachat_start:.1f}s)")
                
            except Exception as e:
                print(f"⚠️  GigaChat не загружен: {e}")
                print(f"    Продолжаем без GigaChat...")
                self.model = None
        else:
            print(f"⚠️  GigaChat пропущен (нет vllm)")
        
        # ====================================================================
        # ШАГ 2: DataProcessor (простой)
        # ====================================================================
        print("\n[2/4] Инициализация DataProcessor...")
        processor_start = time.time()
        
        self.data_processor = DataProcessor(max_workers=16)
        print(f"✓ DataProcessor готов ({time.time() - processor_start:.1f}s)")
        
        # ====================================================================
        # ШАГ 3: SearchEngine (ВСЯ МАГИЯ!)
        # ====================================================================
        print("\n[3/4] Инициализация SearchEngine...")
        print("    Загрузка: 3 эмбеддера + reranker...")
        search_start = time.time()
        
        self.search_engine = SearchEngine(
            weights_dir=weights_dir,
            cache_dir="./cache"
        )
        
        print(f"✓ SearchEngine готов ({time.time() - search_start:.1f}s)")
        print(f"    Эмбеддеры: fast/quality/best")
        
        # ====================================================================
        # ШАГ 4: AnswerBuilder
        # ====================================================================
        print("\n[4/4] Инициализация AnswerBuilder...")
        builder_start = time.time()
        
        self.answer_builder = AnswerBuilder(gigachat_model=self.model)
        print(f"✓ AnswerBuilder готов ({time.time() - builder_start:.1f}s)")
        
        # ====================================================================
        # Буферы и статистика
        # ====================================================================
        self.write_buffer: Dict[str, List[List[Message]]] = defaultdict(list)
        self.buffer_size = 50
        self.last_flush_time: Dict[str, float] = {}
        
        self.stats = {
            'init_time': time.time() - init_start,
            'write_calls': 0,
            'batch_flushes': 0,
            'questions_answered': 0,
            'total_answer_time': 0.0,
            'errors': 0,
        }
        
        self.timings = {
            'search': [],
            'generation': [],
            'total': []
        }
        
        self._initialized = True
        
        print("\n" + "="*80)
        print(f"✅ ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА: {self.stats['init_time']:.1f}s")
        print("="*80)
        print(f"Модули: DataProcessor + SearchEngine (веса!) + AnswerBuilder (веса!)")
        if torch and torch.cuda.is_available():
            print(f"GPU: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        else:
            print(f"GPU: недоступен")
        print("="*80 + "\n")
    
    # ========================================================================
    # WRITE TO MEMORY
    # ========================================================================

    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        """Запись в память с батчингом"""
        self.stats['write_calls'] += 1
        self.write_buffer[dialogue_id].append(messages)
        
        if dialogue_id not in self.last_flush_time:
            self.last_flush_time[dialogue_id] = time.time()
        
        # Flush по размеру или времени
        should_flush = (
            len(self.write_buffer[dialogue_id]) >= self.buffer_size or
            time.time() - self.last_flush_time[dialogue_id] > 10.0
        )
        
        if should_flush:
            self._flush_buffer(dialogue_id)
    
    def _flush_buffer(self, dialogue_id: str) -> None:
        """Обработка батча - ИСПРАВЛЕНО: не перезаписываем чанки!"""
        if not self.write_buffer[dialogue_id]:
            return
        
        flush_start = time.time()
        batch_size = len(self.write_buffer[dialogue_id])
        
        try:
            messages_batch = self.write_buffer[dialogue_id]
            dialogue_ids = [dialogue_id] * batch_size
            
            # Препроцессинг
            processed_batch = self.data_processor.process_messages_batch(
                messages_batch, dialogue_ids
            )
            
            # ИСПРАВЛЕНО: Собираем ВСЕ чанки из всех processed результатов
            all_chunks = []
            for processed in processed_batch:
                if processed.chunks:
                    all_chunks.extend(processed.chunks)
            
            # Индексация ОДИН РАЗ со всеми чанками
            if all_chunks:
                self.search_engine.build_index(all_chunks, dialogue_id)
            
            self.write_buffer[dialogue_id] = []
            self.last_flush_time[dialogue_id] = time.time()
            self.stats['batch_flushes'] += 1
            
            if self.stats['batch_flushes'] % 10 == 0:
                flush_time = time.time() - flush_start
                chunks_count = len(all_chunks)
                print(f"[Flush] Батч #{self.stats['batch_flushes']}: "
                      f"{batch_size} сообщений, {chunks_count} чанков за {flush_time:.2f}s")
                
        except Exception as e:
            print(f"❌ Ошибка flush: {e}")
            self.stats['errors'] += 1
            self.write_buffer[dialogue_id] = []

    def clear_memory(self, dialogue_id: str) -> None:
        """Очистка памяти"""
        if dialogue_id in self.write_buffer and self.write_buffer[dialogue_id]:
            self._flush_buffer(dialogue_id)
        
        if dialogue_id in self.write_buffer:
            del self.write_buffer[dialogue_id]
        if dialogue_id in self.last_flush_time:
            del self.last_flush_time[dialogue_id]
    
    # ========================================================================
    # ANSWER TO QUESTION (УПРОЩЕННЫЙ ПАЙПЛАЙН!)
    # ========================================================================

    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        """
        УПРОЩЕННЫЙ пайплайн:
        
        1. Flush буфера
        2. Поиск через SearchEngine (ВЕСА: эмбеддинги + reranker)
        3. Генерация через AnswerBuilder (ВЕСА: GigaChat)
        
        БЕЗ КЛАССИФИКАЦИИ! Всё через веса!
        """
        total_start = time.time()
        
        try:
            # 1. Flush буфера
            if dialogue_id in self.write_buffer and self.write_buffer[dialogue_id]:
                self._flush_buffer(dialogue_id)
            
            # ============================================================
            # 2. ПОИСК ЧЕРЕЗ ВЕСА (эмбеддинги + FAISS + BM25 + reranker)
            # ============================================================
            search_start = time.time()
            
            search_results = self.search_engine.search(
                question=question,
                dialogue_id=dialogue_id,
                top_k=50  # Топ-50 после reranker
            )
            
            search_time = time.time() - search_start
            self.timings['search'].append(search_time)
            
            # ============================================================
            # 3. ГЕНЕРАЦИЯ ЧЕРЕЗ GIGACHAT (веса)
            # ============================================================
            generation_start = time.time()
            
            answer = self.answer_builder.generate_answer(
                question=question,
                search_results=search_results,
                dialogue_id=dialogue_id
            )
            
            generation_time = time.time() - generation_start
            self.timings['generation'].append(generation_time)
            
            # Статистика
            total_time = time.time() - total_start
            self.timings['total'].append(total_time)
            
            self.stats['questions_answered'] += 1
            self.stats['total_answer_time'] += total_time
            
            # Логирование каждые 50 вопросов
            if self.stats['questions_answered'] % 50 == 0:
                self._print_progress()
            
            return answer

        except Exception as e:
            print(f"\n❌ ОШИБКА при ответе:")
            print(f"   Вопрос: {question[:100]}...")
            print(f"   Ошибка: {e}")
            self.stats['errors'] += 1
            return "Не удалось найти ответ в диалоге."
    
    # ========================================================================
    # СТАТИСТИКА
    # ========================================================================
    
    def _print_progress(self) -> None:
        """Прогресс обработки"""
        answered = self.stats['questions_answered']
        total = 1167
        progress = (answered / total) * 100
        
        avg_total = np.mean(self.timings['total'][-50:]) if self.timings['total'] else 0
        avg_search = np.mean(self.timings['search'][-50:]) if self.timings['search'] else 0
        avg_gen = np.mean(self.timings['generation'][-50:]) if self.timings['generation'] else 0
        
        remaining = total - answered
        eta_minutes = (remaining * avg_total) / 60
        
        print(f"\n{'='*80}")
        print(f"📊 ПРОГРЕСС: {answered}/{total} ({progress:.1f}%)")
        print(f"{'='*80}")
        print(f"Среднее время: {avg_total:.2f}s")
        print(f"  ├─ Поиск (веса):     {avg_search:.3f}s")
        print(f"  └─ Генерация (веса): {avg_gen:.3f}s")
        print(f"\nETA: ~{eta_minutes:.1f} минут")
        print(f"Ошибок: {self.stats['errors']}")
        
        # Кэш
        hit_rate = self.search_engine.embedding_cache.get_hit_rate()
        print(f"Кэш эмбеддингов: {hit_rate*100:.1f}% hit rate")
        print(f"{'='*80}\n")
    
    def print_final_stats(self) -> None:
        """Финальная статистика"""
        print("\n" + "="*80)
        print("📈 ФИНАЛЬНАЯ СТАТИСТИКА")
        print("="*80)
        
        avg_time = self.stats['total_answer_time'] / max(1, self.stats['questions_answered'])
        
        print(f"\n⏱️  ВРЕМЯ:")
        print(f"  Инициализация: {self.stats['init_time']:.1f}s")
        print(f"  Общее:         {self.stats['total_answer_time']:.1f}s")
        print(f"  Среднее:       {avg_time:.2f}s/вопрос")
        
        if self.timings['search']:
            print(f"\n📊 ЭТАПЫ (среднее):")
            print(f"  Поиск:     {np.mean(self.timings['search']):.3f}s")
            print(f"  Генерация: {np.mean(self.timings['generation']):.3f}s")
        
        print(f"\n📝 ОБРАБОТКА:")
        print(f"  write_to_memory: {self.stats['write_calls']} вызовов")
        print(f"  Батчей:          {self.stats['batch_flushes']}")
        print(f"  Вопросов:        {self.stats['questions_answered']}")
        print(f"  Ошибок:          {self.stats['errors']}")
        
        hit_rate = self.search_engine.embedding_cache.get_hit_rate()
        print(f"\n💾 КЭШ:")
        print(f"  Hit rate: {hit_rate*100:.1f}%")
        
        # Статистика DataProcessor
        if hasattr(self.data_processor, 'get_stats'):
            dp_stats = self.data_processor.get_stats()
            print(f"\n📦 DATAPROCESSOR:")
            print(f"  Обработано: {dp_stats['total_processed']}")
            print(f"  Чанков: {dp_stats['total_chunks']}")
            print(f"  Ошибок: {dp_stats['errors']}")
        
        print(f"\n💾 GPU:")
        if torch and torch.cuda.is_available():
            print(f"  Использовано: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        else:
            print(f"  недоступен")
        print("\n" + "="*80 + "\n")
    
    def shutdown(self):
        """
        Правильное закрытие всех ресурсов.
        
        ВАЖНО: Вызывать при завершении!
        """
        if not self._initialized:
            return
        
        print("\n🧹 Закрытие ресурсов...")
        
        # Закрываем DataProcessor
        if hasattr(self, 'data_processor'):
            try:
                self.data_processor.shutdown()
                print("✓ DataProcessor закрыт")
            except Exception as e:
                print(f"⚠️ Ошибка закрытия DataProcessor: {e}")
        
        # Финальная статистика
        if hasattr(self, 'stats'):
            self.print_final_stats()
        
        self._initialized = False
        print("✓ Все ресурсы освобождены\n")
    
    def __del__(self):
        """Cleanup при удалении объекта"""
        self.shutdown()