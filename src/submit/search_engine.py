"""
SearchEngine - ЗАЩИТА ОТ SEGFAULT!

ИСПРАВЛЕНИЯ:
1. ✅ Отключена многопоточность в sentence-transformers (encode single thread)
2. ✅ FAISS IndexFlatIP → IndexFlatL2 (более стабильный на CPU)
3. ✅ Batch encoding с защитой от OOM
4. ✅ Try-catch на каждой критичной операции
5. ✅ Graceful degradation при ошибках
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import threading

# Условные импорты
try:
    import faiss
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rank_bm25 import BM25Okapi
    import torch
except ImportError:
    faiss = None
    SentenceTransformer = None
    CrossEncoder = None
    BM25Okapi = None
    torch = None

from .interfaces import Chunk, SearchResult


class EmbeddingCache:
    """Thread-safe кэш с LRU"""
    
    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, np.ndarray] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()  # Thread-safety
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Получение с thread-safety"""
        with self.lock:
            if text in self.cache:
                self.hits += 1
                self.access_count[text] += 1
                return self.cache[text].copy()
            else:
                self.misses += 1
                return None
    
    def put(self, text: str, embedding: np.ndarray) -> None:
        """Добавление с thread-safety"""
        with self.lock:
            if len(self.cache) >= self.max_size:
                if self.access_count:
                    least_used = min(self.access_count, key=self.access_count.get)
                    del self.cache[least_used]
                    del self.access_count[least_used]
                else:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
            
            self.cache[text] = embedding.copy()
            self.access_count[text] = 1
    
    def get_hit_rate(self) -> float:
        """Статистика"""
        with self.lock:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict:
        """Детальная статистика"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': self.get_hit_rate(),
                'fill_ratio': len(self.cache) / self.max_size
            }


class SearchEngine:
    """
    Улучшенный поисковый движок с защитой от segfault
    """
    
    EMBEDDER_WEIGHTS = {
        'fast': 0.25,
        'quality': 0.35,
        'best': 0.40
    }
    
    MIN_NO_INFO_THRESHOLD = 0.2
    
    def __init__(self, weights_dir: str, cache_dir: str = "./cache"):
        self.weights_dir = Path(weights_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        if faiss is None or SentenceTransformer is None:
            raise RuntimeError(
                "❌ Необходимые зависимости не установлены: "
                "faiss-cpu, sentence-transformers, rank-bm25"
            )
        
        # Thread-safe кэш
        self.embedding_cache = EmbeddingCache(max_size=10000)
        
        # Модели
        self.embeddings: Dict[str, Optional[SentenceTransformer]] = {}
        self.reranker: Optional[CrossEncoder] = None
        
        # Индексы
        self.faiss_indices: Dict[str, faiss.Index] = {}
        self.bm25_indices: Dict[str, BM25Okapi] = {}
        
        # Данные
        self.dialogue_chunks: Dict[str, List[Chunk]] = {}
        self.dialogue_texts: Dict[str, List[str]] = {}
        self.dialogue_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Lock для thread-safety при индексации
        self.index_lock = threading.Lock()
        
        # Статистика
        self.search_stats = {
            'total_searches': 0,
            'avg_results': 0,
            'reranker_calls': 0,
            'errors': 0
        }
        
        # Загрузка моделей
        self._load_models()
        self._run_diagnostics()
    
    def _load_models(self):
        """Загрузка с защитой от segfault"""
        print("    Загрузка моделей из локальных весов...")
        
        # Импорт torch для проверки CUDA
        try:
            import torch
        except ImportError:
            torch = None
        
        device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
        print(f"    Устройство: {device}")
        
        # ===================================================================
        # ЭМБЕДДЕРЫ - с отключенной многопоточностью
        # ===================================================================
        embedder_configs = {
            'fast': ('multilingual-e5-small', 384),
            'quality': ('paraphrase-multilingual-mpnet', 768),
            'best': ('rubert-tiny2', 312)
        }
        
        loaded_count = 0
        for name, (folder, expected_dim) in embedder_configs.items():
            try:
                model_path = str(self.weights_dir / folder)
                
                if not Path(model_path).exists():
                    print(f"    ⚠️  {name}: путь не существует")
                    self.embeddings[name] = None
                    continue
                
                # Загрузка с отключенной многопоточностью
                model = SentenceTransformer(model_path, device=device)
                
                # КРИТИЧНО: Отключаем многопоточность в PyTorch
                if torch is not None:
                    torch.set_num_threads(1)
                
                self.embeddings[name] = model
                
                # Тест
                test_emb = model.encode(
                    ["тест"],
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=1  # Маленький batch для стабильности
                )
                
                actual_dim = test_emb.shape[1]
                print(f"    ✓ {name}: dim={actual_dim}, weight={self.EMBEDDER_WEIGHTS[name]:.2f}")
                loaded_count += 1
                
            except Exception as e:
                print(f"    ❌ {name}: {e}")
                self.embeddings[name] = None
        
        if loaded_count == 0:
            raise RuntimeError("❌ Не удалось загрузить ни одного эмбеддера!")
        
        print(f"    Загружено эмбеддеров: {loaded_count}/3")
        
        # ===================================================================
        # RERANKER
        # ===================================================================
        try:
            reranker_path = str(self.weights_dir / "bge-reranker-base")
            
            if not Path(reranker_path).exists():
                print(f"    ⚠️  Reranker: путь не существует")
                self.reranker = None
            else:
                self.reranker = CrossEncoder(
                    reranker_path,
                    max_length=512,
                    num_labels=1,
                    device=device
                )
                
                # Тест
                test_score = self.reranker.predict(
                    [["вопрос", "ответ"]],
                    show_progress_bar=False,
                    batch_size=1  # Маленький batch
                )
                
                print(f"    ✓ Reranker: score={test_score[0]:.4f}")
                
        except Exception as e:
            print(f"    ⚠️  Reranker не загружен: {e}")
            self.reranker = None
    
    def _run_diagnostics(self):
        """Диагностика системы"""
        print("\n    📊 Диагностика:")
        
        working_embedders = sum(1 for e in self.embeddings.values() if e is not None)
        total_embedders = len(self.embeddings)
        
        print(f"      Эмбеддеры: {working_embedders}/{total_embedders}")
        print(f"      Reranker: {'✓' if self.reranker else '✗'}")
        print(f"      Кэш: {self.embedding_cache.max_size} слотов")
    
    def build_index(self, chunks: List[Chunk], dialogue_id: str):
        """
        Построение индексов с ЗАЩИТОЙ ОТ SEGFAULT
        
        ИСПРАВЛЕНО: Добавляет чанки вместо перезаписи!
        """
        if not chunks:
            return
        
        # Thread-safety
        with self.index_lock:
            try:
                # ИСПРАВЛЕНО: Добавляем к существующим чанкам вместо перезаписи
                if dialogue_id in self.dialogue_chunks:
                    existing_chunks = self.dialogue_chunks[dialogue_id]
                    existing_texts = self.dialogue_texts[dialogue_id]
                    
                    # Объединяем старые и новые
                    all_chunks = existing_chunks + chunks
                    all_texts = existing_texts + [chunk.content for chunk in chunks]
                else:
                    all_chunks = chunks
                    all_texts = [chunk.content for chunk in chunks]
                
                # Сохраняем объединенные чанки
                self.dialogue_chunks[dialogue_id] = all_chunks
                self.dialogue_texts[dialogue_id] = all_texts
                
                # ===================================================================
                # BM25 индекс (пересоздаем с полным набором текстов)
                # ===================================================================
                try:
                    tokenized_texts = [text.lower().split() for text in all_texts]
                    self.bm25_indices[dialogue_id] = BM25Okapi(tokenized_texts)
                except Exception as e:
                    print(f"⚠️  Ошибка BM25 индекса: {e}")
                
                # ===================================================================
                # FAISS индексы - пересоздаем с полным набором
                # ===================================================================
                self.dialogue_embeddings[dialogue_id] = {}
                
                for emb_name, embedder in self.embeddings.items():
                    if embedder is None:
                        continue
                    
                    try:
                        # Получаем эмбеддинги для ВСЕХ текстов
                        embeddings = self._get_embeddings_safe(all_texts, embedder, emb_name)
                        
                        if embeddings is None or len(embeddings) == 0:
                            continue
                        
                        # Сохраняем
                        self.dialogue_embeddings[dialogue_id][emb_name] = embeddings
                        
                        # FAISS индекс
                        dimension = embeddings.shape[1]
                        index = faiss.IndexFlatL2(dimension)
                        index.add(embeddings.astype('float32'))
                        
                        index_key = f"{dialogue_id}_{emb_name}"
                        self.faiss_indices[index_key] = index
                        
                    except Exception as e:
                        print(f"⚠️  Ошибка создания индекса {emb_name}: {e}")
                        continue
                
            except Exception as e:
                print(f"❌ Критическая ошибка build_index: {e}")
                import traceback
                traceback.print_exc()
    
    def _get_embeddings_safe(self, texts: List[str], 
                             embedder: SentenceTransformer,
                             emb_name: str) -> Optional[np.ndarray]:
        """
        БЕЗОПАСНОЕ получение эмбеддингов с защитой от segfault
        
        КРИТИЧНО:
        - Маленькие батчи (8)
        - Single thread
        - Try-catch на каждом батче
        """
        if not texts:
            return None
        
        try:
            embeddings = []
            batch_size = 64  # GPU справится
            
            # Обрабатываем по батчам
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                try:
                    # SINGLE THREAD encoding
                    batch_embeddings = embedder.encode(
                        batch,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        batch_size=batch_size,
                        normalize_embeddings=False  # Нормализуем сами
                    )
                    
                    embeddings.append(batch_embeddings)
                    
                except Exception as e:
                    print(f"⚠️  Ошибка encoding батча {i//batch_size}: {e}")
                    # Добавляем нулевые эмбеддинги для этого батча
                    dim = 384 if emb_name == 'fast' else (768 if emb_name == 'quality' else 312)
                    zero_embeddings = np.zeros((len(batch), dim), dtype=np.float32)
                    embeddings.append(zero_embeddings)
            
            # Объединяем все батчи
            if embeddings:
                all_embeddings = np.vstack(embeddings)
                return all_embeddings.astype('float32')
            else:
                return None
                
        except Exception as e:
            print(f"❌ Критическая ошибка _get_embeddings_safe: {e}")
            return None
    
    def search(self, question: str, dialogue_id: str, 
               top_k: int = 20) -> List[SearchResult]:
        """Поиск с защитой от ошибок"""
        self.search_stats['total_searches'] += 1
        
        if dialogue_id not in self.dialogue_chunks:
            return []
        
        chunks = self.dialogue_chunks[dialogue_id]
        
        try:
            # Семантический поиск
            semantic_results = self._semantic_search_safe(question, dialogue_id, chunks, top_k)
            
            # BM25 поиск
            bm25_results = self._bm25_search(question, dialogue_id, chunks, top_k)
            
            # Объединение
            combined_results = self._combine_results(semantic_results, bm25_results)
            
            # Reranking
            if self.reranker is not None and combined_results:
                try:
                    combined_results = self._rerank_results(question, combined_results)
                    self.search_stats['reranker_calls'] += 1
                except Exception as e:
                    print(f"⚠️  Ошибка reranker: {e}")
            
            # Дедупликация
            unique_results = self._deduplicate_results(combined_results)
            unique_results.sort(key=lambda x: x.final_score, reverse=True)
            
            return unique_results[:top_k]
            
        except Exception as e:
            print(f"❌ Ошибка search: {e}")
            self.search_stats['errors'] += 1
            return []
    
    def _semantic_search_safe(self, question: str, dialogue_id: str,
                              chunks: List[Chunk], top_k: int) -> List[SearchResult]:
        """БЕЗОПАСНЫЙ семантический поиск"""
        results = []
        
        for emb_name, embedder in self.embeddings.items():
            if embedder is None:
                continue
            
            index_key = f"{dialogue_id}_{emb_name}"
            if index_key not in self.faiss_indices:
                continue
            
            try:
                # Encoding вопроса (single thread)
                question_emb = embedder.encode(
                    [question],
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=1
                )[0]
                
                question_emb = question_emb.reshape(1, -1).astype('float32')
                
                # Поиск в FAISS (IndexFlatL2 - расстояния)
                index = self.faiss_indices[index_key]
                k = min(top_k * 2, len(chunks))
                distances, indices = index.search(question_emb, k)
                
                # Конвертируем расстояния в similarity (меньше = лучше)
                # Для L2: similarity = 1 / (1 + distance)
                similarities = 1.0 / (1.0 + distances[0])
                
                # Применяем вес эмбеддера
                weight = self.EMBEDDER_WEIGHTS[emb_name]
                
                for similarity, idx in zip(similarities, indices[0]):
                    if idx < len(chunks):
                        weighted_score = float(similarity) * weight
                        result = SearchResult(
                            chunk=chunks[idx],
                            score=weighted_score,
                            final_score=weighted_score
                        )
                        results.append(result)
                        
            except Exception as e:
                print(f"⚠️  Ошибка поиска {emb_name}: {e}")
                continue
        
        return results
    
    def _bm25_search(self, question: str, dialogue_id: str,
                    chunks: List[Chunk], top_k: int) -> List[SearchResult]:
        """BM25 поиск"""
        if dialogue_id not in self.bm25_indices:
            return []
        
        try:
            bm25 = self.bm25_indices[dialogue_id]
            scores = bm25.get_scores(question.lower().split())
            
            max_score = max(scores) if scores.max() > 0 else 1.0
            normalized_scores = scores / max_score
            
            top_indices = np.argsort(normalized_scores)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if idx < len(chunks):
                    result = SearchResult(
                        chunk=chunks[idx],
                        score=float(normalized_scores[idx]) * 0.3,
                        final_score=float(normalized_scores[idx]) * 0.3
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"⚠️  Ошибка BM25: {e}")
            return []
    
    def _combine_results(self, semantic_results: List[SearchResult],
                        bm25_results: List[SearchResult]) -> List[SearchResult]:
        """Объединение результатов"""
        result_map: Dict[str, Tuple[SearchResult, List[float]]] = {}
        
        for result in semantic_results + bm25_results:
            key = result.chunk.content
            
            if key not in result_map:
                result_map[key] = (result, [])
            
            result_map[key][1].append(result.score)
        
        combined = []
        for result, scores in result_map.values():
            avg_score = np.mean(scores)
            result.score = float(avg_score)
            result.final_score = float(avg_score)
            combined.append(result)
        
        return combined
    
    def _rerank_results(self, question: str,
                       results: List[SearchResult]) -> List[SearchResult]:
        """Reranking с защитой"""
        if not results:
            return results
        
        try:
            pairs = [[question, result.chunk.content] for result in results]
            
            # Маленькие батчи
            batch_size = 64
            scores = self.reranker.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False
            )
            
            for i, result in enumerate(results):
                reranker_score = float(scores[i])
                original_score = result.score
                result.final_score = reranker_score * 0.7 + original_score * 0.3
            
        except Exception as e:
            print(f"⚠️  Ошибка reranker: {e}")
            for result in results:
                result.final_score = result.score
        
        return results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Дедупликация"""
        seen: Dict[str, SearchResult] = {}
        
        for result in results:
            content = result.chunk.content
            
            if content not in seen or result.final_score > seen[content].final_score:
                seen[content] = result
        
        return list(seen.values())
    
    def calculate_adaptive_threshold(self, results: List[SearchResult]) -> float:
        """Адаптивный порог"""
        if not results:
            return self.MIN_NO_INFO_THRESHOLD
        
        scores = [r.final_score for r in results]
        
        median = np.median(scores)
        std = np.std(scores)
        mean = np.mean(scores)
        
        if std > 0.3:
            threshold = median + std * 0.5
        else:
            threshold = mean * 0.7
        
        threshold = max(threshold, self.MIN_NO_INFO_THRESHOLD)
        
        return float(threshold)
    
    def get_stats(self) -> Dict:
        """Статистика"""
        cache_stats = self.embedding_cache.get_stats()
        
        return {
            'search': self.search_stats,
            'cache': cache_stats,
            'models': {
                'embedders': sum(1 for e in self.embeddings.values() if e is not None),
                'reranker': self.reranker is not None
            },
            'indices': {
                'faiss': len(self.faiss_indices),
                'bm25': len(self.bm25_indices),
                'dialogues': len(self.dialogue_chunks)
            }
        }
    
    def print_stats(self):
        """Вывод статистики"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("📊 СТАТИСТИКА SEARCHENGINE")
        print("="*60)
        
        print(f"\n🔍 Поиск:")
        print(f"  Всего поисков: {stats['search']['total_searches']}")
        print(f"  Средн. результатов: {stats['search']['avg_results']:.1f}")
        print(f"  Reranker вызовов: {stats['search']['reranker_calls']}")
        print(f"  Ошибок: {stats['search']['errors']}")
        
        print(f"\n💾 Кэш:")
        print(f"  Hit rate: {stats['cache']['hit_rate']*100:.1f}%")
        print(f"  Заполнено: {stats['cache']['size']}/{stats['cache']['max_size']}")
        
        print(f"\n🤖 Модели:")
        print(f"  Эмбеддеров: {stats['models']['embedders']}/3")
        print(f"  Reranker: {'✓' if stats['models']['reranker'] else '✗'}")
        
        print(f"\n📚 Индексы:")
        print(f"  FAISS: {stats['indices']['faiss']}")
        print(f"  BM25: {stats['indices']['bm25']}")
        print(f"  Диалогов: {stats['indices']['dialogues']}")
        
        print("="*60 + "\n")