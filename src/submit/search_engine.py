"""
SearchEngine - поиск через ВЕСА!

АРХИТЕКТУРА:
1. 3 эмбеддера (fast/quality/best)
2. FAISS индексы
3. BM25 для точного поиска
4. Reranker для финальной сортировки

ВСЯ МАГИЯ ЧЕРЕЗ ВЕСА!
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch

from .interfaces import Chunk, SearchResult


class EmbeddingCache:
    """Кэш для эмбеддингов"""
    
    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Получение эмбеддинга из кэша"""
        if text in self.cache:
            self.hits += 1
            return self.cache[text]
        else:
            self.misses += 1
            return None
    
    def put(self, text: str, embedding: np.ndarray) -> None:
        """Добавление эмбеддинга в кэш"""
        if len(self.cache) >= self.max_size:
            # Удаляем самый старый элемент
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[text] = embedding
    
    def get_hit_rate(self) -> float:
        """Статистика hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class SearchEngine:
    """
    Поисковый движок через ВЕСА!
    
    Компоненты:
    1. 3 эмбеддера (fast/quality/best)
    2. FAISS индексы для каждого эмбеддера
    3. BM25 для точного поиска
    4. Reranker для финальной сортировки
    """
    
    def __init__(self, weights_dir: str, cache_dir: str = "./cache"):
        self.weights_dir = Path(weights_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Кэш эмбеддингов
        self.embedding_cache = EmbeddingCache()
        
        # Эмбеддеры
        self.embeddings = {}
        self.faiss_indices = {}
        self.bm25_indices = {}
        
        # Данные для каждого диалога
        self.dialogue_chunks: Dict[str, List[Chunk]] = {}
        self.dialogue_texts: Dict[str, List[str]] = {}
        
        # Загружаем модели
        self._load_models()
    
    def _load_models(self):
        """Загрузка всех моделей"""
        print("    Загрузка эмбеддеров...")
        
        # Fast эмбеддер (быстрый)
        try:
            self.embeddings['fast'] = SentenceTransformer(
                str(self.weights_dir / "multilingual-e5-small"),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("    ✓ Fast эмбеддер загружен")
        except Exception as e:
            print(f"    ❌ Ошибка загрузки fast эмбеддера: {e}")
        
        # Quality эмбеддер (качественный)
        try:
            self.embeddings['quality'] = SentenceTransformer(
                str(self.weights_dir / "paraphrase-multilingual-mpnet"),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("    ✓ Quality эмбеддер загружен")
        except Exception as e:
            print(f"    ❌ Ошибка загрузки quality эмбеддера: {e}")
        
        # Best эмбеддер (лучший)
        try:
            self.embeddings['best'] = SentenceTransformer(
                str(self.weights_dir / "rubert-tiny2"),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("    ✓ Best эмбеддер загружен")
        except Exception as e:
            print(f"    ❌ Ошибка загрузки best эмбеддера: {e}")
        
        # Reranker
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                str(self.weights_dir / "bge-reranker-base")
            )
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                str(self.weights_dir / "bge-reranker-base")
            )
            if torch.cuda.is_available():
                self.reranker_model = self.reranker_model.cuda()
            print("    ✓ Reranker загружен")
        except Exception as e:
            print(f"    ❌ Ошибка загрузки reranker: {e}")
            self.reranker_model = None
    
    def build_index(self, chunks: List[Chunk], dialogue_id: str):
        """Построение индексов для диалога"""
        if not chunks:
            return
        
        # Сохраняем чанки
        self.dialogue_chunks[dialogue_id] = chunks
        texts = [chunk.content for chunk in chunks]
        self.dialogue_texts[dialogue_id] = texts
        
        # Создаем BM25 индекс
        tokenized_texts = [text.split() for text in texts]
        self.bm25_indices[dialogue_id] = BM25Okapi(tokenized_texts)
        
        # Создаем FAISS индексы для каждого эмбеддера
        for emb_name, embedder in self.embeddings.items():
            if embedder is None:
                continue
            
            # Получаем эмбеддинги
            embeddings = self._get_embeddings(texts, embedder)
            
            # Создаем FAISS индекс
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner Product для косинусного сходства
            
            # Нормализуем эмбеддинги для косинусного сходства
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            
            self.faiss_indices[f"{dialogue_id}_{emb_name}"] = index
    
    def _get_embeddings(self, texts: List[str], embedder) -> np.ndarray:
        """Получение эмбеддингов с кэшированием"""
        embeddings = []
        
        for text in texts:
            # Проверяем кэш
            cached = self.embedding_cache.get(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                # Генерируем новый эмбеддинг
                embedding = embedder.encode([text], convert_to_numpy=True)[0]
                self.embedding_cache.put(text, embedding)
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def search(self, question: str, dialogue_id: str, top_k: int = 20) -> List[SearchResult]:
        """
        Поиск через ВЕСА!
        
        1. Семантический поиск (3 эмбеддера)
        2. BM25 поиск
        3. Reranker для финальной сортировки
        """
        if dialogue_id not in self.dialogue_chunks:
            return []
        
        chunks = self.dialogue_chunks[dialogue_id]
        texts = self.dialogue_texts[dialogue_id]
        
        # Собираем все результаты
        all_results = []
        
        # 1. Семантический поиск через каждый эмбеддер
        for emb_name, embedder in self.embeddings.items():
            if embedder is None:
                continue
            
            index_key = f"{dialogue_id}_{emb_name}"
            if index_key not in self.faiss_indices:
                continue
            
            # Получаем эмбеддинг вопроса
            question_emb = embedder.encode([question], convert_to_numpy=True)[0]
            question_emb = question_emb.reshape(1, -1)
            faiss.normalize_L2(question_emb)
            
            # Поиск в FAISS
            index = self.faiss_indices[index_key]
            scores, indices = index.search(question_emb, min(top_k, len(chunks)))
            
            # Добавляем результаты
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(chunks):
                    result = SearchResult(
                        chunk=chunks[idx],
                        score=float(score),
                        final_score=float(score)  # Пока без reranker
                    )
                    all_results.append(result)
        
        # 2. BM25 поиск
        if dialogue_id in self.bm25_indices:
            bm25_scores = self.bm25_indices[dialogue_id].get_scores(question.split())
            
            for i, score in enumerate(bm25_scores):
                if i < len(chunks):
                    result = SearchResult(
                        chunk=chunks[i],
                        score=float(score),
                        final_score=float(score)  # Пока без reranker
                    )
                    all_results.append(result)
        
        # 3. Reranker (если доступен)
        if self.reranker_model is not None and all_results:
            all_results = self._rerank_results(question, all_results)
        
        # 4. Дедупликация и сортировка
        unique_results = self._deduplicate_results(all_results)
        unique_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return unique_results[:top_k]
    
    def _rerank_results(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        """Reranker для финальной сортировки"""
        if not results:
            return results
        
        try:
            # Подготавливаем пары для reranker
            pairs = []
            for result in results:
                pairs.append([question, result.chunk.content])
            
            # Reranker inference
            inputs = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                scores = self.reranker_model(**inputs).logits.squeeze(-1)
            
            # Обновляем финальные скоры
            for i, result in enumerate(results):
                result.final_score = float(scores[i])
            
        except Exception as e:
            print(f"⚠️  Ошибка reranker: {e}")
            # Если reranker не работает, используем исходные скоры
            for result in results:
                result.final_score = result.score
        
        return results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Дедупликация результатов"""
        seen_content = set()
        unique_results = []
        
        for result in results:
            content = result.chunk.content
            if content not in seen_content:
                seen_content.add(content)
                unique_results.append(result)
        
        return unique_results