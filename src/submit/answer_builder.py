"""
УПРОЩЕННЫЙ генератор ответов - БЕЗ ТИПОВ ВОПРОСОВ!

ФИЛОСОФИЯ:
- Reranker УЖЕ отсортировал результаты по релевантности (веса!)
- Просто берем топ-N и генерируем ответ через GigaChat (веса!)
- Никаких правил и адаптивных стратегий!

ОДИН ВЫЗОВ GIGACHAT НА ВОПРОС!
"""

from typing import List, Dict
from dataclasses import asdict
import re

from .interfaces import IAnswerGenerator, SearchResult, Message, NO_INFO_THRESHOLD

# Условный импорт vllm
try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None


class AnswerBuilder(IAnswerGenerator):
    """
    Генератор ответов - УПРОЩЕННЫЙ!
    
    Логика:
    1. Проверка кэша
    2. Проверка на NO_INFO (по порогу score)
    3. Генерация через GigaChat (1 вызов!)
    4. Постобработка
    """
    
    def __init__(self, gigachat_model):
        self.gigachat = gigachat_model
        
        # Параметры генерации
        if SamplingParams is not None:
            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=100,
                seed=42
            )
        else:
            self.sampling_params = None
        
        # Кэш
        self.answer_cache: Dict[str, str] = {}
        
        # Статистика
        self.gigachat_calls = 0
    
    def generate_answer(self, question: str,
                       search_results: List[SearchResult],
                       dialogue_id: str) -> str:
        """
        Генерация ответа - ОДИН вызов GigaChat!
        
        Args:
            question: Вопрос
            search_results: Результаты поиска (УЖЕ отсортированы reranker'ом!)
            dialogue_id: ID диалога
        """
        
        # 1. Кэш
        cache_key = f"{dialogue_id}:{hash(question)}"
        if cache_key in self.answer_cache:
            return self.answer_cache[cache_key]
        
        # 2. Проверка на NO_INFO
        if self._is_no_info(search_results):
            answer = "Эта информация не упоминалась в диалоге."
            self.answer_cache[cache_key] = answer
            return answer
        
        # 3. Подготовка контекста
        context = self._prepare_context(search_results)
        
        # 4. Генерация промпта
        prompt = self._build_prompt(question, context)
        
        # 5. ЕДИНСТВЕННЫЙ вызов GigaChat
        answer = self._inference(prompt)
        self.gigachat_calls += 1
        
        # 6. Постобработка
        answer = self._cleanup_answer(answer)
        
        # 7. Кэш
        self.answer_cache[cache_key] = answer
        
        return answer
    
    def _is_no_info(self, results: List[SearchResult]) -> bool:
        """
        Определение отсутствия информации.
        
        Просто проверяем лучший score!
        """
        if not results:
            return True
        
        # Лучший score (results УЖЕ отсортированы!)
        best_score = results[0].final_score
        
        return best_score < NO_INFO_THRESHOLD
    
    def _prepare_context(self, results: List[SearchResult]) -> str:
        """
        Подготовка контекста для промпта.
        
        Берем топ-5 результатов (они УЖЕ отсортированы reranker'ом!)
        """
        facts = []
        seen = set()
        
        for result in results[:5]:
            fact = result.chunk.original_user or result.chunk.content
            fact_normalized = fact.lower().strip()
            
            if fact_normalized not in seen and len(facts) < 5:
                facts.append(fact)
                seen.add(fact_normalized)
        
        return "\n".join([f"- {f}" for f in facts])
    
    def _build_prompt(self, question: str, context: str) -> str:
        """
        Создание промпта для GigaChat.
        
        ОДИН универсальный промпт для всех вопросов!
        """
        prompt = f"""На основе информации из диалога ответь на вопрос кратко (одно предложение).

Информация из диалога:
{context}

Вопрос: {question}

Инструкция: Дай точный краткий ответ одним предложением. Используй только информацию из диалога выше.

Ответ:"""
        
        return prompt
    
    def _cleanup_answer(self, answer: str) -> str:
        """
        Постобработка ответа.
        
        - Только первое предложение
        - Убираем водные фразы
        - Capitalize
        """
        # Убираем лишние пробелы
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Только первое предложение
        match = re.search(r'[.!?]', answer)
        if match:
            answer = answer[:match.end()].strip()
        
        # Убираем водные фразы
        water_patterns = [
            r'^на основе (диалога|информации),?\s*',
            r'^согласно (диалогу|информации),?\s*',
            r'^из диалога следует,?\s*',
            r'^судя по диалогу,?\s*',
        ]
        
        for pattern in water_patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE).strip()
        
        # Capitalize
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        
        return answer
    
    def _inference(self, prompt: str) -> str:
        """
        Вызов GigaChat для генерации.
        """
        try:
            if self.sampling_params is None:
                return "Модель не инициализирована (vllm не установлен)."
            
            messages = [
                Message(role='system', content='Ты помощник, который отвечает на вопросы кратко и точно.'),
                Message(role='user', content=prompt)
            ]
            
            msg_dicts = [asdict(m) for m in messages]
            
            input_ids = self.gigachat.get_tokenizer().apply_chat_template(
                msg_dicts,
                add_generation_prompt=True
            )
            
            outputs = self.gigachat.generate(
                prompt_token_ids=[input_ids],
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            
            result = outputs[0].outputs[0].text
            return result.strip()
            
        except Exception as e:
            print(f"⚠️  Ошибка генерации: {e}")
            return "Не удалось сгенерировать ответ."
    
    def get_stats(self) -> Dict:
        """Статистика"""
        return {
            'total_calls': self.gigachat_calls,
            'cache_size': len(self.answer_cache)
        }
