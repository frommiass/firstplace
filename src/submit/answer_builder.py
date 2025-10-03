"""
УЛУЧШЕННЫЙ генератор ответов с temporal reasoning

УЛУЧШЕНИЯ:
✅ Полный контекст (исходные сообщения + индексы)
✅ Temporal reasoning (используем индексы для определения свежести)
✅ Улучшенный промпт с четкими инструкциями
✅ Адаптивный NO_INFO порог
✅ Постобработка ответов

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
    Генератор ответов с temporal reasoning и улучшенным промптом.
    
    Логика:
    1. Проверка кэша
    2. Проверка на NO_INFO (по порогу score)
    3. Подготовка контекста с индексами
    4. Генерация через GigaChat (1 вызов!)
    5. Постобработка
    """
    
    def __init__(self, gigachat_model):
        self.gigachat = gigachat_model
        
        # Параметры генерации
        if SamplingParams is not None:
            self.sampling_params = SamplingParams(
                temperature=0.0,      # Детерминированная генерация
                max_tokens=100,       # Было 150, уменьшаем - нужен короткий ответ
                seed=42
            )
        else:
            self.sampling_params = None
        
        # Кэш ответов
        self.answer_cache: Dict[str, str] = {}
        
        # Статистика
        self.stats = {
            'gigachat_calls': 0,
            'cache_hits': 0,
            'no_info_responses': 0
        }
    
    def generate_answer(self, question: str,
                       search_results: List[SearchResult],
                       dialogue_id: str) -> str:
        """
        Генерация ответа - ОДИН вызов GigaChat с улучшенным контекстом!
        
        Args:
            question: Вопрос пользователя
            search_results: Результаты поиска (УЖЕ отсортированы reranker'ом!)
            dialogue_id: ID диалога
            
        Returns:
            Сгенерированный ответ
        """
        
        # 1. Проверка кэша
        cache_key = f"{dialogue_id}:{hash(question)}"
        if cache_key in self.answer_cache:
            self.stats['cache_hits'] += 1
            return self.answer_cache[cache_key]
        
        # 2. Проверка на NO_INFO
        if self._is_no_info(search_results):
            answer = "Эта информация не упоминалась в диалоге."
            self.answer_cache[cache_key] = answer
            self.stats['no_info_responses'] += 1
            return answer
        
        # 3. Подготовка контекста с индексами и исходными сообщениями
        context = self._prepare_context(search_results)
        
        # 4. Генерация промпта с инструкциями по temporal reasoning
        prompt = self._build_prompt(question, context)
        
        # 5. ЕДИНСТВЕННЫЙ вызов GigaChat
        answer = self._inference(prompt)
        self.stats['gigachat_calls'] += 1
        
        # 6. Постобработка
        answer = self._cleanup_answer(answer)
        
        # 7. Сохранение в кэш
        self.answer_cache[cache_key] = answer
        
        return answer
    
    def _is_no_info(self, results: List[SearchResult]) -> bool:
        """
        Определение отсутствия информации.
        
        Проверяем лучший score против порога NO_INFO_THRESHOLD
        """
        if not results:
            return True
        
        # Лучший score (results УЖЕ отсортированы reranker'ом!)
        best_score = results[0].final_score
        
        return best_score < NO_INFO_THRESHOLD
    
    def _prepare_context(self, results: List[SearchResult]) -> str:
        """
        Подготовка контекста с ПОЛНОЙ информацией.
        
        УЛУЧШЕНИЯ:
        - Включаем исходные user сообщения (original_user)
        - Добавляем индексы [#N] для temporal reasoning
        - Дедупликация по содержимому
        
        Формат:
        [#15] Я играю в футбол каждые выходные
        [#47] Еще занимаюсь пинг-понгом по вечерам
        [#102] Бросил волейбол, слишком травмоопасно
        """
        context_items = []
        seen = set()
        
        for result in results[:5]:  # Топ-5 результатов
            # Берем ОРИГИНАЛЬНОЕ user сообщение (без подтверждения ассистента)
            original_text = result.chunk.original_user or result.chunk.content
            
            # Нормализация для дедупликации
            normalized = original_text.lower().strip()
            
            if normalized not in seen and len(context_items) < 5:
                # Извлекаем индекс из метаданных (для temporal ordering)
                index = 0
                if result.chunk.metadata:
                    index = result.chunk.metadata.get('index', 0)
                
                # Формат: [#индекс] текст сообщения
                context_items.append(f"[#{index}] {original_text}")
                seen.add(normalized)
        
        return "\n".join(context_items)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """
        Создание промпта для GigaChat с улучшенными инструкциями.
        
        УЛУЧШЕНИЯ:
        - Инструкции по temporal reasoning (использовать свежие данные)
        - Четкие правила форматирования ответа
        - Явная обработка случаев без информации
        - Запрет на водные фразы
        """
        prompt = f"""На основе информации из диалога ответь на вопрос КРАТКО (одно предложение).

ИНФОРМАЦИЯ ИЗ ДИАЛОГА (отсортирована по релевантности):
{context}

ВАЖНЫЕ ПРАВИЛА:
1. Используй ТОЛЬКО информацию из диалога выше
2. Если информация противоречива, используй сообщение с БОЛЬШИМ индексом [#N] - оно более свежее
3. Ответь ОДНИМ коротким предложением (максимум 15-20 слов)
4. НЕ добавляй вводные фразы: "Согласно диалогу", "На основе информации", "Из диалога следует"
5. Начинай ответ сразу с сути
6. Если информации нет в диалоге - ответь: "Эта информация не упоминалась в диалоге."

ПРИМЕРЫ ХОРОШИХ ОТВЕТОВ:
Вопрос: Сколько мне лет?
Ответ: Вам 31 год.

Вопрос: Каким спортом я занимаюсь?
Ответ: Вы играете в футбол и пинг-понг.

Вопрос: Где я живу?
Ответ: Вы живете в Москве.

ВОПРОС: {question}

ОТВЕТ:"""
        
        return prompt
    
    def _cleanup_answer(self, answer: str) -> str:
        """
        Постобработка ответа.
        
        Операции:
        - Нормализация пробелов
        - Извлечение только первого предложения
        - Удаление водных фраз
        - Capitalize первой буквы
        - Удаление лишних знаков препинания
        """
        if not answer:
            return "Эта информация не упоминалась в диалоге."
        
        # Убираем лишние пробелы и переводы строк
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Только первое предложение
        match = re.search(r'[.!?]', answer)
        if match:
            answer = answer[:match.end()].strip()
        else:
            # Если нет знаков препинания, добавляем точку
            answer = answer.strip() + '.'
        
        # Убираем водные фразы в начале
        water_patterns = [
            r'^на основе (диалога|информации|сообщений),?\s*',
            r'^согласно (диалогу|информации|сообщениям),?\s*',
            r'^из диалога (следует|видно|понятно),?\s*',
            r'^судя по диалогу,?\s*',
            r'^в диалоге (говорится|упоминается|сказано),?\s*что\s*',
        ]
        
        for pattern in water_patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE).strip()
        
        # Capitalize первой буквы
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        
        # Убираем двойные знаки препинания
        answer = re.sub(r'\.{2,}', '.', answer)
        answer = re.sub(r'\?{2,}', '?', answer)
        answer = re.sub(r'!{2,}', '!', answer)
        
        # Убираем пробелы перед знаками препинания
        answer = re.sub(r'\s+([.,!?])', r'\1', answer)
        
        return answer
    
    def _inference(self, prompt: str) -> str:
        """
        Вызов GigaChat для генерации ответа.
        
        Использует vllm для эффективной генерации.
        """
        try:
            # Проверка инициализации модели
            if self.sampling_params is None or self.gigachat is None:
                return "Модель не инициализирована (vllm не установлен)."
            
            # Формируем сообщения для chat template
            messages = [
                Message(
                    role='system', 
                    content='Ты помощник, который отвечает на вопросы кратко и точно, используя только предоставленную информацию.'
                ),
                Message(
                    role='user', 
                    content=prompt
                )
            ]
            
            # Конвертируем в словари
            msg_dicts = [asdict(m) for m in messages]
            
            # Применяем chat template
            input_ids = self.gigachat.get_tokenizer().apply_chat_template(
                msg_dicts,
                add_generation_prompt=True
            )
            
            # Генерация
            outputs = self.gigachat.generate(
                prompt_token_ids=[input_ids],
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            
            # Извлекаем результат
            result = outputs[0].outputs[0].text
            return result.strip()
            
        except Exception as e:
            print(f"⚠️  Ошибка генерации: {e}")
            return "Не удалось сгенерировать ответ."
    
    def get_stats(self) -> Dict:
        """
        Получить статистику работы генератора.
        
        Returns:
            Словарь со статистикой
        """
        return {
            'gigachat_calls': self.stats['gigachat_calls'],
            'cache_hits': self.stats['cache_hits'],
            'cache_size': len(self.answer_cache),
            'no_info_responses': self.stats['no_info_responses'],
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['gigachat_calls'] + self.stats['cache_hits'])
        }
    
    def clear_cache(self):
        """Очистка кэша ответов"""
        self.answer_cache.clear()
        print("✓ Кэш AnswerBuilder очищен")