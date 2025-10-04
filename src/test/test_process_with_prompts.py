#!/usr/bin/env python3
"""
Скрипт для обработки диалогов с записью промтов в файлы вместо GigaChat
"""
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Отключаем GPU для стабильной работы
os.environ['CUDA_VISIBLE_DEVICES'] = ''
print("🔧 GPU отключен для стабильной работы")

# Добавляем путь к модулям
sys.path.append('src')

from submit.interfaces import Message
from submit.model_inference import SubmitModelWithMemory

def load_dialogues_from_json(file_path: str) -> List[Dict[str, Any]]:
    """Загружает диалоги из JSONL файла."""
    dialogues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                if isinstance(data, dict) and 'question' in data and 'sessions' in data:
                    dialogues.append(data)
            except json.JSONDecodeError as e:
                print(f"⚠️  Ошибка парсинга строки {line_num}: {e}")
                continue


    print(f"✓ Загружено {len(dialogues)} диалогов")
    return dialogues

def convert_to_messages(dialogue_messages: List[Dict]) -> List[Message]:
    """Конвертирует сообщения диалога в объекты Message."""
    messages = []
    for msg in dialogue_messages:
        message = Message(
            content=msg.get('content', ''),
            role=msg.get('role', 'user'),
            session_id=str(msg.get('session_id', 'unknown'))
        )
        messages.append(message)
    
    return messages

class PromptCaptureModel(SubmitModelWithMemory):
    """Модель с захватом промтов вместо их отправки в GigaChat"""
    
    def __init__(self, model_path: str, weights_dir: str = "./weights", prompt_dir: str = "./output_prompts"):
        super().__init__(model_path, weights_dir)
        self.prompt_dir = Path(prompt_dir)
        self.prompt_dir.mkdir(exist_ok=True)
        self.prompt_counter = 0
    
    def get_answer_builder(self):
        """Возвращаем модифицированный AnswerBuilder который захватывает промты"""
        from submit.answer_builder import AnswerBuilder
        return PromptCaptureAnswerBuilder(None, self.prompt_dir)

class PromptCaptureAnswerBuilder:
    """Захватывает промты вместо отправки в GigaChat"""
    
    def __init__(self, gigachat_model, prompt_dir: Path):
        self.gigachat = gigachat_model
        self.prompt_dir = prompt_dir
        self.prompt_counter = 0
    
    def generate_answer(self, question: str, search_results, dialogue_id: str) -> str:
        """Генерирует ответ с захватом промта"""
        from submit.interfaces import SearchResult
        from submit.answer_builder import NO_INFO_THRESHOLD
        
        # Проверка на NO_INFO
        if self._is_no_info(search_results):
            answer = "Эта информация не упоминалась в диалоге."
            return answer
        
        # Подготовка контекста  
        context = self._prepare_context(search_results)
        
        # Создание промта
        prompt = self._build_prompt(question, context)
        
        # Сохранение промта в файл
        prompt_file = self.prompt_dir / f"prompt_{dialogue_id}_{self.prompt_counter}.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        self.prompt_counter += 1
        print(f"📝 Промт сохранен: {prompt_file}")
        
        # Возвращаем простое сообщение вместо реального ответа
        return f"Промт записан в файл {prompt_file}"
    
    def _is_no_info(self, results):
        """Проверяет достаточно ли информации"""
        if not results:
            return True
        
        best_score = results[0].final_score
        return best_score < NO_INFO_THRESHOLD
        
    def _prepare_context(self, results):
        """Подготовка контекста"""
        context_items = []
        seen = set()
        
        for result in results[:5]:
            original_text = result.chunk.original_user or result.chunk.content
            
            normalized = original_text.lower().strip()
            
            if normalized not in seen and len(context_items) < 5:
                index = 0
                if result.chunk.metadata:
                    index = result.chunk.metadata.get('index', 0)
                
                context_items.append(f"[#{index}] {original_text}")
                seen.add(normalized)
        
        return "\n".join(context_items)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Создание промта"""
        prompt = f"""Информация из диалога:
{context}

ПРАВИЛА:
1. Используй ТОЛЬКО информацию выше
2. При противоречиях - бери сообщение с БОЛЬШИМ индексом [#N]
3. Ответь КРАТКО одним предложением (макс. 15 слов)
4. БЕЗ вводных фраз: "Согласно", "На основе", "Из диалога"
5. Начинай сразу с сути
6. Если информации нет - "Эта информация не упоминалась в диалоге."

ПРИМЕРЫ:
Вопрос: Сколько мне лет?
Ответ: Вам 31 год.

Вопрос: Где я живу?
Ответ: Вы живете в Москве.

ВОПРОС: {question}
ОТВЕТ:"""
        
        return prompt

def main():
    """Основная функция скрипта."""
    print("🚀 СКРИПТ С ЗАХВАТОМ ПРОМТОВ")
    print("=" * 60)
    
    # Параметры
    input_file = "data/format_example.jsonl"
    output_dir = Path("output_simple")
    prompt_dir = Path("output_prompts")
    
    # Создаем директории
    output_dir.mkdir(exist_ok=True)
    prompt_dir.mkdir(exist_ok=True)
    print(f"📁 Результаты: {output_dir}")
    print(f"📁 Промты: {prompt_dir}")
    
    # Проверяем файл
    if not os.path.exists(input_file):
        print(f"❌ Файл {input_file} не найден!")
        return
    
    # Загружаем диалоги
    print(f"\n📖 Загрузка диалогов...")
    dialogues = load_dialogues_from_json(input_file)
    
    if not dialogues:
        print("❌ Не удалось загрузить диалоги!")
        return
    
    # Инициализация модели с захватом промтов
    print(f"\n🤖 Инициализация модели с захватом промтов...")
    model = PromptCaptureModel(
        model_path="dummy_path",
        weights_dir="src/submit/weights",
        prompt_dir=prompt_dir
    )
    print("✓ Модель инициализирована")
    
    # Обрабатываем первые 4 диалога
    dialogues_to_process = dialogues[:4]
    print(f"\n🔄 Обработка {len(dialogues_to_process)} диалогов...")
    
    for i, dialogue in enumerate(dialogues_to_process, 1):
        print(f"\n{'='*60}")
        print(f"ДИАЛОГ {i}/{len(dialogues_to_process)}: {dialogue['id']}")
        print(f"{'='*60}")
        
        dialogue_id = dialogue['id']
        question = dialogue['question']
        correct_answer = dialogue['ans']
        
        print(f"Вопрос: {question}")
        print(f"Правильный ответ: {correct_answer}")
        
        try:
            # Записываем в память
            print(f"\n[1/3] Запись в память...")
            total_messages = 0
            
            for session in dialogue['sessions']:
                messages = convert_to_messages(session['messages'])
                model.write_to_memory(messages, dialogue_id)
                total_messages += len(messages)
            
            print(f"✓ Записано {total_messages} сообщений")
            
            # Поиск
            print(f"\n[2/3] Поиск...")
            search_results = model.search_engine.search(
                question=question,
                dialogue_id=dialogue_id,
                top_k=10
            )
            
            print(f"✓ Найдено {len(search_results)} результатов")
            
            # Показываем топ-3 результата
            if search_results:
                print(f"\n📋 Топ-3 результата:")
                for j, result in enumerate(search_results[:3], 1):
                    print(f"   {j}. Score: {result.final_score:.3f}")
                    print(f"      Content: {result.chunk.content[:80]}...")
            
            # Генерация ответа (с захватом промта)
            print(f"\n[3/3] Генерация ответа (с захватом промта)...")
            try:
                generated_answer = model.answer_to_question(dialogue_id, question)
                print(f"✓ Ответ: {generated_answer}")
            except Exception as e:
                print(f"❌ Ошибка генерации: {e}")
                generated_answer = f"Ошибка: {e}"
            
            # Сохраняем результат
            result_text = f"Диалог {dialogue_id}:\n"
            result_text += f"Вопрос: {question}\n"
            result_text += f"Правильный ответ: {correct_answer}\n"
            result_text += f"Сгенерированный ответ: {generated_answer}\n"
            result_text += f"Найдено результатов: {len(search_results)}\n\n"
            
            if search_results:
                result_text += "ТОП-10 РЕЗУЛЬТАТОВ:\n"
                for j, result in enumerate(search_results[:10], 1):
                    result_text += f"{j}. Score: {result.final_score:.3f}\n"
                    result_text += f"   Content: {result.chunk.content[:100]}...\n\n"
            
            output_file = output_dir / f"dialogue_{dialogue_id}_simple.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result_text)
            
            print(f"✓ Результат сохранен: {output_file}")
            
            # Очистка памяти
            print(f"\n🧹 Очистка памяти...")
            model.clear_memory(dialogue_id)
            print(f"✓ Память очищена")
            
        except Exception as e:
            print(f"❌ Ошибка обработки диалога {dialogue_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
    print(f"📁 Результаты в: {output_dir}")
    print(f"📝 Промты сохранены в: {prompt_dir}")
    
    # Показываем созданные файлы
    print(f"\n📋 Созданные промты:")
    for prompt_file in sorted(prompt_dir.glob("*.txt")):
        print(f"   {prompt_file.name}")

if __name__ == "__main__":
    main()
