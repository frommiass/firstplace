#!/usr/bin/env python3
"""
Простой скрипт для обработки диалогов - РАБОЧАЯ ВЕРСИЯ
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

def main():
    """Основная функция скрипта."""
    print("🚀 РАБОЧИЙ СКРИПТ ОБРАБОТКИ ДИАЛОГОВ")
    print("=" * 50)
    
    # Параметры
    input_file = "data/format_example.jsonl"
    output_dir = Path("output_working")
    
    # Создаем директорию
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Результаты: {output_dir}")
    
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
    
    # Инициализация модели
    print(f"\n🤖 Инициализация модели...")
    model = SubmitModelWithMemory(
        model_path="dummy_path",
        weights_dir="src/submit/weights"
    )
    print("✓ Модель инициализирована")
    
    # Обрабатываем первые 3 диалога
    dialogues_to_process = dialogues[:3]
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
            
            # Генерация ответа
            print(f"\n[3/3] Генерация ответа...")
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
            
            output_file = output_dir / f"dialogue_{dialogue_id}_working.txt"
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

if __name__ == "__main__":
    main()
