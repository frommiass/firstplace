#!/usr/bin/env python3
"""
Скрипт для обработки диалогов из JSON файла.

Функции:
1. Загружает диалоги из JSON файла
2. Пропускает каждый диалог через SubmitModelWithMemory.write_to_memory()
3. Извлекает промпты через метод extract()
4. Сохраняет промпты в отдельные файлы
5. Очищает память после каждого диалога

БЕЗ ВЫЗОВА GIGACHAT!
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Добавляем путь к модулям
sys.path.append('src')

from submit.interfaces import Message
from submit.model_inference import SubmitModelWithMemory
from submit.data_processor import DataProcessor
from submit.search_engine import SearchEngine


def load_dialogues_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Загружает диалоги из JSONL файла.
    
    Args:
        file_path: Путь к JSONL файлу
        
    Returns:
        Список диалогов с вопросами
    """
    try:
        dialogues = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Каждая строка - это один диалог с вопросом
                    if isinstance(data, dict) and 'question' in data and 'sessions' in data:
                        dialogues.append({
                            'id': data['id'],
                            'question': data['question'],
                            'answer': data['ans'],
                            'question_type': data['question_type'],
                            'sessions': data['sessions'],
                            'ans_session_ids': data.get('ans_session_ids', [])
                        })
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️  Ошибка парсинга строки {line_num}: {e}")
                    continue
        
        print(f"✓ Загружено {len(dialogues)} диалогов")
        return dialogues
        
    except Exception as e:
        print(f"❌ Ошибка загрузки файла {file_path}: {e}")
        return []


def convert_to_messages(dialogue_messages: List[Dict]) -> List[Message]:
    """
    Конвертирует сообщения диалога в объекты Message.
    
    Args:
        dialogue_messages: Список сообщений из JSON
        
    Returns:
        Список объектов Message
    """
    messages = []
    for msg in dialogue_messages:
        message = Message(
            content=msg.get('content', ''),
            role=msg.get('role', 'user'),
            session_id=str(msg.get('session_id', 'unknown'))
        )
        messages.append(message)
    
    return messages


def process_dialogue(model, dialogue: Dict[str, Any], 
                    output_dir: Path) -> None:
    """
    Обрабатывает один диалог с вопросом.
    
    Args:
        model: Экземпляр SubmitModelWithMemory
        dialogue: Данные диалога с вопросом
        output_dir: Директория для сохранения результатов
    """
    dialogue_id = dialogue['id']
    question = dialogue['question']
    answer = dialogue['answer']
    question_type = dialogue['question_type']
    sessions = dialogue['sessions']
    
    print(f"\n📝 Обработка диалога {dialogue_id}")
    print(f"   Вопрос: {question}")
    print(f"   Тип вопроса: {question_type}")
    print(f"   Количество сессий: {len(sessions)}")
    
    try:
        # 1. Записываем все сессии в память
        print(f"   → Запись всех сессий в память...")
        total_messages = 0
        
        for session in sessions:
            session_id = session['id']
            messages = convert_to_messages(session['messages'])
            model.write_to_memory(messages, f"{dialogue_id}_{session_id}")
            total_messages += len(messages)
        
        print(f"   ✓ Записано {total_messages} сообщений из {len(sessions)} сессий")
        
        # 2. Генерируем ответ на вопрос через SubmitModelWithMemory
        print(f"   → Генерация ответа на вопрос через модель...")
        
        try:
            # Вызываем answer_to_question для получения ответа
            generated_answer = model.answer_to_question(dialogue_id, question)
            print(f"   ✓ Ответ сгенерирован: {generated_answer[:100]}...")
            
        except Exception as e:
            print(f"   ⚠️  Ошибка генерации ответа: {e}")
            generated_answer = f"Ошибка генерации: {e}"
        
        # Создаем результат с реальным ответом
        result_text = f"Диалог {dialogue_id}:\n"
        result_text += f"Вопрос: {question}\n"
        result_text += f"Тип вопроса: {question_type}\n"
        result_text += f"Правильный ответ: {answer}\n"
        result_text += f"Сгенерированный ответ: {generated_answer}\n\n"
        result_text += f"Контекст из {len(sessions)} сессий:\n"
        
        # Добавляем краткий контекст из сессий
        for session in sessions[:5]:  # Показываем только первые 5 сессий
            session_id = session['id']
            messages = convert_to_messages(session['messages'])
            result_text += f"\n--- Сессия {session_id} ---\n"
            for msg in messages[:3]:  # Показываем только первые 3 сообщения
                result_text += f"{msg.role}: {msg.content[:100]}...\n"
        
        if len(sessions) > 5:
            result_text += f"\n... и еще {len(sessions) - 5} сессий\n"
        
        # 3. Сохраняем результат в файл
        output_file = output_dir / f"dialogue_{dialogue_id}_question_answer.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_text)
        
        print(f"   ✓ Результат сохранен в {output_file}")
        
        # 4. Очищаем память
        print(f"   → Очистка памяти...")
        for session in sessions:
            session_id = session['id']
            model.clear_memory(f"{dialogue_id}_{session_id}")
        
        print(f"   ✓ Память очищена")
        
    except Exception as e:
        print(f"   ❌ Ошибка обработки диалога {dialogue_id}: {e}")


def main():
    """Основная функция скрипта."""
    print("🚀 СКРИПТ ОБРАБОТКИ ДИАЛОГОВ")
    print("=" * 50)
    
    # Параметры
    input_file = "data/format_example.jsonl"
    output_dir = Path("output_prompts")
    
    # Создаем директорию для результатов
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Результаты будут сохранены в: {output_dir}")
    
    # Проверяем существование входного файла
    if not os.path.exists(input_file):
        print(f"❌ Файл {input_file} не найден!")
        return
    
    # Загружаем диалоги
    print(f"\n📖 Загрузка диалогов из {input_file}...")
    dialogues = load_dialogues_from_json(input_file)
    
    if not dialogues:
        print("❌ Не удалось загрузить диалоги!")
        return
    
    # Создаем модель С ВЕСАМИ (но без GigaChat)
    print(f"\n🤖 Инициализация модели с весами...")
    
    
    try:
        # Пытаемся создать реальную модель с весами
        print("   → Попытка инициализации SubmitModelWithMemory...")
        model = SubmitModelWithMemory(
            model_path="dummy_path",  # Заглушка для пути модели
            weights_dir="src/submit/weights"
        )
        print("✓ SubmitModelWithMemory инициализирована с весами")
        
    except Exception as e:
        print(f"❌ Ошибка инициализации SubmitModelWithMemory: {e}")
        print("💡 Убедитесь, что все зависимости установлены:")
        print("   pip install faiss-cpu sentence-transformers rank-bm25 transformers")
        print("\n🔄 Создаем упрощенную версию без весов для демонстрации...")
        
        # Создаем упрощенную версию без весов
        class SimpleModelWithoutWeights:
            """Упрощенная версия без весов для демонстрации."""
            
            def __init__(self):
                self.basic_memory = {}
                print("✓ Упрощенная модель создана (без весов)")
            
            def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
                """Записывает сообщения в память."""
                self.basic_memory[dialogue_id] = messages
                print(f"     ✓ Записано {len(messages)} сообщений (без обработки весами)")
            
            def answer_to_question(self, dialogue_id: str, question: str) -> str:
                """Заглушка для answer_to_question."""
                return f"[ЗАГЛУШКА] Ответ на вопрос '{question}' для диалога {dialogue_id}. Для реального ответа нужны веса и GigaChat."
            
            def clear_memory(self, dialogue_id: str) -> None:
                """Очищает память."""
                if dialogue_id in self.basic_memory:
                    del self.basic_memory[dialogue_id]
                print(f"     ✓ Память очищена")
        
        model = SimpleModelWithoutWeights()
    
    # Обрабатываем каждый диалог
    print(f"\n🔄 Начинаем обработку {len(dialogues)} диалогов...")
    
    for i, dialogue in enumerate(dialogues, 1):
        print(f"\n{'='*60}")
        print(f"ДИАЛОГ {i}/{len(dialogues)}")
        print(f"{'='*60}")
        
        process_dialogue(model, dialogue, output_dir)
    
    print(f"\n🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
    print(f"📁 Результаты сохранены в директории: {output_dir}")
    print(f"📊 Обработано диалогов: {len(dialogues)}")
    
    # Показываем созданные файлы
    output_files = list(output_dir.glob("*.txt"))
    print(f"📄 Создано файлов: {len(output_files)}")
    
    if output_files:
        print("\n📋 Созданные файлы:")
        for file in sorted(output_files):
            print(f"   • {file.name}")


if __name__ == "__main__":
    main()
