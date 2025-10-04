#!/usr/bin/env python3
"""
Тестирование SubmitModelWithMemory на небольшом диалоге.
"""

from src.submit import SubmitModelWithMemory
from src.submit.interfaces import Message

def test_model():
    """Тестирование модели на простом диалоге."""
    
    print("="*80)
    print("🧪 ТЕСТИРОВАНИЕ SubmitModelWithMemory")
    print("="*80)
    
    # Инициализация модели
    print("\n[1/4] Инициализация модели...")
    try:
        model = SubmitModelWithMemory(
            model_path="path/to/gigachat",
            weights_dir="./src/submit/weights"
        )
        print("✅ Модель инициализирована успешно!")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return False
    
    # Тестовый диалог
    print("\n[2/4] Время тестового диалога...")
    messages = [
        Message("user", "Меня зовут Иван"),
        Message("assistant", "Приятно познакомиться, Иван!"),
        Message("user", "Мне 30 лет"),
        Message("assistant", "Понятно, Ивану 30 лет. Как дела?"),
        Message("user", "Хорошо, работаю программистом")
    ]
    
    try:
        model.write_to_memory(messages, "test_dialogue")
        print("✅ Диалог записан в память!")
    except Exception as e:
        print(f"❌ Ошибка записи в память: {e}")
        return False
    
    # Тестирование вопросов
    print("\n[3/4] Тестирование вопросов...")
    
    test_questions = [
        "Как меня зовут?",
        "Сколько мне лет?",
        "Чем я занимаюсь?",
        "Что я сказал о своей работе?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n  Вопрос {i}: {question}")
        try:
            answer = model.answer_to_question("test_dialogue", question)
            print(f"  Ответ: {answer}")
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
    
    # Проверка статистики
    print("\n[4/4] Проверка статистики...")
    try:
        stats = model.data_processor.get_stats()
        print(f"  ✅ Отфильтровано длинных: {stats['filtered_too_long']}")
        print(f"  ✅ Отфильтровано ассистента: {stats['filtered_assistant']}")
        print(f"  ✅ Всего обработано чанков: {stats.get('total_chunks_processed', 0)}")
        print(f"  ✅ Диалогов в памяти: {len(model.data_processor.dialogue_chunks)}")
    except Exception as e:
        print(f"  ❌ Ошибка получения статистики: {e}")
    
    print("\n" + "="*80)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("="*80)
    
    return True


if __name__ == "__main__":
    test_model()
