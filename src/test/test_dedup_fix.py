#!/usr/bin/env python3
"""
Быстрый тест исправления дедупликации
"""
import json
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.append('src')

from submit.interfaces import Message
from submit.model_inference import SubmitModelWithMemory

def load_dialogues_from_json(file_path: str):
    dialogues = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and 'question' in data and 'sessions' in data:
                    dialogues.append(data)
            except:
                continue
    return dialogues

def convert_to_messages_safe(dialogue_messages):
    """
    Безопасная конвертация сообщений в объекты Message.
    
    ИСПРАВЛЕНО: content может быть float, int, None
    """
    messages = []
    for msg in dialogue_messages:
        # Безопасная конвертация content
        content = msg.get('content', '')
        if content is None:
            content = ''
        content = str(content)  # Конвертируем любой тип в строку
        
        message = Message(
            content=content,
            role=msg.get('role', 'user'),
            session_id=str(msg.get('session_id', 'unknown'))
        )
        messages.append(message)
    
    return messages

def main():
    print("🧪 ТЕСТ ИСПРАВЛЕНИЯ ДЕДУПЛИКАЦИИ")
    print("=" * 60)
    
    # Загрузка
    dialogues = load_dialogues_from_json("data/format_example.jsonl")
    dialogue = dialogues[1]  # Второй диалог
    dialogue_id = dialogue['id']
    question = dialogue['question']
    correct_answer = dialogue['ans']
    
    print(f"\nДиалог: {dialogue_id}")
    print(f"Вопрос: {question}")
    print(f"Правильный ответ: {correct_answer}")
    print(f"Сессий: {len(dialogue['sessions'])}")
    
    # Подсчет общего количества сообщений
    total_messages = sum(len(s['messages']) for s in dialogue['sessions'])
    print(f"Всего сообщений: {total_messages}")
    
    # Инициализация
    print(f"\n🤖 Инициализация модели...")
    model = SubmitModelWithMemory(
        model_path="dummy_path",
        weights_dir="src/submit/weights"
    )
    
    # Записываем в память
    print(f"\n📝 Запись в память...")
    for i, session in enumerate(dialogue['sessions'], 1):
        messages = convert_to_messages(session['messages'])
        print(f"   Сессия {i}/{len(dialogue['sessions'])}: {len(messages)} сообщений")
        model.write_to_memory(messages, dialogue_id)
    
    # Flush
    print(f"\n🔄 Flush буфера...")
    if dialogue_id in model.write_buffer:
        model._flush_buffer(dialogue_id)
    
    # Проверяем сколько чанков создалось
    print(f"\n{'='*60}")
    print(f"📊 РЕЗУЛЬТАТЫ")
    print(f"{'='*60}")
    
    if dialogue_id in model.search_engine.dialogue_chunks:
        chunks = model.search_engine.dialogue_chunks[dialogue_id]
        print(f"\n✅ Создано чанков: {len(chunks)}")
        print(f"   Сообщений было: {total_messages}")
        print(f"   Процент сохранения: {len(chunks)/total_messages*100:.1f}%")
        
        # Ищем релевантные
        keywords = ['футбол', 'пинг-понг', 'пинг', 'понг', 'спорт']
        relevant = [c for c in chunks if any(kw in c.content.lower() for kw in keywords)]
        
        print(f"\n🔍 Релевантных чанков: {len(relevant)}")
        
        if relevant:
            print(f"\n📋 РЕЛЕВАНТНЫЕ ЧАНКИ:")
            for i, chunk in enumerate(relevant[:5], 1):
                print(f"\n{i}. {chunk.content}")
        
        # Поиск
        print(f"\n🔎 Поиск...")
        search_results = model.search_engine.search(
            question=question,
            dialogue_id=dialogue_id,
            top_k=10
        )
        
        print(f"   Найдено результатов: {len(search_results)}")
        
        if search_results:
            print(f"   Лучший score: {search_results[0].final_score:.4f}")
            
            # Топ-3
            print(f"\n   Топ-3:")
            for i, result in enumerate(search_results[:3], 1):
                is_relevant = any(kw in result.chunk.content.lower() for kw in keywords)
                marker = "✓" if is_relevant else "✗"
                print(f"   {i}. {marker} [score={result.final_score:.4f}] "
                      f"{result.chunk.content[:80]}...")
            
            # Генерация
            print(f"\n🤖 Генерация ответа...")
            answer = model.answer_to_question(dialogue_id, question)
            print(f"   Сгенерированный: {answer}")
            print(f"   Правильный: {correct_answer}")
    else:
        print(f"\n❌ Чанки не найдены для диалога {dialogue_id}")
    
    print(f"\n{'='*60}")
    
    # Статистика DataProcessor
    if hasattr(model.data_processor, 'get_stats'):
        dp_stats = model.data_processor.get_stats()
        print(f"📊 DataProcessor статистика:")
        print(f"   Обработано: {dp_stats['total_processed']}")
        print(f"   Чанков: {dp_stats['total_chunks']}")
        print(f"   Ошибок: {dp_stats['errors']}")

if __name__ == "__main__":
    main()
