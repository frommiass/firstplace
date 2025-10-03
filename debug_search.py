#!/usr/bin/env python3
"""
Скрипт для отладки поиска - ПРОВЕРЯЕМ КАКИЕ ЧАНКИ СОЗДАЮТСЯ
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
    print("🔍 ОТЛАДКА ПОИСКА - ПРОВЕРКА ЧАНКОВ")
    print("=" * 80)
    
    # Загрузка
    dialogues = load_dialogues_from_json("data/format_example.jsonl")
    dialogue = dialogues[0]
    dialogue_id = dialogue['id']
    question = dialogue['question']
    correct_answer = dialogue['ans']
    
    print(f"\nДиалог: {dialogue_id}")
    print(f"Вопрос: {question}")
    print(f"Правильный ответ: {correct_answer}")
    
    # Инициализация
    print(f"\n🤖 Инициализация модели...")
    model = SubmitModelWithMemory(
        model_path="dummy_path",
        weights_dir="src/submit/weights"
    )
    
    # Записываем в память
    print(f"\n📝 Запись в память...")
    for session in dialogue['sessions']:
        messages = convert_to_messages(session['messages'])
        model.write_to_memory(messages, dialogue_id)
    
    # Flush
    if dialogue_id in model.write_buffer:
        model._flush_buffer(dialogue_id)
    
    # ===================================================================
    # ПРОВЕРКА 1: Какие чанки создались?
    # ===================================================================
    print(f"\n{'='*80}")
    print(f"📊 АНАЛИЗ СОЗДАННЫХ ЧАНКОВ")
    print(f"{'='*80}")
    
    if dialogue_id in model.search_engine.dialogue_chunks:
        chunks = model.search_engine.dialogue_chunks[dialogue_id]
        print(f"\nВсего чанков: {len(chunks)}")
        
        # Ищем релевантные чанки вручную
        print(f"\n🔍 Поиск ключевых слов: футбол, пинг-понг, спорт")
        
        relevant_chunks = []
        for i, chunk in enumerate(chunks):
            content_lower = chunk.content.lower()
            
            # Ключевые слова
            keywords = ['футбол', 'пинг-понг', 'пинг', 'понг', 'спорт', 
                       'играю', 'занимаюсь', 'тренир']
            
            if any(kw in content_lower for kw in keywords):
                relevant_chunks.append((i, chunk))
        
        print(f"\nНайдено релевантных чанков вручную: {len(relevant_chunks)}")
        
        if relevant_chunks:
            print(f"\n📋 ТОП-10 РЕЛЕВАНТНЫХ ЧАНКОВ (по ключевым словам):")
            for rank, (idx, chunk) in enumerate(relevant_chunks[:10], 1):
                print(f"\n{rank}. Чанк #{idx}")
                print(f"   Роль: {chunk.role}")
                print(f"   Содержание: {chunk.content}")
                if chunk.metadata:
                    print(f"   Метаданные: index={chunk.metadata.get('index')}, "
                          f"has_fact={chunk.metadata.get('has_fact')}")
        else:
            print("\n❌ НЕ НАЙДЕНО РЕЛЕВАНТНЫХ ЧАНКОВ!")
            print("\nПоказываем первые 10 чанков:")
            for i, chunk in enumerate(chunks[:10], 1):
                print(f"\n{i}. {chunk.content[:100]}...")
    
    # ===================================================================
    # ПРОВЕРКА 2: Что находит SearchEngine?
    # ===================================================================
    print(f"\n{'='*80}")
    print(f"🔎 РЕЗУЛЬТАТЫ ПОИСКА ЧЕРЕЗ SEARCHENGINE")
    print(f"{'='*80}")
    
    # Поиск с разными top_k
    for top_k in [10, 50, 100]:
        print(f"\n🔍 Поиск с top_k={top_k}:")
        
        search_results = model.search_engine.search(
            question=question,
            dialogue_id=dialogue_id,
            top_k=top_k
        )
        
        print(f"   Найдено: {len(search_results)} результатов")
        
        if search_results:
            print(f"   Лучший score: {search_results[0].final_score:.4f}")
            print(f"   Худший score: {search_results[-1].final_score:.4f}")
            
            # Проверяем есть ли релевантные
            relevant_found = 0
            for result in search_results:
                content_lower = result.chunk.content.lower()
                if any(kw in content_lower for kw in ['футбол', 'пинг', 'спорт']):
                    relevant_found += 1
            
            print(f"   Релевантных: {relevant_found}/{len(search_results)}")
            
            # Топ-5
            if top_k == 50:
                print(f"\n   ТОП-5 результатов:")
                for i, result in enumerate(search_results[:5], 1):
                    is_relevant = any(kw in result.chunk.content.lower() 
                                     for kw in ['футбол', 'пинг', 'спорт'])
                    marker = "✓" if is_relevant else "✗"
                    print(f"   {i}. {marker} [score={result.final_score:.4f}] "
                          f"{result.chunk.content[:80]}...")
    
    # ===================================================================
    # ПРОВЕРКА 3: Адаптивный порог
    # ===================================================================
    print(f"\n{'='*80}")
    print(f"📊 АНАЛИЗ ПОРОГА NO_INFO")
    print(f"{'='*80}")
    
    search_results = model.search_engine.search(
        question=question,
        dialogue_id=dialogue_id,
        top_k=50
    )
    
    if search_results:
        threshold = model.search_engine.calculate_adaptive_threshold(search_results)
        best_score = search_results[0].final_score
        
        print(f"\nАдаптивный порог: {threshold:.4f}")
        print(f"Лучший score: {best_score:.4f}")
        print(f"Порог превышен: {'✓ ДА' if best_score >= threshold else '✗ НЕТ'}")
        
        # Статистика скоров
        scores = [r.final_score for r in search_results]
        import numpy as np
        print(f"\nСтатистика скоров:")
        print(f"   Mean: {np.mean(scores):.4f}")
        print(f"   Median: {np.median(scores):.4f}")
        print(f"   Std: {np.std(scores):.4f}")
        print(f"   Min: {np.min(scores):.4f}")
        print(f"   Max: {np.max(scores):.4f}")
    
    # ===================================================================
    # ПРОВЕРКА 4: Эмбеддинги вопроса
    # ===================================================================
    print(f"\n{'='*80}")
    print(f"🧮 АНАЛИЗ ЭМБЕДДИНГОВ")
    print(f"{'='*80}")
    
    # Проверяем эмбеддинги вопроса через каждый эмбеддер
    for emb_name, embedder in model.search_engine.embeddings.items():
        if embedder is None:
            continue
        
        try:
            question_emb = embedder.encode([question], show_progress_bar=False)[0]
            print(f"\n{emb_name}:")
            print(f"   Размерность: {question_emb.shape}")
            print(f"   Норма: {np.linalg.norm(question_emb):.4f}")
            print(f"   Min/Max: [{question_emb.min():.4f}, {question_emb.max():.4f}]")
        except Exception as e:
            print(f"\n{emb_name}: ❌ Ошибка - {e}")
    
    print(f"\n{'='*80}")
    print(f"✅ ОТЛАДКА ЗАВЕРШЕНА")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
