#!/usr/bin/env python3
"""
Улучшенный скрипт для обработки диалогов с полным использованием SearchEngine

УЛУЧШЕНИЯ:
1. ✅ Использует улучшенный SearchEngine с weighted ensemble
2. ✅ Показывает top-k результатов поиска с их scores
3. ✅ Сравнивает сгенерированный ответ с правильным
4. ✅ Выводит статистику SearchEngine
5. ✅ Показывает адаптивный порог NO_INFO
6. ✅ Диагностика качества поиска
7. ✅ Сохраняет детальный отчет
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import time

# Отключаем GPU для стабильной работы
os.environ['CUDA_VISIBLE_DEVICES'] = ''
print("🔧 GPU отключен для стабильной работы")

# Добавляем путь к модулям
sys.path.append('src')

from submit.interfaces import Message
from submit.model_inference import SubmitModelWithMemory


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


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Простая мера схожести между двумя текстами (Jaccard similarity).
    
    Args:
        text1: Первый текст
        text2: Второй текст
        
    Returns:
        Коэффициент схожести (0-1)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def process_dialogue(model: SubmitModelWithMemory, 
                     dialogue: Dict[str, Any], 
                     output_dir: Path) -> Dict[str, Any]:
    """
    Обрабатывает один диалог с детальным анализом.
    
    Args:
        model: Экземпляр SubmitModelWithMemory
        dialogue: Данные диалога с вопросом
        output_dir: Директория для сохранения результатов
        
    Returns:
        Статистика обработки
    """
    dialogue_id = dialogue['id']
    question = dialogue['question']
    correct_answer = dialogue['answer']
    question_type = dialogue['question_type']
    sessions = dialogue['sessions']
    
    print(f"\n{'='*80}")
    print(f"📝 Диалог {dialogue_id}")
    print(f"{'='*80}")
    print(f"Вопрос: {question}")
    print(f"Тип: {question_type}")
    print(f"Правильный ответ: {correct_answer}")
    print(f"Сессий: {len(sessions)}")
    
    stats = {
        'dialogue_id': dialogue_id,
        'question_type': question_type,
        'success': False,
        'similarity': 0.0,
        'search_time': 0.0,
        'generation_time': 0.0,
        'total_messages': 0
    }
    
    try:
        # ===================================================================
        # 1. ЗАПИСЬ В ПАМЯТЬ
        # ===================================================================
        print(f"\n[1/4] Запись сессий в память...")
        total_messages = 0
        
        for session in sessions:
            session_id = session['id']
            messages = convert_to_messages(session['messages'])
            
            # Используем единый dialogue_id для всех сессий
            model.write_to_memory(messages, dialogue_id)
            total_messages += len(messages)
        
        stats['total_messages'] = total_messages
        print(f"✓ Записано {total_messages} сообщений из {len(sessions)} сессий")
        
        # Принудительный flush
        if dialogue_id in model.write_buffer and model.write_buffer[dialogue_id]:
            model._flush_buffer(dialogue_id)
            print(f"✓ Буфер обработан")
        
        # ===================================================================
        # 2. ПОИСК РЕЛЕВАНТНЫХ ФРАГМЕНТОВ
        # ===================================================================
        print(f"\n[2/4] Поиск релевантных фрагментов...")
        search_start = time.time()
        
        # Используем внутренний SearchEngine для получения результатов
        search_results = model.search_engine.search(
            question=question,
            dialogue_id=dialogue_id,
            top_k=20
        )
        
        search_time = time.time() - search_start
        stats['search_time'] = search_time
        
        print(f"✓ Найдено {len(search_results)} результатов за {search_time:.3f}s")
        
        # Вычисляем адаптивный порог
        if search_results:
            adaptive_threshold = model.search_engine.calculate_adaptive_threshold(search_results)
            print(f"📊 Адаптивный порог NO_INFO: {adaptive_threshold:.3f}")
            
            # Показываем топ-5 результатов
            print(f"\n📋 Топ-5 результатов поиска:")
            for i, result in enumerate(search_results[:5], 1):
                content_preview = result.chunk.content[:80]
                print(f"   {i}. [score={result.final_score:.3f}] {content_preview}...")
        
        # ===================================================================
        # 3. ГЕНЕРАЦИЯ ОТВЕТА
        # ===================================================================
        print(f"\n[3/4] Генерация ответа...")
        generation_start = time.time()
        
        try:
            generated_answer = model.answer_to_question(dialogue_id, question)
            generation_time = time.time() - generation_start
            stats['generation_time'] = generation_time
            
            print(f"✓ Ответ сгенерирован за {generation_time:.3f}s")
            print(f"🤖 Сгенерированный ответ: {generated_answer}")
            
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            generated_answer = f"Ошибка: {e}"
            generation_time = time.time() - generation_start
            stats['generation_time'] = generation_time
        
        # ===================================================================
        # 4. СРАВНЕНИЕ С ПРАВИЛЬНЫМ ОТВЕТОМ
        # ===================================================================
        print(f"\n[4/4] Анализ качества...")
        
        similarity = calculate_similarity(generated_answer, correct_answer)
        stats['similarity'] = similarity
        
        # Простая эвристика успеха
        is_correct = similarity > 0.3 or generated_answer.lower() in correct_answer.lower()
        stats['success'] = is_correct
        
        print(f"📊 Схожесть с правильным ответом: {similarity:.2%}")
        print(f"✓ Статус: {'✅ ПРАВИЛЬНО' if is_correct else '❌ НЕПРАВИЛЬНО'}")
        
        # ===================================================================
        # 5. СОЗДАНИЕ ДЕТАЛЬНОГО ОТЧЕТА
        # ===================================================================
        report = []
        report.append("=" * 80)
        report.append(f"ОТЧЕТ ПО ДИАЛОГУ {dialogue_id}")
        report.append("=" * 80)
        report.append("")
        
        report.append(f"Вопрос: {question}")
        report.append(f"Тип вопроса: {question_type}")
        report.append("")
        
        report.append("ОТВЕТЫ:")
        report.append(f"✓ Правильный: {correct_answer}")
        report.append(f"🤖 Сгенерированный: {generated_answer}")
        report.append(f"📊 Схожесть: {similarity:.2%}")
        report.append(f"📈 Статус: {'✅ ПРАВИЛЬНО' if is_correct else '❌ НЕПРАВИЛЬНО'}")
        report.append("")
        
        report.append("СТАТИСТИКА:")
        report.append(f"• Сообщений в памяти: {total_messages}")
        report.append(f"• Найдено результатов: {len(search_results)}")
        report.append(f"• Время поиска: {search_time:.3f}s")
        report.append(f"• Время генерации: {generation_time:.3f}s")
        report.append(f"• Общее время: {search_time + generation_time:.3f}s")
        
        if search_results:
            report.append(f"• Адаптивный порог: {adaptive_threshold:.3f}")
            report.append(f"• Лучший score: {search_results[0].final_score:.3f}")
        
        report.append("")
        report.append("=" * 80)
        report.append(f"ТОП-{min(20, len(search_results))} РЕЛЕВАНТНЫХ ФРАГМЕНТОВ")
        report.append("=" * 80)
        report.append("")
        
        for i, result in enumerate(search_results[:20], 1):
            report.append(f"[{i}] Score: {result.final_score:.4f}")
            report.append(f"    Исходный score: {result.score:.4f}")
            report.append(f"    Содержание: {result.chunk.content}")
            report.append(f"    Роль: {result.chunk.role}")
            report.append(f"    Session: {result.chunk.session_id}")
            
            if result.chunk.metadata:
                report.append(f"    Метаданные: {result.chunk.metadata}")
            
            report.append("")
        
        report.append("=" * 80)
        report.append("СТАТИСТИКА SEARCHENGINE")
        report.append("=" * 80)
        report.append("")
        
        search_stats = model.search_engine.get_stats()
        report.append(f"Поиски: {search_stats['search']['total_searches']}")
        report.append(f"Средн. результатов: {search_stats['search']['avg_results']:.1f}")
        report.append(f"Reranker вызовов: {search_stats['search']['reranker_calls']}")
        report.append(f"Hit rate кэша: {search_stats['cache']['hit_rate']*100:.1f}%")
        report.append(f"Эмбеддеров: {search_stats['models']['embedders']}/3")
        report.append(f"Reranker: {'✓' if search_stats['models']['reranker'] else '✗'}")
        report.append("")
        
        # Сохраняем отчет
        output_file = output_dir / f"dialogue_{dialogue_id}_report.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n💾 Отчет сохранен: {output_file}")
        
        # ===================================================================
        # 6. ОЧИСТКА ПАМЯТИ
        # ===================================================================
        print(f"\n🧹 Очистка памяти...")
        model.clear_memory(dialogue_id)
        print(f"✓ Память очищена")
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        stats['success'] = False
    
    return stats


def main():
    """Основная функция скрипта."""
    try:
        print("=" * 80)
        print("🚀 УЛУЧШЕННЫЙ СКРИПТ ОБРАБОТКИ ДИАЛОГОВ")
        print("=" * 80)
        print("\nФункции:")
        print("• Weighted ensemble поиск (3 эмбеддера + BM25 + reranker)")
        print("• Адаптивные пороги NO_INFO")
        print("• Детальная статистика SearchEngine")
        print("• Сравнение с правильными ответами")
        print("• Детальные отчеты по каждому диалогу")
        print("=" * 80)
        
        # ===================================================================
        # ПАРАМЕТРЫ
        # ===================================================================
    input_file = "data/format_example.jsonl"
    output_dir = Path("output_reports")
    model_path = "path/to/gigachat"  # Путь к GigaChat модели
    weights_dir = "src/submit/weights"
    
    # Создаем директорию
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Отчеты: {output_dir}")
    
    # Проверяем файл
    if not os.path.exists(input_file):
        print(f"❌ Файл {input_file} не найден!")
        return
    
    # ===================================================================
    # ЗАГРУЗКА ДИАЛОГОВ
    # ===================================================================
    print(f"\n📖 Загрузка диалогов из {input_file}...")
    dialogues = load_dialogues_from_json(input_file)
    
    if not dialogues:
        print("❌ Не удалось загрузить диалоги!")
        return
    
    # ===================================================================
    # ИНИЦИАЛИЗАЦИЯ МОДЕЛИ
    # ===================================================================
    print(f"\n🤖 Инициализация SubmitModelWithMemory...")
    print(f"   Model path: {model_path}")
    print(f"   Weights dir: {weights_dir}")
    
    try:
        model = SubmitModelWithMemory(
            model_path=model_path,
            weights_dir=weights_dir
        )
        print("✓ Модель инициализирована")
        
        # Показываем статистику моделей
        model.search_engine.print_stats()
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        print("\nВозможные причины:")
        print("1. Отсутствуют веса моделей в src/submit/weights/")
        print("2. Не установлены зависимости:")
        print("   pip install torch transformers sentence-transformers")
        print("   pip install faiss-cpu rank-bm25 vllm")
        return
    
    # ===================================================================
    # ОБРАБОТКА ДИАЛОГОВ
    # ===================================================================
    # ОГРАНИЧИВАЕМ ДЛЯ ОТЛАДКИ
    dialogues_to_process = dialogues[:1]  # Только первый диалог
    print(f"\n🔄 Начинаем обработку {len(dialogues_to_process)} диалогов (ограничено для отладки)...")
    
    all_stats = []
    start_time = time.time()
    
    for i, dialogue in enumerate(dialogues_to_process, 1):
        print(f"\n{'='*80}")
        print(f"ПРОГРЕСС: {i}/{len(dialogues)} ({i/len(dialogues)*100:.1f}%)")
        print(f"{'='*80}")
        
        stats = process_dialogue(model, dialogue, output_dir)
        all_stats.append(stats)
        
        # Промежуточная статистика каждые 10 диалогов
        if i % 10 == 0:
            correct = sum(1 for s in all_stats if s['success'])
            accuracy = correct / len(all_stats) * 100
            avg_time = sum(s['search_time'] + s['generation_time'] for s in all_stats) / len(all_stats)
            
            print(f"\n📊 ПРОМЕЖУТОЧНАЯ СТАТИСТИКА:")
            print(f"   Обработано: {i}/{len(dialogues)}")
            print(f"   Точность: {accuracy:.1f}% ({correct}/{len(all_stats)})")
            print(f"   Среднее время: {avg_time:.2f}s")
    
    total_time = time.time() - start_time
    
    # ===================================================================
    # ФИНАЛЬНАЯ СТАТИСТИКА
    # ===================================================================
    print(f"\n{'='*80}")
    print("📊 ФИНАЛЬНАЯ СТАТИСТИКА")
    print(f"{'='*80}")
    
    total = len(all_stats)
    correct = sum(1 for s in all_stats if s['success'])
    accuracy = correct / total * 100 if total > 0 else 0
    
    avg_search_time = sum(s['search_time'] for s in all_stats) / total if total > 0 else 0
    avg_gen_time = sum(s['generation_time'] for s in all_stats) / total if total > 0 else 0
    avg_similarity = sum(s['similarity'] for s in all_stats) / total if total > 0 else 0
    
    print(f"\n⏱️  ВРЕМЯ:")
    print(f"   Общее: {total_time:.1f}s ({total_time/60:.1f} мин)")
    print(f"   Средн. на диалог: {total_time/total:.2f}s")
    print(f"   Средн. поиск: {avg_search_time:.3f}s")
    print(f"   Средн. генерация: {avg_gen_time:.3f}s")
    
    print(f"\n📈 КАЧЕСТВО:")
    print(f"   Правильных: {correct}/{total} ({accuracy:.1f}%)")
    print(f"   Средняя схожесть: {avg_similarity:.2%}")
    
    # Статистика по типам вопросов
    print(f"\n📋 ПО ТИПАМ ВОПРОСОВ:")
    types = {}
    for s in all_stats:
        q_type = s['question_type']
        if q_type not in types:
            types[q_type] = {'total': 0, 'correct': 0}
        types[q_type]['total'] += 1
        if s['success']:
            types[q_type]['correct'] += 1
    
    for q_type, counts in sorted(types.items()):
        acc = counts['correct'] / counts['total'] * 100 if counts['total'] > 0 else 0
        print(f"   {q_type}: {counts['correct']}/{counts['total']} ({acc:.1f}%)")
    
    # Финальная статистика SearchEngine
    print(f"\n{'='*80}")
    model.search_engine.print_stats()
    
    # Сохраняем сводный отчет
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("СВОДНЫЙ ОТЧЕТ\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Обработано диалогов: {total}\n")
        f.write(f"Правильных ответов: {correct} ({accuracy:.1f}%)\n")
        f.write(f"Средняя схожесть: {avg_similarity:.2%}\n")
        f.write(f"Общее время: {total_time:.1f}s\n")
        f.write(f"Среднее время на диалог: {total_time/total:.2f}s\n\n")
        
        f.write("ПО ТИПАМ ВОПРОСОВ:\n")
        for q_type, counts in sorted(types.items()):
            acc = counts['correct'] / counts['total'] * 100 if counts['total'] > 0 else 0
            f.write(f"  {q_type}: {counts['correct']}/{counts['total']} ({acc:.1f}%)\n")
    
        print(f"\n💾 Сводный отчет: {summary_file}")
        
        print(f"\n🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
        print(f"📁 Все отчеты в: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА В MAIN(): {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()