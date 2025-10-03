#!/usr/bin/env python3
"""
Патч для исправления message.content во всех файлах проекта.

ПРОБЛЕМА: message.content может быть float, int, None
РЕШЕНИЕ: Всегда конвертируем str(content) и проверяем на None

Запуск:
    python fix_message_content.py
"""

import os
import re
from pathlib import Path


def fix_data_processor():
    """Исправляет data_processor.py"""
    file_path = Path("src/submit/data_processor.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Ищем метод create_context_chunks
    # Заменяем user_text = msg.content.strip()
    old_pattern = r"user_text = msg\.content\.strip\(\)"
    new_code = "user_text = str(msg.content).strip() if msg.content is not None else \"\""
    content = re.sub(old_pattern, new_code, content)
    
    # Заменяем assistant_confirmation
    old_pattern2 = r"assistant_confirmation = messages\[i \+ 1\]\.content\.strip\(\)"
    new_code2 = """assistant_content = messages[i + 1].content
                    assistant_confirmation = str(assistant_content).strip() if assistant_content is not None else \"\" """
    content = re.sub(old_pattern2, new_code2, content)
    
    # Заменяем для assistant chunks
    old_pattern3 = r"content = msg\.content\.strip\(\)"
    new_code3 = "content = str(msg.content).strip() if msg.content is not None else \"\""
    content = re.sub(old_pattern3, new_code3, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Исправлен: {file_path}")


def create_safe_message_converter():
    """Создаёт вспомогательную функцию для конвертации сообщений"""
    
    code = '''
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
'''
    
    return code


def fix_test_scripts():
    """Исправляет все тестовые скрипты"""
    
    test_files = [
        "test_dedup_fix.py",
        "test_dialogue_2.py", 
        "debug_search.py",
        "working_process.py"
    ]
    
    safe_converter = create_safe_message_converter()
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            continue
        
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Заменяем старую функцию convert_to_messages на безопасную
        old_function = r"def convert_to_messages\(dialogue_messages\):.*?return messages"
        
        if re.search(old_function, content, re.DOTALL):
            content = re.sub(old_function, safe_converter.strip(), content, flags=re.DOTALL)
            
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✓ Исправлен: {test_file}")


def create_validation_script():
    """Создаёт скрипт для проверки типов в датасете"""
    
    script = '''#!/usr/bin/env python3
"""
Проверка типов message.content в датасете
"""
import json

def check_message_types(file_path):
    """Проверяет типы content в датасете"""
    
    print(f"Проверка {file_path}...")
    
    type_stats = {}
    problematic = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                if 'sessions' in data:
                    for session in data['sessions']:
                        for msg in session['messages']:
                            content = msg.get('content')
                            
                            # Определяем тип
                            content_type = type(content).__name__
                            type_stats[content_type] = type_stats.get(content_type, 0) + 1
                            
                            # Сохраняем проблемные
                            if content_type != 'str':
                                problematic.append({
                                    'line': line_num,
                                    'type': content_type,
                                    'value': content,
                                    'role': msg.get('role')
                                })
            except:
                continue
    
    print("\\nСтатистика типов:")
    for type_name, count in sorted(type_stats.items()):
        print(f"  {type_name}: {count}")
    
    if problematic:
        print(f"\\n⚠️  Найдено {len(problematic)} проблемных сообщений:")
        for item in problematic[:10]:
            print(f"  Строка {item['line']}: {item['type']} = {item['value']} ({item['role']})")
        
        if len(problematic) > 10:
            print(f"  ... и ещё {len(problematic) - 10}")
    else:
        print("\\n✓ Проблемных типов не найдено")

if __name__ == "__main__":
    check_message_types("data/format_example.jsonl")
'''
    
    with open("check_message_types.py", 'w', encoding='utf-8') as f:
        f.write(script)
    
    os.chmod("check_message_types.py", 0o755)
    print("✓ Создан: check_message_types.py")


def main():
    print("="*60)
    print("ИСПРАВЛЕНИЕ message.content ВО ВСЕХ ФАЙЛАХ")
    print("="*60)
    
    print("\n[1/3] Исправление data_processor.py...")
    try:
        fix_data_processor()
    except Exception as e:
        print(f"  ⚠️  Ошибка: {e}")
    
    print("\n[2/3] Исправление тестовых скриптов...")
    try:
        fix_test_scripts()
    except Exception as e:
        print(f"  ⚠️  Ошибка: {e}")
    
    print("\n[3/3] Создание скрипта проверки...")
    try:
        create_validation_script()
    except Exception as e:
        print(f"  ⚠️  Ошибка: {e}")
    
    print("\n" + "="*60)
    print("✅ ПАТЧ ЗАВЕРШЕН")
    print("="*60)
    
    print("\nТеперь запусти проверку:")
    print("  python check_message_types.py")
    print("\nЕсли найдутся проблемные типы - патч их исправит!")


if __name__ == "__main__":
    main()
