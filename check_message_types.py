#!/usr/bin/env python3
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
    
    print("\nСтатистика типов:")
    for type_name, count in sorted(type_stats.items()):
        print(f"  {type_name}: {count}")
    
    if problematic:
        print(f"\n⚠️  Найдено {len(problematic)} проблемных сообщений:")
        for item in problematic[:10]:
            print(f"  Строка {item['line']}: {item['type']} = {item['value']} ({item['role']})")
        
        if len(problematic) > 10:
            print(f"  ... и ещё {len(problematic) - 10}")
    else:
        print("\n✓ Проблемных типов не найдено")

if __name__ == "__main__":
    check_message_types("data/format_example.jsonl")
