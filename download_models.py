"""
Правильное скачивание ВСЕХ моделей для офлайн-работы
"""
from sentence_transformers import SentenceTransformer, CrossEncoder
import os

# Путь для сохранения
SAVE_DIR = "./src/submit/weights"
os.makedirs(SAVE_DIR, exist_ok=True)

print("="*80)
print("📥 СКАЧИВАНИЕ МОДЕЛЕЙ")
print("="*80)

# =============================================================================
# 1. ЭМБЕДДЕРЫ
# =============================================================================
embedders = [
    ('intfloat/multilingual-e5-small', 'multilingual-e5-small'),
    ('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 'paraphrase-multilingual-mpnet'),
    ('cointegrated/rubert-tiny2', 'rubert-tiny2')
]

print("\n[1/2] Эмбеддеры...")
for hub_name, folder_name in embedders:
    print(f"\n  📦 {hub_name}")
    try:
        model = SentenceTransformer(hub_name)
        save_path = os.path.join(SAVE_DIR, folder_name)
        model.save(save_path)
        
        # Проверка
        files = os.listdir(save_path)
        has_modules = 'modules.json' in files
        has_config = 'config_sentence_transformers.json' in files
        has_weights = any('model' in f for f in files)
        
        if has_modules and has_config and has_weights:
            print(f"  ✅ Сохранено в {folder_name}/ ({len(files)} файлов)")
        else:
            print(f"  ⚠️  Сохранено, но могут быть проблемы:")
            print(f"      modules.json: {has_modules}")
            print(f"      config_sentence_transformers.json: {has_config}")
            print(f"      веса: {has_weights}")
            
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")

# =============================================================================
# 2. RERANKER
# =============================================================================
print("\n[2/2] Reranker...")
print(f"\n  📦 BAAI/bge-reranker-base")

try:
    reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512, num_labels=1)
    save_path = os.path.join(SAVE_DIR, 'bge-reranker-base')
    reranker.save(save_path)
    
    # Проверка
    files = os.listdir(save_path)
    has_config = 'config.json' in files
    has_weights = any('model' in f for f in files)
    
    if has_config and has_weights:
        print(f"  ✅ Сохранено в bge-reranker-base/ ({len(files)} файлов)")
    else:
        print(f"  ⚠️  Сохранено, но могут быть проблемы:")
        print(f"      config.json: {has_config}")
        print(f"      веса: {has_weights}")
        
except Exception as e:
    print(f"  ❌ Ошибка: {e}")

# =============================================================================
# 3. ПРОВЕРКА
# =============================================================================
print("\n" + "="*80)
print("🔍 ПРОВЕРКА ЗАГРУЖЕННЫХ МОДЕЛЕЙ")
print("="*80)

print("\n[Тест 1] Эмбеддеры...")
for _, folder_name in embedders:
    try:
        model = SentenceTransformer(os.path.join(SAVE_DIR, folder_name))
        emb = model.encode(["тест"], show_progress_bar=False)
        print(f"  ✅ {folder_name}: {emb.shape}")
    except Exception as e:
        print(f"  ❌ {folder_name}: {e}")

print("\n[Тест 2] Reranker...")
try:
    reranker = CrossEncoder(os.path.join(SAVE_DIR, 'bge-reranker-base'), num_labels=1)
    score = reranker.predict([["вопрос", "ответ"]], show_progress_bar=False)
    print(f"  ✅ bge-reranker-base: score={score[0]:.4f}")
except Exception as e:
    print(f"  ❌ bge-reranker-base: {e}")

print("\n" + "="*80)
print("✅ ГОТОВО!")
print("="*80)
print(f"\nМодели сохранены в: {SAVE_DIR}/")
print("\nТеперь можно упаковывать и отправлять в облако!")
