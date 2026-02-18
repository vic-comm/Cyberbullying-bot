# scripts/migrate_cache.py
import pandas as pd
import numpy as np
import pickle
import hashlib
from pathlib import Path

DATA_PATH = Path("data/training_data_backup.parquet")
OLD_CACHE_PATH = Path("cache/bert_embeddings.pkl")
NEW_CACHE_PATH = Path("cache/embedding_store.pkl")

def migrate():
    print(" Starting Cache Migration...")
    
    if not DATA_PATH.exists():
        print(" Data file not found!")
        return
    df = pd.read_parquet(DATA_PATH)
    texts = df['text'].astype(str).tolist()
    print(f"   Loaded {len(texts)} texts.")

    if not OLD_CACHE_PATH.exists():
        print(" Old cache not found! You must run the full compute.")
        return
    
    import joblib 
    try:
        old_embeddings = joblib.load(OLD_CACHE_PATH)
    except:
        with open(OLD_CACHE_PATH, 'rb') as f:
            old_embeddings = pickle.load(f)
            
    print(f"   Loaded {len(old_embeddings)} old embeddings.")

    # 3. Validation
    if len(texts) != len(old_embeddings):
        print(f"❌ MISMATCH: {len(texts)} texts vs {len(old_embeddings)} embeddings.")
        print("   Cannot migrate safely. Please recompute from scratch.")
        return

    # 4. Conversion (Array -> Dictionary)
    print("   Converting to Hash Store...")
    new_store = {}
    
    for text, vector in zip(texts, old_embeddings):
        # Calculate Hash
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        new_store[h] = vector

    # 5. Save New Cache
    print(f"   Saving {len(new_store)} items to {NEW_CACHE_PATH}...")
    with open(NEW_CACHE_PATH, 'wb') as f:
        pickle.dump(new_store, f)
        
    print("✅ Migration Complete! Next run will use this cache.")

if __name__ == "__main__":
    migrate()