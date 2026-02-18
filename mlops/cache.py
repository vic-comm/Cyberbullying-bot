# cache_manager.py
import os
import hashlib
import joblib
import pandas as pd
import boto3
from pathlib import Path
from typing import Optional, Tuple
import tempfile

class EmbeddingCache:
    def __init__(self, use_s3, cache_dir="../cache", s3_bucket: Optional[str] = None, s3_prefix: str = "embeddings-cache"):
        """
        Initialize cache manager with optional S3 support
        
        Args:
            cache_dir: Local cache directory
            s3_bucket: S3 bucket name (e.g., 'cyberbullying-artifacts-victor-obi')
            s3_prefix: Prefix/folder in S3 bucket (e.g., 'embeddings-cache')
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # File names
        self.embeddings_file = self.cache_dir / "bert_embeddings.pkl"
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # S3 configuration
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.use_s3 = use_s3
        
        if self.use_s3:
            self.s3_client = boto3.client('s3')
            self.s3_embeddings_key = f"{s3_prefix}/bert_embeddings.pkl"
            self.s3_metadata_key = f"{s3_prefix}/cache_metadata.json"
            print(f"   🪣 S3 Cache enabled: s3://{s3_bucket}/{s3_prefix}/")
    
    def get_data_hash(self, data_path: str) -> str:
        """Generate hash of the data file to detect changes"""
        file_stats = os.stat(data_path)
        # Quick hash: file size + modification time
        quick_hash = f"{file_stats.st_size}_{file_stats.st_mtime}"
        return quick_hash
    
    def _download_from_s3(self) -> bool:
        """Download cache files from S3 to local"""
        try:
            print("   ⬇️  Downloading cache from S3...")
            
            # Download embeddings
            self.s3_client.download_file(
                self.s3_bucket,
                self.s3_embeddings_key,
                str(self.embeddings_file)
            )
            
            # Download metadata
            self.s3_client.download_file(
                self.s3_bucket,
                self.s3_metadata_key,
                str(self.metadata_file)
            )
            
            print("   ✅ Cache downloaded from S3")
            return True
        except Exception as e:
            print(f"   ⚠️  Could not download from S3: {e}")
            return False
    
    def _upload_to_s3(self) -> bool:
        """Upload cache files from local to S3"""
        try:
            print("   ⬆️  Uploading cache to S3...")
            
            # Upload embeddings
            self.s3_client.upload_file(
                str(self.embeddings_file),
                self.s3_bucket,
                self.s3_embeddings_key
            )
            
            # Upload metadata
            self.s3_client.upload_file(
                str(self.metadata_file),
                self.s3_bucket,
                self.s3_metadata_key
            )
            
            print(f"   ✅ Cache uploaded to S3: s3://{self.s3_bucket}/{self.s3_prefix}/")
            return True
        except Exception as e:
            print(f"   ⚠️  Could not upload to S3: {e}")
            return False
    
    def is_cache_valid(self, data_path: str) -> bool:
        """Check if cached embeddings are still valid"""
        # Check local cache first
        local_exists = self.embeddings_file.exists() and self.metadata_file.exists()
        
        # If not local and S3 enabled, try downloading
        if not local_exists and self.use_s3:
            if self._download_from_s3():
                local_exists = True
        
        if not local_exists:
            return False
        
        try:
            metadata = joblib.load(self.metadata_file)
            current_hash = self.get_data_hash(data_path)
            is_valid = metadata.get('data_hash') == current_hash
            
            if is_valid:
                print("   ✅ Cache is valid")
            else:
                print("   ⚠️  Cache is stale (data changed)")
            
            return is_valid
        except Exception as e:
            print(f"   ⚠️  Error validating cache: {e}")
            return False
    
    # def save_embeddings(
    #     self, 
    #     embeddings, 
    #     data_path: str, 
    #     additional_info: Optional[dict] = None
    # ):
    #     """Save embeddings and metadata (local + S3 if enabled)"""
    #     data_hash = self.get_data_hash(data_path)
        
    #     metadata = {
    #         'data_hash': data_hash,
    #         'data_path': str(data_path),
    #         'timestamp': pd.Timestamp.now().isoformat(),
    #         'shape': embeddings.shape,
    #         's3_enabled': self.use_s3
    #     }
        
    #     if additional_info:
    #         metadata.update(additional_info)
        
    #     # Save locally
    #     joblib.dump(embeddings, self.embeddings_file)
    #     joblib.dump(metadata, self.metadata_file)
        
    #     print(f"   ✅ Cached embeddings locally: {embeddings.shape}")
    #     print(f"   📁 Local location: {self.embeddings_file}")
        
    #     # Upload to S3 if enabled
    #     if self.use_s3:
    #         self._upload_to_s3()
    
    # def load_embeddings(self) -> Tuple:
    #     """Load cached embeddings (from local or S3)"""
    #     # Ensure we have local cache
    #     if not self.embeddings_file.exists() and self.use_s3:
    #         self._download_from_s3()
        
    #     if not self.embeddings_file.exists():
    #         raise FileNotFoundError("No cached embeddings found locally or in S3")
        
    #     embeddings = joblib.load(self.embeddings_file)
    #     metadata = joblib.load(self.metadata_file)
        
    #     print(f"   ♻️  Loaded cached embeddings: {embeddings.shape}")
    #     print(f"   📅 Cache date: {metadata.get('timestamp', 'Unknown')}")
        
    #     return embeddings, metadata
    def save_embeddings(
        self, 
        embeddings, 
        data_path: str, 
        filename: str = "bert_embeddings.pkl",
        additional_info: Optional[dict] = None
    ):
        """Save embeddings with configurable filename"""
        target_file = self.cache_dir / filename
        
        # ... existing metadata logic ...
        metadata = {
            'timestamp': pd.Timestamp.now().isoformat(),
            's3_enabled': self.use_s3,
            # We skip data_hash check for incremental stores usually
            # as the data grows, but the store remains valid for subsets
        }
        if additional_info:
            metadata.update(additional_info)

        # Save locally
        joblib.dump(embeddings, target_file)
        joblib.dump(metadata, self.metadata_file)
        
        print(f"   ✅ Cached to {target_file}")
        
        # Upload to S3
        if self.use_s3:
            try:
                print(f"   ⬆️  Uploading {filename} to S3...")
                self.s3_client.upload_file(
                    str(target_file),
                    self.s3_bucket,
                    f"{self.s3_prefix}/{filename}"
                )
                # Upload metadata
                self.s3_client.upload_file(
                    str(self.metadata_file),
                    self.s3_bucket,
                    self.s3_metadata_key
                )
            except Exception as e:
                print(f"   ⚠️  S3 Upload failed: {e}")

    def load_embeddings(self, filename: str = "bert_embeddings.pkl") -> Tuple:
        """Load cached embeddings with configurable filename"""
        
        target_file = self.cache_dir / filename
        target_metadata = self.cache_dir / "cache_metadata.json"
        
        # S3 Support
        if not target_file.exists() and self.use_s3:
            try:
                print(f"   ⬇️  Downloading {filename} from S3...")
                self.s3_client.download_file(
                    self.s3_bucket,
                    f"{self.s3_prefix}/{filename}",
                    str(target_file)
                )
                # Try downloading metadata too, but don't fail if missing
                try:
                    self.s3_client.download_file(
                        self.s3_bucket,
                        self.s3_metadata_key,
                        str(target_metadata)
                    )
                except: pass
            except Exception as e:
                raise FileNotFoundError(f"Could not download {filename}: {e}")
        
        if not target_file.exists():
            raise FileNotFoundError(f"Cache file {filename} not found")
        
        data = joblib.load(target_file)
        
        # Load metadata if exists, else empty dict
        metadata = {}
        if target_metadata.exists():
            metadata = joblib.load(target_metadata)
            
        return data, metadata
    
    def clear_cache(self, clear_s3: bool = False):
        """Clear cache files"""
        # Clear local
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        print("   🗑️  Local cache cleared")
        
        # Clear S3 if requested
        if clear_s3 and self.use_s3:
            try:
                self.s3_client.delete_object(Bucket=self.s3_bucket, Key=self.s3_embeddings_key)
                self.s3_client.delete_object(Bucket=self.s3_bucket, Key=self.s3_metadata_key)
                print("   🗑️  S3 cache cleared")
            except Exception as e:
                print(f"   ⚠️  Could not clear S3 cache: {e}")


# import hashlib
# import numpy as np
# import pandas as pd
# from pathlib import Path
# import pickle

# # ... existing imports ...

# class DataPreparator:
#     # ... existing __init__ ...

#     def _compute_text_hashes(self, text_series: pd.Series) -> pd.Series:
#         """Helper to create unique IDs for text content"""
#         return text_series.apply(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())

#     def _get_embeddings(self, df: pd.DataFrame, force_recompute: bool) -> np.ndarray:
#         """
#         Incremental Embedding Generation:
#         1. Load existing embeddings map (Hash -> Vector)
#         2. Identify new texts
#         3. Compute only new embeddings
#         4. Merge and save
#         """
#         print("\n2. Text Embeddings (DistilBERT - Incremental)...")
        
#         # Path to the "Master Embedding Store" (Dictionary: Hash -> Vector)
#         # We use a dict or a parquet file for faster lookups than a raw numpy array
#         master_store_path = self.config.CACHE_DIR / 'embedding_store.pkl'
        
#         # 1. Compute hashes for current data
#         print("   -> Computing text hashes...")
#         df['text_hash'] = self._compute_text_hashes(df['text'].astype(str))
#         current_hashes = df['text_hash'].values
        
#         # 2. Load existing store
#         embedding_store = {}
#         if master_store_path.exists() and not force_recompute:
#             print("   -> Loading existing embedding store...")
#             try:
#                 with open(master_store_path, 'rb') as f:
#                     embedding_store = pickle.load(f)
#                 print(f"   ✅ Loaded {len(embedding_store):,} existing embeddings")
#             except Exception as e:
#                 print(f"   ⚠️  Corrupt store, starting fresh: {e}")
#                 embedding_store = {}
#         else:
#             if force_recompute:
#                 print("   🔄 Force recompute active: Ignoring existing cache.")
        
#         # 3. Identify missing hashes
#         # This is the "Delta" logic
#         missing_hashes = [h for h in current_hashes if h not in embedding_store]
        
#         if not missing_hashes:
#             print("   ✅ No new data found. Using 100% cached embeddings.")
#         else:
#             print(f"   ⚡ Found {len(missing_hashes):,} new samples to compute.")
            
#             # Filter the dataframe to get text for missing hashes
#             # We assume hashes are unique per text content
#             new_texts_df = df[df['text_hash'].isin(missing_hashes)].drop_duplicates(subset=['text_hash'])
#             new_texts = new_texts_df['text'].tolist()
#             new_hashes = new_texts_df['text_hash'].tolist()
            
#             # 4. Generate embeddings ONLY for new data
#             if new_texts:
#                 new_embeddings = self.embedding_generator.generate(new_texts)
                
#                 # Update the store
#                 for h, emb in zip(new_hashes, new_embeddings):
#                     embedding_store[h] = emb
                
#                 print(f"   ✅ Computed and merged {len(new_embeddings)} new embeddings")
                
#                 # 5. Save updated store (Atomic write pattern recommended in production)
#                 print("   -> Saving updated embedding store...")
#                 # Ensure directory exists
#                 master_store_path.parent.mkdir(parents=True, exist_ok=True)
#                 with open(master_store_path, 'wb') as f:
#                     pickle.dump(embedding_store, f)
        
#         # 6. Align embeddings with the current DataFrame order
#         # This ensures row 0 in df matches row 0 in the returned numpy array
#         print("   -> Aligning embeddings to current dataset...")
        
#         # Pre-allocate array for speed
#         # Assuming embedding dim is 768 for DistilBERT
#         final_embeddings = np.zeros((len(df), 768), dtype=np.float32)
        
#         # Fill array using the store lookup
#         # (This is fast because it's just dictionary lookups)
#         for idx, text_hash in enumerate(current_hashes):
#             final_embeddings[idx] = embedding_store[text_hash]
            
#         return final_embeddings