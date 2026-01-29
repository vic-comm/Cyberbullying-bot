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
    
    def save_embeddings(
        self, 
        embeddings, 
        data_path: str, 
        additional_info: Optional[dict] = None
    ):
        """Save embeddings and metadata (local + S3 if enabled)"""
        data_hash = self.get_data_hash(data_path)
        
        metadata = {
            'data_hash': data_hash,
            'data_path': str(data_path),
            'timestamp': pd.Timestamp.now().isoformat(),
            'shape': embeddings.shape,
            's3_enabled': self.use_s3
        }
        
        if additional_info:
            metadata.update(additional_info)
        
        # Save locally
        joblib.dump(embeddings, self.embeddings_file)
        joblib.dump(metadata, self.metadata_file)
        
        print(f"   ✅ Cached embeddings locally: {embeddings.shape}")
        print(f"   📁 Local location: {self.embeddings_file}")
        
        # Upload to S3 if enabled
        if self.use_s3:
            self._upload_to_s3()
    
    def load_embeddings(self) -> Tuple:
        """Load cached embeddings (from local or S3)"""
        # Ensure we have local cache
        if not self.embeddings_file.exists() and self.use_s3:
            self._download_from_s3()
        
        if not self.embeddings_file.exists():
            raise FileNotFoundError("No cached embeddings found locally or in S3")
        
        embeddings = joblib.load(self.embeddings_file)
        metadata = joblib.load(self.metadata_file)
        
        print(f"   ♻️  Loaded cached embeddings: {embeddings.shape}")
        print(f"   📅 Cache date: {metadata.get('timestamp', 'Unknown')}")
        
        return embeddings, metadata
    
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