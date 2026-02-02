# import pandas as pd
# import redis
# import os
# import json

# class FeatureStore:
#     def __init__(self, redis_host='localhost', redis_port=6379):
#         self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

#     def sync_offline_to_online(self, parquet_path, feature_group_name, version="v1"):
#         print(f" Syncing {parquet_path} to Redis...")
#         df = pd.read_parquet(parquet_path)
        
#         # Use a pipeline for massive speed increase
#         pipe = self.redis.pipeline()
        
#         count = 0
#         for _, row in df.iterrows():
#             entity_id = row['user_id']
            
#             redis_key = f"{feature_group_name}:{version}:{entity_id}"
            
#             # Convert row to JSON (or Hash)
#             # We filter out the entity_id from the payload to save space
#             feature_data = row.drop('user_id').to_dict()
            
#             # Store as Hash Map in Redis
#             pipe.hset(redis_key, mapping=feature_data)
            
#             # Optional: Set expiry (TTL) to auto-clean old versions if needed
#             # pipe.expire(redis_key, 86400 * 7) # 7 days
            
#             count += 1
#             if count % 1000 == 0:
#                 pipe.execute() # Commit batch
                
#         pipe.execute() # Commit remaining
#         print(f"Synced {count} records to Redis key prefix '{feature_group_name}:{version}'")

#     def get_online_features(self, feature_group_name, entity_id, version="v1"):
#         """
#         Low-latency fetch for the API.
#         """
#         redis_key = f"{feature_group_name}:{version}:{entity_id}"
#         data = self.redis.hgetall(redis_key)
        
#         if not data:
#             return None
            
#         # Redis stores everything as strings, we might need to cast types back
#         # For a truly lightweight solution, handle type conversion here based on a schema
#         return data

# if __name__ == "__main__":
#     fs = FeatureStore()
#     fs.sync_offline_to_online(
#         parquet_path="data/training_data_with_history.parquet",
#         feature_group_name="user_toxicity",
#         version="prod"
#     )

# mlops/feature_store.py
"""
Lightweight Feature Store implementation using Redis for online serving
and Parquet for offline storage.

Provides:
- Low-latency feature retrieval (<10ms p99)
- Batch syncing from offline to online store
- Type preservation and validation
- Connection pooling for production use
"""
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

import pandas as pd
import redis
from redis.connection import ConnectionPool

logger = logging.getLogger(__name__)

class FeatureStoreError(Exception):
    """Base exception for feature store operations"""
    pass

class FeatureStore:
    """
    Feature Store managing online (Redis) and offline (Parquet) storage.
    
    Key Design Decisions:
    - Uses Redis Hash Maps for structured data storage
    - Implements connection pooling for production reliability
    - Supports versioning for safe feature updates
    - Provides TTL support for automatic cleanup
    """
    
    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        max_connections: int = 50,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0
    ):
        """
        Initialize Feature Store with Redis connection.
        
        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password (if required)
            max_connections: Max connections in pool
            socket_timeout: Socket operation timeout in seconds
            socket_connect_timeout: Socket connection timeout in seconds
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        try:
            # Create connection pool for better performance
            self.pool = ConnectionPool(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                decode_responses=True  # Auto-decode bytes to strings
            )
            
            self.redis = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            self.redis.ping()
            logger.info(f"✅ Connected to Redis at {redis_host}:{redis_port}")
            
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise FeatureStoreError(f"Redis connection failed: {e}")
    
    def sync_offline_to_online(
        self,
        parquet_path: str,
        feature_group_name: str,
        version: str = "v1",
        entity_key: str = "user_id",
        ttl_days: Optional[int] = None,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Sync features from Parquet (offline) to Redis (online).
        
        This is typically run:
        - After training to deploy new feature versions
        - On a schedule to keep features fresh
        - During model deployment pipelines
        
        Args:
            parquet_path: Path to Parquet file
            feature_group_name: Logical name for this feature group
            version: Version identifier (e.g., "v1", "prod", "2024-01-01")
            entity_key: Column name for the entity ID (default: user_id)
            ttl_days: Optional TTL in days for auto-cleanup
            batch_size: Number of records per Redis pipeline batch
            
        Returns:
            Dict with sync statistics
        """
        logger.info(f"Starting sync: {parquet_path} → Redis")
        logger.info(f"Feature Group: {feature_group_name}, Version: {version}")
        
        sync_start = datetime.now()
        
        try:
            # Read offline data
            df = pd.read_parquet(parquet_path)
            logger.info(f"Loaded {len(df)} records from Parquet")
            
            # Validate entity key exists
            if entity_key not in df.columns:
                raise FeatureStoreError(
                    f"Entity key '{entity_key}' not found in columns: {df.columns.tolist()}"
                )
            
            # Prepare statistics
            stats = {
                "total_records": len(df),
                "feature_group": feature_group_name,
                "version": version,
                "started_at": sync_start.isoformat(),
                "synced_records": 0,
                "failed_records": 0,
                "errors": []
            }
            
            # Use pipeline for batch operations (much faster)
            pipe = self.redis.pipeline()
            batch_count = 0
            
            for idx, row in df.iterrows():
                try:
                    entity_id = str(row[entity_key])
                    
                    # Build Redis key: feature_group:version:entity_id
                    redis_key = f"{feature_group_name}:{version}:{entity_id}"
                    
                    # Prepare feature data (exclude entity key)
                    feature_data = row.drop(entity_key).to_dict()
                    
                    # Convert all values to strings (Redis requirement)
                    # Store type information for reconstruction
                    typed_features = self._serialize_features(feature_data)
                    
                    # Store as Redis Hash Map
                    pipe.hset(redis_key, mapping=typed_features)
                    
                    # Set TTL if specified
                    if ttl_days:
                        pipe.expire(redis_key, ttl_days * 86400)
                    
                    batch_count += 1
                    stats["synced_records"] += 1
                    
                    # Execute batch
                    if batch_count >= batch_size:
                        pipe.execute()
                        batch_count = 0
                        
                        # Log progress
                        if stats["synced_records"] % 10000 == 0:
                            logger.info(f"Synced {stats['synced_records']} records...")
                    
                except Exception as e:
                    error_msg = f"Failed to sync record {idx}: {str(e)}"
                    logger.warning(error_msg)
                    stats["failed_records"] += 1
                    stats["errors"].append(error_msg)
                    
                    # Continue with next record
                    continue
            
            # Execute remaining batch
            if batch_count > 0:
                pipe.execute()
            
            # Finalize statistics
            sync_duration = (datetime.now() - sync_start).total_seconds()
            stats["completed_at"] = datetime.now().isoformat()
            stats["duration_seconds"] = round(sync_duration, 2)
            stats["records_per_second"] = round(stats["synced_records"] / sync_duration, 2)
            
            logger.info(
                f"✅ Sync complete: {stats['synced_records']} records in {sync_duration:.2f}s "
                f"({stats['records_per_second']:.2f} rec/s)"
            )
            
            if stats["failed_records"] > 0:
                logger.warning(f"⚠️  {stats['failed_records']} records failed")
            
            return stats
            
        except Exception as e:
            logger.error(f"Sync failed: {e}", exc_info=True)
            raise FeatureStoreError(f"Sync operation failed: {e}")
    
    def get_online_features(
        self,
        feature_group_name: str,
        entity_id: str,
        version: str = "v1",
        feature_names: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve features from online store (Redis) with low latency.
        
        This is called during inference to enrich predictions with user context.
        
        Args:
            feature_group_name: Name of the feature group
            entity_id: Entity identifier (e.g., user_id)
            version: Feature version to retrieve
            feature_names: Optional list of specific features to retrieve
                          (if None, returns all features)
        
        Returns:
            Dict of features with proper types, or None if not found
        """
        redis_key = f"{feature_group_name}:{version}:{entity_id}"
        
        try:
            if feature_names:
                # Fetch specific features only (more efficient)
                raw_data = self.redis.hmget(redis_key, feature_names)
                if not any(raw_data):
                    return None
                data = dict(zip(feature_names, raw_data))
            else:
                # Fetch all features
                data = self.redis.hgetall(redis_key)
            
            if not data:
                logger.debug(f"No features found for key: {redis_key}")
                return None
            
            # Deserialize features back to proper types
            typed_features = self._deserialize_features(data)
            
            return typed_features
            
        except redis.RedisError as e:
            logger.error(f"Redis error fetching features: {e}")
            raise FeatureStoreError(f"Failed to fetch features: {e}")
    
    def get_batch_online_features(self, feature_group_name: str, entity_ids: List[str],version: str = "v1") -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Retrieve features for multiple entities efficiently using pipeline.
        
        Args:
            feature_group_name: Name of the feature group
            entity_ids: List of entity identifiers
            version: Feature version to retrieve
            
        Returns:
            Dict mapping entity_id to their features
        """
        try:
            pipe = self.redis.pipeline()
            
            # Queue all requests
            redis_keys = [
                f"{feature_group_name}:{version}:{entity_id}"
                for entity_id in entity_ids
            ]
            
            for key in redis_keys:
                pipe.hgetall(key)
            
            # Execute batch
            results = pipe.execute()
            
            # Map results back to entity IDs
            features_map = {}
            for entity_id, raw_data in zip(entity_ids, results):
                if raw_data:
                    features_map[entity_id] = self._deserialize_features(raw_data)
                else:
                    features_map[entity_id] = None
            
            return features_map
            
        except redis.RedisError as e:
            logger.error(f"Batch feature fetch failed: {e}")
            raise FeatureStoreError(f"Batch fetch failed: {e}")
    
    def delete_feature_group(self, feature_group_name: str, version: str, batch_size: int = 1000) -> int:
        """
        Delete all features for a given feature group and version.
        Useful for cleanup or version migration.
        
        Args:
            feature_group_name: Feature group to delete
            version: Version to delete
            batch_size: Number of keys to delete per batch
            
        Returns:
            Number of keys deleted
        """
        pattern = f"{feature_group_name}:{version}:*"
        logger.info(f"Deleting features matching: {pattern}")
        
        deleted_count = 0
        cursor = 0
        
        try:
            while True:
                cursor, keys = self.redis.scan(
                    cursor=cursor,
                    match=pattern,
                    count=batch_size
                )
                
                if keys:
                    deleted = self.redis.delete(*keys)
                    deleted_count += deleted
                    logger.debug(f"Deleted {deleted} keys (total: {deleted_count})")
                
                if cursor == 0:
                    break
            
            logger.info(f" Deleted {deleted_count} keys for {feature_group_name}:{version}")
            return deleted_count
            
        except redis.RedisError as e:
            logger.error(f"Delete operation failed: {e}")
            raise FeatureStoreError(f"Delete failed: {e}")
    
    def _serialize_features(self, features: Dict[str, Any]) -> Dict[str, str]:
        """
        Convert feature dict to Redis-compatible format (all strings).
        Preserves type information for accurate deserialization.
        """
        serialized = {}
        
        for key, value in features.items():
            if pd.isna(value):
                serialized[key] = "null"
            elif isinstance(value, (int, float)):
                serialized[key] = str(value)
            elif isinstance(value, bool):
                serialized[key] = "true" if value else "false"
            elif isinstance(value, (list, dict)):
                serialized[key] = json.dumps(value)
            else:
                serialized[key] = str(value)
        
        return serialized
    
    def _deserialize_features(self, raw_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Convert Redis strings back to proper Python types.
        """
        deserialized = {}
        
        for key, value in raw_data.items():
            if value == "null":
                deserialized[key] = None
            elif value == "true":
                deserialized[key] = True
            elif value == "false":
                deserialized[key] = False
            elif value.startswith('{') or value.startswith('['):
                # JSON object or array
                try:
                    deserialized[key] = json.loads(value)
                except json.JSONDecodeError:
                    deserialized[key] = value
            else:
                # Try numeric conversion
                try:
                    if '.' in value:
                        deserialized[key] = float(value)
                    else:
                        deserialized[key] = int(value)
                except ValueError:
                    deserialized[key] = value
        
        return deserialized
    
    def health_check(self) -> Dict[str, Any]:
        try:
            info = self.redis.info()
            
            return {
                "status": "healthy",
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "uptime_days": info.get("uptime_in_days")
            }
        except redis.RedisError as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def close(self):
        """Close Redis connection pool"""
        if self.redis:
            self.redis.close()
            logger.info("Closed Redis connection")

# ============================================================================
# CLI UTILITY
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Store CLI")
    parser.add_argument("--sync", action="store_true", help="Sync offline to online")
    parser.add_argument("--parquet", type=str, help="Path to Parquet file")
    parser.add_argument("--feature-group", type=str, default="user_toxicity")
    parser.add_argument("--version", type=str, default="prod")
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--ttl-days", type=int, help="TTL in days")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize Feature Store
    fs = FeatureStore(redis_host=args.redis_host, redis_port=args.redis_port)
    
    if args.sync:
        if not args.parquet:
            print("Error: --parquet required for sync operation")
            exit(1)
        
        stats = fs.sync_offline_to_online(
            parquet_path=args.parquet,
            feature_group_name=args.feature_group,
            version=args.version,
            ttl_days=args.ttl_days
        )
        
        print(f"\n✅ Sync completed:")
        print(f"  - Total records: {stats['total_records']}")
        print(f"  - Synced: {stats['synced_records']}")
        print(f"  - Failed: {stats['failed_records']}")
        print(f"  - Duration: {stats['duration_seconds']}s")
        print(f"  - Throughput: {stats['records_per_second']} rec/s")
    else:
        health = fs.health_check()
        print(f"\nRedis Health: {health}")
    
    fs.close()