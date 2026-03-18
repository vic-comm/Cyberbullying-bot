import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import asyncio
import asyncpg
import pandas as pd
import redis
from redis.connection import ConnectionPool
from dotenv import load_dotenv
import os
load_dotenv()
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
            redis_url: str = os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            max_connections: int = 50,
            socket_timeout: float = 5.0,
            socket_connect_timeout: float = 5.0
        ):
            """
            Initialize Feature Store with Redis connection.
            
            Args:
                redis_url: Full Redis connection string (e.g., rediss://default:password@host:port)
                max_connections: Max connections in pool
                socket_timeout: Socket operation timeout in seconds
                socket_connect_timeout: Socket connection timeout in seconds
            """
            self.redis_url = redis_url
            
            try:
                # Create connection pool directly from URL for better performance
                self.pool = ConnectionPool.from_url(
                    url=redis_url,
                    max_connections=max_connections,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=socket_connect_timeout,
                    decode_responses=True,  # Auto-decode bytes to strings
                    health_check_interval=15,   # Pings Upstash every 15s to keep the AWS socket alive
                    retry_on_timeout=True
                )
                
                self.redis = redis.Redis(connection_pool=self.pool)
                
                # Test connection
                self.redis.ping()
                logger.info("✅ Connected to Redis successfully via URL")
                
            except redis.ConnectionError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise FeatureStoreError(f"Redis connection failed: {e}")    
            
    def sync_offline_to_online(self, parquet_path: str, feature_group_name: str, version: str = "v1", entity_key: str = "user_id", ttl_days: Optional[int] = None, batch_size: int = 1000) -> Dict[str, Any]:
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
    
    async def sync_from_supabase_async(
        self,
        database_url: str,
        feature_group_name: str = "user_toxicity",
        version: str = "prod",
        entity_key: str = "user_id",
        lookback_days: int = 30,
        ttl_days: Optional[int] = 7,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Sync features directly from Supabase to Redis.
        
        This replaces reading from Parquet when you want real-time features.
        
        Args:
            database_url: Supabase connection string
            feature_group_name: Logical name for feature group
            version: Feature version
            entity_key: Entity column (user_id)
            lookback_days: How far back to fetch user history
            ttl_days: Redis key expiration
            batch_size: Records per batch
        
        Returns:
            Sync statistics
        """
        logger.info(f"🔄 Syncing from Supabase → Redis")
        logger.info(f"   Feature Group: {feature_group_name}, Version: {version}")
        
        sync_start = datetime.now()
        cutoff = sync_start - timedelta(days=lookback_days)
        
        try:
            # Connect to Supabase
            conn = await asyncpg.connect(database_url, statement_cache_size=0)
            
            # Query aggregated user features
            query = """
                WITH user_stats AS (
                    SELECT 
                        user_id,
                        COUNT(*) FILTER (WHERE severity IN ('LOW', 'MEDIUM', 'HIGH')) as violation_count_7d,
                        COUNT(*) as total_messages_7d,
                        AVG(CASE WHEN severity IN ('LOW', 'MEDIUM', 'HIGH') THEN 1 ELSE 0 END) as user_bad_ratio_7d,
                        MAX(timestamp) as last_message_time,
                        MIN(timestamp) as first_message_time
                    FROM logs
                    WHERE timestamp > $1
                    GROUP BY user_id
                )
                SELECT 
                    user_id,
                    violation_count_7d,
                    total_messages_7d,
                    user_bad_ratio_7d,
                    EXTRACT(EPOCH FROM (NOW() - last_message_time)) / 3600.0 as hours_since_last_msg,
                    EXTRACT(EPOCH FROM (NOW() - first_message_time)) / 86400.0 as account_age_days
                FROM user_stats
                WHERE total_messages_7d >= 3
            """
            
            rows = await conn.fetch(query, cutoff)
            await conn.close()
            
            logger.info(f"📥 Fetched {len(rows)} users from Supabase")
            
            # Stats
            stats = {
                "total_records": len(rows),
                "feature_group": feature_group_name,
                "version": version,
                "started_at": sync_start.isoformat(),
                "synced_records": 0,
                "failed_records": 0,
                "errors": []
            }
            
            # Sync to Redis using pipeline
            pipe = self.redis.pipeline()
            batch_count = 0
            
            for row in rows:
                try:
                    entity_id = str(row['user_id'])
                    redis_key = f"{feature_group_name}:{version}:{entity_id}"
                    
                    # Build feature dict (exclude user_id)
                    features = {
                        k: v for k, v in dict(row).items() 
                        if k != 'user_id'
                    }
                    
                    # Serialize and store
                    typed_features = self._serialize_features(features)
                    pipe.hset(redis_key, mapping=typed_features)
                    
                    # Set TTL
                    if ttl_days:
                        pipe.expire(redis_key, ttl_days * 86400)
                    
                    batch_count += 1
                    stats["synced_records"] += 1
                    
                    # Execute batch
                    if batch_count >= batch_size:
                        pipe.execute()
                        batch_count = 0
                        
                        if stats["synced_records"] % 5000 == 0:
                            logger.info(f"   Synced {stats['synced_records']} users...")
                
                except Exception as e:
                    error_msg = f"Failed to sync user {row.get('user_id')}: {e}"
                    logger.warning(error_msg)
                    stats["failed_records"] += 1
                    stats["errors"].append(error_msg)
            
            # Execute remaining
            if batch_count > 0:
                pipe.execute()
            
            # Finalize
            sync_duration = (datetime.now() - sync_start).total_seconds()
            stats["completed_at"] = datetime.now().isoformat()
            stats["duration_seconds"] = round(sync_duration, 2)
            stats["records_per_second"] = round(stats["synced_records"] / sync_duration, 2)
            
            logger.info(
                f"✅ Sync complete: {stats['synced_records']} users in {sync_duration:.2f}s "
                f"({stats['records_per_second']:.2f} rec/s)"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Supabase sync failed: {e}", exc_info=True)
            raise FeatureStoreError(f"Supabase sync failed: {e}")
    
    def sync_from_supabase(
        self,
        database_url: str,
        feature_group_name: str = "user_toxicity",
        version: str = "prod",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for async Supabase sync.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(
                self.sync_from_supabase_async(
                    database_url=database_url,
                    feature_group_name=feature_group_name,
                    version=version,
                    **kwargs
                )
            )
        finally:
            loop.close()

    def close(self):
        """Close Redis connection pool"""
        if self.redis:
            self.redis.close()
            logger.info("Closed Redis connection")

# CLI UTILITY
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
    logging.basicConfig(level=logging.INFO)
    
    
    fs = FeatureStore(redis_host=os.getenv('REDIS_HOST'), redis_port=os.getenv("REDIS_PORT"))

    try:
        sync_stats = fs.sync_from_supabase(
            database_url=os.getenv('DATABASE_URL'),
            feature_group_name="user_toxicity",
            version="prod",
            lookback_days=30, 
            ttl_days=7        
        )
        
        logger.info(f"✅ Synced {sync_stats['synced_records']} user features to Redis")
    except Exception as e:
        logger.error(f"⚠️ Feature sync failed: {e}")
    
    if args.sync:
        if not args.parquet:
            print("Error: --parquet required")
            exit(1)
        fs.sync_from_supabase(args.parquet, "user_toxicity", "prod", 7)


