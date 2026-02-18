from typing import Dict, Optional
from ..models.server_config import ServerConfig
from ..database import DatabaseManager

# class ConfigManager:
#     def __init__(self, db: DatabaseManager):
#         self.db = db
#         self._cache: Dict[str, ServerConfig] = {}
    
#     async def get_config(self, server_id: str, server_name: str = "Unknown", platform='discord') -> ServerConfig:
#         """
#         Get configuration for a server (with caching)
#         Creates default if doesn't exist
#         """
#         # Check cache first
#         if server_id in self._cache:
#             return self._cache[server_id]
        
#         # Load from database
#         config_dict = await self.db.get_server_config(server_id)
        
#         if config_dict:
#             config = ServerConfig.from_dict(config_dict)
#         else:
#             # Create default config
#             config = ServerConfig(server_id=server_id, server_name=server_name)
#             await self.save_config(config)
        
#         # Cache it
#         self._cache[server_id] = config
        
#         return config
    
#     async def save_config(self, config: ServerConfig):
#         await self.db.save_server_config(config.to_dict())
        
#         # Update cache
#         self._cache[config.server_id] = config
    
#     def invalidate_cache(self, server_id: str):
#         if server_id in self._cache:
#             del self._cache[server_id]

from typing import Dict, Optional
from ..models.server_config import ServerConfig
from ..database import DatabaseManager

class ConfigManager:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self._cache: Dict[str, ServerConfig] = {}
    
    async def get_config(
        self, 
        server_id: str, 
        server_name: str = "Unknown", 
        platform: str = 'discord'  # <--- FIXED: Added platform argument
    ) -> ServerConfig:
        """
        Get configuration for a server (with caching)
        Creates default if doesn't exist
        """
        # Check cache first
        if server_id in self._cache:
            return self._cache[server_id]
        
        # Load from database (Pass platform down to DB)
        # Note: Ensure your db.get_server_config also accepts platform if you updated it!
        # If your DB method doesn't take platform yet, remove `platform=platform` below.
        config_dict = await self.db.get_server_config(server_id, platform=platform)
        
        if config_dict:
            if isinstance(config_dict, str):
                import json
                try:
                    config_dict = json.loads(config_dict)
                except json.JSONDecodeError:
                    # Fallback if corrupted
                    config_dict = {}
                    
            config = ServerConfig.from_dict(config_dict)
            # Update name if changed
            if server_name != "Unknown":
                config.server_name = server_name
        else:
            # Create default config
            config = ServerConfig(server_id=server_id, server_name=server_name)
            # Save it so it exists in DB
            await self.save_config(config, platform=platform)
        
        # Cache it
        self._cache[server_id] = config
        
        return config
    
    async def save_config(
        self, 
        config: ServerConfig, 
        platform: str = 'discord' 
    ):
        # Pass platform down to DB
        await self.db.save_server_config(config.to_dict(), platform=platform)
        
        # Update cache
        self._cache[config.server_id] = config
    
    def invalidate_cache(self, server_id: str):
        if server_id in self._cache:
            del self._cache[server_id]