import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Essential Credentials
    DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    API_BASE_URL = os.getenv('API_BASE_URL')
    DATABASE_URL = os.getenv('DATABASE_URL')
 
    # Global Bot Settings (Not Server Specific)
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    ENABLE_API = True 
    ENABLE_FALLBACK_DETECTION = True

    def __init__(self):
        if not self.DISCORD_TOKEN:
            raise ValueError("❌ DISCORD_BOT_TOKEN not found in environment variables!")