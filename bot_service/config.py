import os
from dotenv import load_dotenv

load_dotenv()

class Config():
    DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    API_BASE_URL = os.getenv('BASE_URL')
    DB_PATH = os.getenv("DB_PATH", "bot_memory.db")
 
    UNCERTAINTY_THRESHOLD_LOW = 0.45
    UNCERTAINTY_THRESHOLD_HIGH = 0.65

    TIMEOUT_STRIKE_2 = 10
    TIMEOUT_STRIKE_3 = 24 * 60  

    ENABLE_FALLBACK_DETECTION = True
    ENABLE_USER_REPORTS = True
    LOG_SAFE_MESSAGES = False
