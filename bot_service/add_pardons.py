import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def migrate():
    conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
    
    print("🔄 Adding pardon columns...")
    
    await conn.execute('''
        ALTER TABLE server_user_violations
        ADD COLUMN IF NOT EXISTS pardoned        BOOLEAN   DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS pardoned_at     TIMESTAMP,
        ADD COLUMN IF NOT EXISTS pardoned_by     TEXT,
        ADD COLUMN IF NOT EXISTS pardon_reason   TEXT;
    ''')
    
    print("✅ Done")
    await conn.close()

asyncio.run(migrate())