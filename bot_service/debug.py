# debug_env.py
import os
import asyncio
import asyncpg
from dotenv import load_dotenv

# Force reload the .env file
load_dotenv(override=True)

url = os.getenv("DATABASE_URL")

print("\n" + "="*40)
print(f"🧐 DEBUGGING DATABASE CONNECTION")
print("="*40)
print(f"Raw URL length: {len(url) if url else 0}")
print(f"Raw URL value:  '{url}'")  # Quotes help see trailing spaces!

if not url:
    print("❌ ERROR: DATABASE_URL is empty!")
    exit()

if " " in url:
    print("❌ ERROR: Found a SPACE in the URL. Please remove it.")
    exit()

if "[" in url or "]" in url:
    print("❌ ERROR: Found brackets [ ] in the URL. Please remove them.")
    exit()

async def test_connect():
    print(f"\n🔌 Attempting to connect to host...")
    try:
        # Try to parse it manually to see where it breaks
        from urllib.parse import urlparse
        result = urlparse(url)
        print(f"   - User: {result.username}")
        print(f"   - Host: {result.hostname}")
        print(f"   - Port: {result.port}")
        
        conn = await asyncpg.connect(url)
        print("\n✅ SUCCESS! Connected to database.")
        await conn.close()
    except Exception as e:
        print(f"\n❌ CONNECTION FAILED: {e}")

asyncio.run(test_connect())