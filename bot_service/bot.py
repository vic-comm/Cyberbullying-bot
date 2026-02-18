import os
import asyncio
import re
import aiohttp
import discord
from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from unidecode import unidecode
from discord.ext import commands

# Import internal modules
from bot_service.database import DatabaseManager
from bot_service.config import Config
from bot_service.core.config_manager import ConfigManager

class TextCleaner:
    """
    Utility class for normalizing text to catch evasion tactics (leetspeak, spacing).
    """
    LEETSPEAK_MAP = {
        '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's',
        '@': 'a', '$': 's', '!': 'i', '7': 't', '8': 'b'
    }
    
    HARD_SLURS = ["nigger", "faggot", "retard", "n1gger", "f4ggot"]
    
    @classmethod
    def clean(cls, text: str) -> str:
        """Normalize text: remove accents, fix leetspeak, remove weird spacing."""
        text = unidecode(text)
        text = re.sub(r'\s+', ' ', text) # Collapse whitespace
        text = text.replace(' ', '')     # Remove spaces (catch "b a d w o r d")
        
        for char, replacement in cls.LEETSPEAK_MAP.items():
            text = text.replace(char, replacement)
        
        return text.lower().strip()
    
    @classmethod
    def contains_hard_slur(cls, text: str) -> bool:
        cleaned = cls.clean(text)
        return any(slur in cleaned for slur in cls.HARD_SLURS)
    
    @classmethod
    def calculate_caps_ratio(cls, text: str) -> float:
        if not text:
            return 0.0
        return sum(1 for c in text if c.isupper()) / len(text)


class ModerationService:
    def __init__(self, bot):
        self.bot = bot
        self.api_url = f"{bot.config.API_BASE_URL}/predict"
        
    async def predict_toxicity(self, message: discord.Message) -> Optional[Dict[str, Any]]:
        server_id = str(message.guild.id) if message.guild else None
        
        user_history = await self.bot.db.get_server_user_violations(
            server_id, str(message.author.id)
        )
        
        payload = {
            "text": TextCleaner.clean(message.content),
            "user_id": str(message.author.id),
            "msg_len": len(message.content),
            "caps_ratio": TextCleaner.calculate_caps_ratio(message.content),
            "account_age_days": (datetime.now(timezone.utc) - message.author.created_at).days,
            "previous_violations": user_history.get('count', 0),
            "server_id": server_id
        }
        
        try:
            async with self.bot.session.post(
                self.api_url, 
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    print(f"⚠️ API Error {response.status}: {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            print("⚠️ API timeout - falling back to rule-based detection")
            return self._fallback_detection(message)
        except Exception as e:
            print(f"⚠️ API connection failed: {e}")
            return self._fallback_detection(message)
    
    def _fallback_detection(self, message: discord.Message) -> Dict[str, Any]:
        contains_slur = TextCleaner.contains_hard_slur(message.content)
        return {
            "is_toxic": contains_slur,
            "confidence": 0.95 if contains_slur else 0.0,
            "fallback": True
        }


class ModerationBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True

        super().__init__(command_prefix='!', intents=intents)
        self.config = Config()
        self.db = DatabaseManager(self.config.DATABASE_URL)
        self.session: Optional[aiohttp.ClientSession] = None
        self.config_manager: Optional[ConfigManager] = None
        self.moderation_service: Optional[ModerationService] = None

    async def setup_hook(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0))
        await self.db.init_db()
        
        # Initialize Services
        self.config_manager = ConfigManager(self.db)
        self.moderation_service = ModerationService(self)
        
        # Load Cogs
        await self.load_extension('bot_service.cogs.admin')
        await self.load_extension('bot_service.cogs.moderation')
        
        print("✅ Bot setup complete (Database & Cogs loaded)")

    async def close(self):
        if self.session:
            await self.session.close()
        await self.db.close()
        await super().close()

    async def on_guild_join(self, guild: discord.Guild):
        print(f"➕ Joined new server: {guild.name} (ID: {guild.id})")

        # 1. Initialize Default Config Immediately
        await self.config_manager.get_config(str(guild.id), guild.name)

        # 2. Find a place to say hello
        channel = guild.system_channel
        if not channel:
            for c in guild.text_channels:
                if c.permissions_for(guild.me).send_messages:
                    channel = c
                    break

        # 3. Send Welcome
        if channel:
            embed = discord.Embed(
                title="🛡️ Thanks for adding ModerationBot!",
                description=(
                    "I am now protecting this server.\n\n"
                    "**Next Steps:**\n"
                    "1️⃣ Run `/setup` to configure rules.\n"
                    "2️⃣ Run `/setlogchannel` to enable logging.\n"
                    "3️⃣ Ensure my role is above users I need to moderate."
                ),
                color=discord.Color.green()
            )
            try:
                await channel.send(embed=embed)
            except discord.Forbidden:
                pass

bot = ModerationBot()

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")
    print(f"👀 Monitoring {len(bot.guilds)} servers")
    print(f"🔗 API: {bot.config.API_BASE_URL}")
    
    # Set status
    await bot.change_presence(activity=discord.Activity(
        type=discord.ActivityType.watching, 
        name=f"{len(bot.guilds)} servers | /help"
    ))
    
    # Sync Slash Commands
    try:
        synced = await bot.tree.sync()
        print(f"✅ Synced {len(synced)} slash commands")
    except Exception as e:
        print(f"❌ Failed to sync commands: {e}")

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("❌ You don't have permission to use this command.", delete_after=5)
    elif isinstance(error, commands.CommandNotFound):
        pass # Ignore invalid commands
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ Missing argument: {error.param}", delete_after=5)
    else:
        print(f"⚠️ Command Error: {error}")

@bot.command()
async def report(ctx: commands.Context):
    if not ctx.message.reference:
        await ctx.send("❌ Please reply to the message you want to report with `!report`", delete_after=10)
        return 
    
    try:
        reported_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        server_id = str(ctx.guild.id) if ctx.guild else None
        
        await bot.db.log_event(
            user_id=str(reported_msg.author.id),
            server_id=server_id,
            message=reported_msg.content,
            score=0.0,
            severity="USER_REPORT",
            action="FLAGGED_REVIEW",
            metadata={
                "reporter_id": str(ctx.author.id),
                "channel_id": str(ctx.channel.id)
            }
        )
        
        await ctx.send(
            "✅ Message flagged for moderator review. Thank you!",
            delete_after=15
        )
        await ctx.message.delete(delay=5)
        
    except discord.NotFound:
        await ctx.send("❌ Could not find that message.", delete_after=10)
    except discord.Forbidden:
        await ctx.send("❌ I don't have permission to access that message.", delete_after=10)

@bot.command()
@commands.has_permissions(administrator=True)
async def modstats(ctx: commands.Context, days: int = 7):
    server_id = str(ctx.guild.id) if ctx.guild else None
    stats = await bot.db.get_moderation_stats(days, server_id)
    
    embed = discord.Embed(
        title=f"📊 Moderation Stats (Last {days} Days)",
        color=discord.Color.blue()
    )
    embed.add_field(name="Total Violations", value=stats['total_violations'], inline=True)
    embed.add_field(name="Unique Users", value=stats['unique_users'], inline=True)
    embed.add_field(name="Messages Deleted", value=stats['deleted_messages'], inline=True)
    embed.add_field(name="Timeouts Issued", value=stats['timeouts'], inline=True)
    embed.add_field(name="Pending Review", value=stats['pending_review'], inline=True)
    
    await ctx.send(embed=embed)

@bot.command()
@commands.has_permissions(administrator=True)
async def clearstrikes(ctx: commands.Context, user: discord.Member):
    """Admin command: Reset a user's record."""
    server_id = str(ctx.guild.id) if ctx.guild else None
    await bot.db.clear_server_violations(server_id, str(user.id))

    if user.is_timed_out():
        try:
            await user.timeout(None, reason="Admin cleared strikes")
            await ctx.send(f"✅ Cleared strikes and removed timeout for {user.mention}")
        except discord.Forbidden:
            await ctx.send(f"✅ Cleared strikes for {user.mention} (Could not remove timeout due to permissions)")
    else:
        await ctx.send(f"✅ Cleared all strikes for {user.mention}")

if __name__ == "__main__":
    if not bot.config.DISCORD_TOKEN:
        print(" Error: DISCORD_TOKEN not found in environment variables")
    else:
        bot.run(bot.config.DISCORD_TOKEN)