import os
import asyncio
from typing import Optional, Dict, Any
from datetime import timedelta, datetime
import re

import discord
import aiohttp
from discord.ext import commands
from unidecode import unidecode

from database import DatabaseManager, ViolationLevel
from config import Config

class ModerationBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix='!', intents=intents)
        self.config = Config()
        self.db = DatabaseManager(self.config.DB_PATH)
        self.session: Optional[aiohttp.ClientSession] = None

    async def setup_hook(self):
        self.session= aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0))
        await self.db.init_db()

    async def close(self):
        if self.session:
            await self.session.close()
        await self.db.close()
        await super().close()


class TextCleaner:    
    LEETSPEAK_MAP = {
        '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's',
        '@': 'a', '$': 's', '!': 'i', '7': 't', '8': 'b'
    }
    
    HARD_SLURS = ["nigger", "faggot", "retard", "n1gger", "f4ggot"]
    
    @classmethod
    def clean(cls, text: str) -> str:
        """Normalize text to catch evasion tactics"""
        text = unidecode(text)
        
        text = re.sub(r'\s+', ' ', text)
        text = text.replace(' ', '')  
        
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
    def __init__(self, bot: ModerationBot):
        self.bot = bot
        self.api_url = f"{bot.config.API_BASE_URL}/predict"
        
    async def predict_toxicity(self, message: discord.Message) -> Optional[Dict[str, Any]]:
        user_history = await self.bot.db.get_user_violations(str(message.author.id))
        
        payload = {
            "text": TextCleaner.clean(message.content),
            "user_id": str(message.author.id),
            "msg_len": len(message.content),
            "caps_ratio": TextCleaner.calculate_caps_ratio(message.content),
            "account_age_days": (datetime.utcnow() - message.author.created_at).days,
            "previous_violations": user_history.get('count', 0),
            "server_id": str(message.guild.id) if message.guild else None
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
    
    def determine_severity(self, text: str, confidence: float, is_toxic: bool, user_history: Dict[str, Any]) -> ViolationLevel:
        
        if TextCleaner.contains_hard_slur(text):
            return ViolationLevel.HIGH
        
        if not is_toxic:
            return ViolationLevel.SAFE
        
        if 0.45 <= confidence <= 0.65:
            return ViolationLevel.UNCERTAIN
        
        violation_count = user_history.get('count', 0)
        
        if violation_count >= 2:
            if confidence > 0.70:
                return ViolationLevel.HIGH
            elif confidence > 0.60:
                return ViolationLevel.MEDIUM
        
        if confidence > 0.90:
            return ViolationLevel.HIGH
        elif confidence > 0.75:
            return ViolationLevel.MEDIUM
        else:
            return ViolationLevel.LOW


class ModerationHandler:    
    ACTIONS = {
        ViolationLevel.SAFE: None,
        ViolationLevel.UNCERTAIN: "monitor",
        ViolationLevel.LOW: "warn_soft",
        ViolationLevel.MEDIUM: "delete_warn",
        ViolationLevel.HIGH: "escalate"
    }
    
    def __init__(self, bot: ModerationBot):
        self.bot = bot
    
    async def handle(self, message: discord.Message, severity: ViolationLevel, confidence: float):
        user_id = str(message.author.id)
        
        if severity == ViolationLevel.SAFE:
            return
        
        elif severity == ViolationLevel.UNCERTAIN:
            await self.bot.db.log_event(
                user_id=user_id,
                message=message.content,
                score=confidence,
                severity=severity.value,
                action="FLAGGED_REVIEW",
                metadata={"channel_id": str(message.channel.id)}
            )
            print(f" Uncertain: {confidence:.2f} | {message.content[:50]}")
            
        elif severity == ViolationLevel.LOW:
            await message.channel.send(
                f"{message.author.mention}, please keep conversations respectful. "
                "Further violations may result in action.",
                delete_after=15
            )
            await self.bot.db.log_event(user_id, message.content, confidence, severity.value, "WARN_SOFT")
            
        elif severity == ViolationLevel.MEDIUM:
            await message.delete()
            
            count = await self.bot.db.add_violation(user_id, severity)
            
            if count >= 3:
                warning = (
                    f"🚫 {message.author.mention}, your message violated community guidelines. "
                    f"**Strike {count}/3** - Next violation will result in a timeout."
                )
            else:
                warning = (
                    f"🚫 {message.author.mention}, your message was removed for violating guidelines. "
                    f"Strike {count}/3"
                )
            
            await message.channel.send(warning, delete_after=30)
            await self.bot.db.log_event(user_id, message.content, confidence, severity.value, f"DELETE_STRIKE_{count}")
            
        elif severity == ViolationLevel.HIGH:
            await message.delete()
            
            count = await self.bot.db.add_violation(user_id, severity)
            
            if count == 1:
                await message.channel.send(
                    f"🚫 {message.author.mention}, hate speech/harassment is not tolerated. "
                    f"**Official Warning.** Further violations will result in timeout."
                )
                action = "DELETE_WARN_HIGH"
                
            elif count == 2:
                duration = timedelta(minutes=10)
                await message.author.timeout(duration, reason="Toxic behavior - Strike 2")
                await message.channel.send(
                    f"🚫 {message.author.mention} has been muted for 10 minutes. "
                    f"Strike 2/3"
                )
                action = "TIMEOUT_10M"
                
            elif count >= 3:
                duration = timedelta(hours=24)
                await message.author.timeout(duration, reason="Toxic behavior - Strike 3")
                
                if message.guild:
                    mod_channel = discord.utils.get(message.guild.channels, name="mod-logs")
                    if mod_channel:
                        await mod_channel.send(
                            f"🚨 **Automatic 24h Ban Issued**\n"
                            f"User: {message.author.mention}\n"
                            f"Reason: 3rd toxic violation\n"
                            f"Message: `{message.content[:100]}`"
                        )
                
                await message.channel.send(
                    f"🚫 {message.author.mention} has been muted for 24 hours. "
                    f"Moderators have been notified."
                )
                action = "TIMEOUT_24H_ESCALATE"
            
            await self.bot.db.log_event(user_id, message.content, confidence, severity.value, action)


bot = ModerationBot()
moderation_service = ModerationService(bot)
moderation_handler = ModerationHandler(bot)

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")
    print(f"👀 Monitoring {len(bot.guilds)} servers")
    print(f"🔗 API: {bot.config.API_BASE_URL}")

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    
    if not message.guild:
        return
    
    result = await moderation_service.predict_toxicity(message)

    if result is None:
        await bot.process_commands(message)
        return
    
    user_history = await bot.db.get_user_violations(str(message.author.id))

    severity = moderation_service.determine_severity(
        text=message.content,
        confidence=result.get("confidence", 0.0),
        is_toxic=result.get("is_toxic", False),
        user_history=user_history
    )

    await moderation_handler.handle(message, severity, result.get("confidence", 0.0))

    await bot.process_commands(message)

@bot.command()
async def report(ctx: commands.Context):
    if not ctx.message.reference:
        await ctx.send("❌ Please reply to the message you want to report with `!report`", delete_after=10)
        return 
    try:
        reported_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        await bot.db.log_event(user_id=str(reported_msg.author.id),
                                message=reported_msg.content,
                                score=0.0,
                                severity="USER_REPORT",
                                action="FLAGGED_REVIEW",
                                metadata={
                                    "reporter_id": str(ctx.author.id),
                                    "channel_id": str(ctx.channel.id)
                                })
        await ctx.send(
            "Message flagged for moderator review. "
            "Thank you for helping keep the community safe.",
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
    stats = await bot.db.get_moderation_stats(days)
    
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
    await bot.db.clear_violations(str(user.id))
    await ctx.send(f"Cleared all strikes for {user.mention}")


if __name__ == "__main__":
    if not bot.config.DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in environment variables")
    else:
        bot.run(bot.config.DISCORD_TOKEN)