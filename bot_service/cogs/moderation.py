import discord
from discord.ext import commands
from datetime import timedelta
from typing import Optional, Dict, Any
import asyncio
from ..database import ViolationLevel
from ..models.server_config import ServerConfig
import aiohttp
import logging
import json
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ModerationCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.config_manager = bot.config_manager
        self.moderation_service = bot.moderation_service
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        
        if message.author.bot:
            return
        
        if not message.guild:
            return
        
        server_id = str(message.guild.id)
        
        # Get server configuration
        config = await self.config_manager.get_config(server_id, message.guild.name)
        
        # Check if auto-moderation is enabled
        if not config.auto_moderate:
            return
        
        # Check exempt channels
        if str(message.channel.id) in config.exempt_channel_ids:
            return
        
        # Check monitored channels (if set, ONLY moderate these)
        if config.monitored_channel_ids and str(message.channel.id) not in config.monitored_channel_ids:
            return
        
        # Check exempt roles
        user_role_ids = [str(role.id) for role in message.author.roles]
        if any(role_id in config.exempt_role_ids for role_id in user_role_ids):
            return
        
        # Get prediction from API
        result = await self.moderation_service.predict_toxicity(message)
        
        if result is None:
            return
        
        # Get user's violation history for this server
        user_history = await self.bot.db.get_server_user_violations(
            server_id,
            str(message.author.id),
            platform='discord'
        )
        
        # Determine severity using server thresholds
        severity = self.determine_severity(
            text=message.content,
            confidence=result.get("confidence", 0.0),
            is_toxic=result.get("is_toxic", False),
            user_history=user_history,
            config=config
        )
        
        log_id = await self.handle_violation(
            message=message,
            severity=severity,
            confidence=result.get("confidence", 0.0),
            config=config
        )

        if log_id:
            asyncio.create_task(
                self.generate_explanation_async(
                    log_id=log_id,   
                    message=message,
                    severity=severity,
                    config=config
                )
            )
    async def generate_explanation_async(
        self,
        log_id: int,
        message: discord.Message,
        severity: ViolationLevel,
        config: ServerConfig
    ):
        """
        Background task to generate LIME explanation.
        
        This runs asynchronously - the user has already been moderated.
        The explanation is for:
        - Mod logs
        - User appeals
        - Analytics
        """
        try:
            server_id = str(message.guild.id)
            user_id = str(message.author.id)
            
            if not log_id:
                logger.warning("Could not find log entry for explanation")
                return
            
            # Call API for explanation (SLOW - 1-5 seconds)
            async with self.bot.session.post(
                f"{self.bot.config.API_BASE_URL}/explain",
                json={
                    "text": message.content,
                    "num_features": 6,
                    "num_samples": 1000
                },
                timeout=aiohttp.ClientTimeout(total=10.0)
            ) as response:
                if response.status == 200:
                    explanation = await response.json()
                    
                    # Update database with explanation
                    await self.bot.db.update_log_explanation(
                        log_id, explanation
                    )
                    
                    # Send to mod log if configured
                    if config.log_channel_id and severity in [ViolationLevel.MEDIUM, ViolationLevel.HIGH]:
                        await self.send_explanation_to_mods(
                            message, severity, explanation, config
                        )
                    
                    logger.info(f"✅ Generated explanation for log {log_id}")
                else:
                    logger.warning(f"Explanation API returned {response.status}")
                    
        except asyncio.TimeoutError:
            logger.warning("Explanation generation timed out")
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}", exc_info=True)
    
    async def send_explanation_to_mods(
        self,
        message: discord.Message,
        severity: ViolationLevel,
        explanation: Dict[str, Any],
        config: ServerConfig
    ):
        """Send detailed explanation to mod log channel"""
        
        channel = message.guild.get_channel(int(config.log_channel_id))
        
        if not channel:
            return
        
        # Format trigger words
        trigger_words = explanation.get('trigger_words', [])
        top_toxic = [w for w in trigger_words if w['category'] == 'toxic'][:3]
        
        embed = discord.Embed(
            title=f"{'🚨' if severity == ViolationLevel.HIGH else '⚠️'} Moderation Action",
            color=discord.Color.red() if severity == ViolationLevel.HIGH else discord.Color.orange()
        )
        
        embed.add_field(name="User", value=message.author.mention, inline=True)
        embed.add_field(name="Channel", value=message.channel.mention, inline=True)
        embed.add_field(name="Severity", value=severity.value, inline=True)
        
        # Show trigger words
        if top_toxic:
            words_text = "\n".join([
                f"🔴 **{w['word']}** (impact: {w['score']:+.2f})"
                for w in top_toxic
            ])
            embed.add_field(
                name="Key Words Detected",
                value=words_text,
                inline=False
            )
        
        # Show message
        embed.add_field(
            name="Message",
            value=f"```{message.content[:500]}```",
            inline=False
        )
        
        # Add explanation quality indicator
        if explanation.get('cached'):
            embed.set_footer(text="Explanation: Cached")
        else:
            embed.set_footer(text=f"Explanation: {explanation.get('num_samples', 0)} samples analyzed")
        
        embed.timestamp = message.created_at
        
        await channel.send(embed=embed)

    def determine_severity(
        self,
        text: str,
        confidence: float,
        is_toxic: bool,
        user_history: Dict[str, Any],
        config: ServerConfig
    ) -> ViolationLevel:
        """Determine severity using server-specific thresholds"""
        
        # Import TextCleaner
        from ..bot import TextCleaner
        
        # Hard slurs always high severity
        if TextCleaner.contains_hard_slur(text):
            return ViolationLevel.HIGH
        
        # Not toxic
        if not is_toxic:
            return ViolationLevel.SAFE
        
        # Uncertainty range (server-configurable)
        if config.uncertainty_min <= confidence <= config.uncertainty_max:
            return ViolationLevel.UNCERTAIN
        
        # Escalate based on history
        violation_count = user_history.get('count', 0)
        
        if violation_count >= 2:
            if confidence > config.threshold_high:
                return ViolationLevel.HIGH
            elif confidence > config.threshold_medium:
                return ViolationLevel.MEDIUM
        
        # Use server thresholds
        if confidence >= config.threshold_high:
            return ViolationLevel.HIGH
        elif confidence >= config.threshold_medium:
            return ViolationLevel.MEDIUM
        elif confidence >= config.threshold_low:
            return ViolationLevel.LOW
        else:
            return ViolationLevel.SAFE
    
    async def handle_violation(
        self,
        message: discord.Message,
        severity: ViolationLevel,
        confidence: float,
        config: ServerConfig
    ):
        """Handle violation based on server configuration"""
        
        server_id = str(message.guild.id)
        user_id = str(message.author.id)
        log_id = None
        if severity == ViolationLevel.SAFE:
            return
        
        elif severity == ViolationLevel.UNCERTAIN:
            # Flag for review
            log_id = await self.bot.db.log_event(
                user_id=user_id,
                server_id=server_id,
                platform='discord',
                message=message.content,
                score=confidence,
                severity=severity.value,
                action="FLAGGED_REVIEW",
                metadata={"channel_id": str(message.channel.id)}
            )
            
            if config.require_human_review:
                await self.notify_mods(message, severity, confidence, config)
        
        elif severity == ViolationLevel.LOW:
            action = config.low_severity_action
            log_id = await self.execute_action(
                message, action, severity, confidence, config,
                timeout_duration=config.timeout_duration_low
            )
        
        elif severity == ViolationLevel.MEDIUM:
            action = config.medium_severity_action
            log_id = await self.execute_action(
                message, action, severity, confidence, config,
                timeout_duration=config.timeout_duration_medium
            )
        
        elif severity == ViolationLevel.HIGH:
            action = config.high_severity_action
            log_id = await self.execute_action(
                message, action, severity, confidence, config,
                timeout_duration=config.timeout_duration_high
            )

        return log_id
    
    
    async def execute_action(
        self,
        message: discord.Message,
        action: str,
        severity: ViolationLevel,
        confidence: float,
        config: ServerConfig,
        timeout_duration: int = 10
    ):
        """Execute the configured moderation action"""
        
        server_id = str(message.guild.id)
        user_id = str(message.author.id)
        log_id = None
        
        # Add violation to server-specific count
        strike_count = await self.bot.db.add_violation(
            user_id=user_id,
            server_id=server_id,
            severity=severity.value,
            platform='discord'
        )
        
        # Check if strikes have escalated the action
        if strike_count >= config.strikes_before_ban:
            action = "ban"
        elif strike_count >= config.strikes_before_kick:
            action = "kick"
        elif strike_count >= config.strikes_before_timeout:
            action = "timeout"
        
        # ─────────────────────────────────────────────────────────────
        # EXECUTE ACTION
        # NOTE: explanation=None on ALL log_event calls
        #       LIME explanation is generated separately by !explain
        # ─────────────────────────────────────────────────────────────
        
        if action == "warn":
            warning = config.warning_message_template.format(
                mention=message.author.mention
            )
            
            delete_after = config.warning_delete_delay if config.delete_after_timeout else None
            await message.channel.send(warning, delete_after=delete_after)
            
            if config.send_dm_warnings:
                await self.send_dm_warning(message.author, severity, strike_count, config)
            
            log_id = await self.bot.db.log_event(
                user_id=user_id,
                server_id=server_id,
                platform='discord',
                message=message.content,
                score=confidence,
                severity=severity.value,
                action=f"WARN_STRIKE_{strike_count}",
                explanation=None    # ← LIME generated later by !explain
            )
        
        elif action == "delete":
            await message.delete()
            
            strike_msg = config.strike_message_template.format(
                mention=message.author.mention,
                count=strike_count,
                max=config.strikes_before_ban
            )
            
            delete_after = config.warning_delete_delay if config.delete_after_timeout else None
            await message.channel.send(strike_msg, delete_after=delete_after)
            
            log_id = await self.bot.db.log_event(
                user_id=user_id,
                server_id=server_id,
                platform='discord',
                message=message.content,
                score=confidence,
                severity=severity.value,
                action=f"DELETE_STRIKE_{strike_count}",
                explanation=None    # ← LIME generated later by !explain
            )
        
        elif action == "timeout":
            await message.delete()
            
            duration = timedelta(minutes=timeout_duration)
            try:
                await message.author.timeout(
                    duration,
                    reason=f"{severity.value} violation - Strike {strike_count}"
                )
                
                timeout_msg = config.timeout_message_template.format(
                    mention=message.author.mention,
                    duration=f"{timeout_duration} minutes",
                    reason=f"{severity.value} violation"
                )
                
                await message.channel.send(timeout_msg, delete_after=60)
                
                log_id = await self.bot.db.log_event(
                    user_id=user_id,
                    server_id=server_id,
                    platform='discord',
                    message=message.content,
                    score=confidence,
                    severity=severity.value,
                    action=f"TIMEOUT_{timeout_duration}M_STRIKE_{strike_count}",
                    explanation=None    # ← LIME generated later by !explain
                )
                
            except discord.Forbidden:
                logger.error(f"❌ Failed to timeout {message.author}: Missing permissions")
            except discord.HTTPException as e:
                logger.error(f"❌ Failed to timeout {message.author}: {e}")
            
            if severity == ViolationLevel.HIGH:
                await self.notify_mods(message, severity, confidence, config)
        
        elif action == "kick":
            await message.delete()
            
            try:
                await message.author.kick(
                    reason=f"Strike {strike_count}/{config.strikes_before_ban}"
                )
                
                await message.channel.send(
                    f"🚫 {message.author.mention} has been kicked (Strike {strike_count})"
                )
                
                log_id = await self.bot.db.log_event(
                    user_id=user_id,
                    server_id=server_id,
                    platform='discord',
                    message=message.content,
                    score=confidence,
                    severity=severity.value,
                    action=f"KICK_STRIKE_{strike_count}",
                    explanation=None    
                )
                
            except discord.Forbidden:
                logger.error(f"❌ Failed to kick {message.author}: Missing permissions")
            
            await self.notify_mods(message, severity, confidence, config)
        
        elif action == "ban":
            await message.delete()
            
            try:
                await message.author.ban(
                    reason=f"Strike {strike_count} - Automatic ban"
                )
                
                await message.channel.send(
                    f"🚫 {message.author.mention} has been banned (Strike {strike_count})"
                )
                
                log_id = await self.bot.db.log_event(
                    user_id=user_id,
                    server_id=server_id,
                    platform='discord',
                    message=message.content,
                    score=confidence,
                    severity=severity.value,
                    action=f"BAN_STRIKE_{strike_count}",
                    explanation=None    # ← LIME generated later by !explain
                )
                
            except discord.Forbidden:
                logger.error(f"❌ Failed to ban {message.author}: Missing permissions")
            
            await self.notify_mods(message, severity, confidence, config)
        
        return log_id

    async def send_dm_warning(
        self,
        user: discord.Member,
        severity: ViolationLevel,
        strike_count: int,
        config: ServerConfig
    ):
        """Send DM warning to user"""
        try:
            embed = discord.Embed(
                title="⚠️ Moderation Warning",
                description=f"Your message in **{user.guild.name}** violated community guidelines.",
                color=discord.Color.orange()
            )
            
            embed.add_field(
                name="Severity",
                value=severity.value,
                inline=True
            )
            
            embed.add_field(
                name="Strike Count",
                value=f"{strike_count}/{config.strikes_before_ban}",
                inline=True
            )
            
            embed.add_field(
                name="What happens next?",
                value=f"""
                • {config.strikes_before_timeout} strikes: Timeout
                • {config.strikes_before_kick} strikes: Kick
                • {config.strikes_before_ban} strikes: Ban
                """,
                inline=False
            )
            
            await user.send(embed=embed)
        except discord.Forbidden:
            # User has DMs disabled
            pass

    @commands.command(name='whyflagged', aliases=['explain', 'why'])
    @commands.cooldown(1, 60, commands.BucketType.user)  # 1 per minute per user
    async def explain_my_violation(self, ctx):
        """
        Let users see why their last message was flagged.
        Usage: !why
        """
        logger.info(f"DEBUG !explain triggered by {ctx.author.name}")
        logger.info(f"DEBUG server_id: {ctx.guild.id}, user_id: {ctx.author.id}")
        server_id = str(ctx.guild.id)
        user_id = str(ctx.author.id)
        
        # 1. Feedback to user (Ephemeral delete)
        status_msg = await ctx.send("🔍 Looking up your recent history...", delete_after=5)

        # 2. Get user's most recent violation
        # Ensure this DB function returns the full log including 'message' and 'timestamp'
        recent_log = await self.bot.db.get_latest_user_violation(
            server_id, user_id, platform='discord', hours=24 
        )

        logger.info(f"DEBUG recent_log: {recent_log}")
        
        if not recent_log:
            await ctx.send("✅ No violations found in the last 24 hours.", delete_after=10)
            return

        logger.info(f"DEBUG message: '{recent_log.get('message', 'MISSING')}'")
        logger.info(f"DEBUG existing explanation: {recent_log.get('explanation')}")

        # 3. Check/Generate Explanation
        explanation = recent_log.get('explanation')
        if isinstance(explanation, str):
            import json
            try:
                explanation = json.loads(explanation)
            except json.JSONDecodeError:
                explanation = None

        if not explanation:
            logger.info(f"DEBUG calling /explain API...")

            await status_msg.edit(content="⏳ Generating deep analysis... (approx 3s)")
            
            payload = {
                "text": recent_log['message'], 
                "user_id": user_id,           
                "channel_id": str(ctx.channel.id),
                "num_features": 6
            }
            logger.info(f"DEBUG payload: {payload}")
            logger.info(f"DEBUG API URL: {self.bot.config.API_BASE_URL}/explain")
            try:
                async with self.bot.session.post(
                    f"{self.bot.config.API_BASE_URL}/explain",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10.0),
                    ssl=False
                ) as response:
                    logger.info(f"DEBUG /explain response status: {response.status}")
                    response_text = await response.text()
                    logger.info(f"DEBUG /explain raw response: {response_text[:200]}")

                    if response.status == 200:
                        explanation = await response.json()
                        await self.bot.db.update_log_explanation(recent_log['id'], explanation)
                    else:
                        await ctx.send("❌ Analysis failed. Please ask a moderator.", delete_after=10)
                        return
            except Exception as e:
                logger.error(f"Explanation API failed: {e}")
                await ctx.send("❌ System busy. Try again later.", delete_after=10)
                return

        # 4. Use Shared Embed Creator (Consistency!)
        
        # We recreate the 'user' object if needed, or just use ctx.author
        embed = create_explanation_embed(
            explanation=explanation,
            original_text=recent_log['message'],
            user=ctx.author,
            fallback_score=recent_log.get('toxicity_score', 0.0)
        )
        
        # Add a footer specific to this command
        ts = recent_log['timestamp']
        # Convert datetime to Discord timestamp format <t:1234567890:R>
        time_str = f"<t:{int(ts.timestamp())}:R>" if ts else "recently"
        embed.set_footer(text=f"Violation recorded {time_str}")
        # 5. Safe Send (Handle Blocked DMs)
        feedback_view = FeedbackView(log_id=recent_log['id'], log_data=recent_log, db=self.bot.db, user_id=str(ctx.author.id), server_id=str(ctx.guild.id))
        try:
            await ctx.author.send(content="**Was this flag correct?**", embed=embed, view=feedback_view)   
            await ctx.message.delete()
            await status_msg.edit(content="Example sent to your DMs! chk 📬")
        except discord.Forbidden:
            await status_msg.edit(content=f"{ctx.author.mention} 🔒 I couldn't DM you. Here is your report:", view=feedback_view, embed=embed, delete_after=120)
            
    async def notify_mods(
        self,
        message: discord.Message,
        severity: ViolationLevel,
        confidence: float,
        config: ServerConfig
    ):
        """Send notification to mod channel"""
        
        channel_id = config.alert_channel_id if severity == ViolationLevel.HIGH else config.log_channel_id
        
        if not channel_id:
            return
        
        channel = message.guild.get_channel(int(channel_id))
        
        if not channel:
            return
        
        embed = discord.Embed(
            title=f"{'🚨' if severity == ViolationLevel.HIGH else '⚠️'} Moderation Action",
            color=discord.Color.red() if severity == ViolationLevel.HIGH else discord.Color.orange()
        )
        
        embed.add_field(name="User", value=message.author.mention, inline=True)
        embed.add_field(name="Channel", value=message.channel.mention, inline=True)
        embed.add_field(name="Severity", value=severity.value, inline=True)
        embed.add_field(name="Confidence", value=f"{confidence:.1%}", inline=True)
        
        embed.add_field(
            name="Message",
            value=f"```{message.content[:500]}```",
            inline=False
        )
        
        embed.timestamp = message.created_at
        
        await channel.send(embed=embed)

async def setup(bot):
    await bot.add_cog(ModerationCog(bot))


def create_explanation_embed(explanation: dict, original_text: str, user: discord.User, fallback_score: float = 0.0):
    """
    Converts API JSON explanation into a Discord Embed.
    """
    if isinstance(explanation, str):
        try:
            explanation = json.loads(explanation)
        except json.JSONDecodeError:
            # If it fails to parse, create a fallback dict so .get() doesn't crash
            explanation = {}
    # 1. Extract Core Data

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"create_explanation_embed received keys: {list(explanation.keys())}")
    logger.info(f"original_text: '{original_text[:50] if original_text else 'EMPTY'}'")
    # ────────────────────────────────────────────────────────────────
    score = explanation.get('toxic_probability', 0.0)
    if score is None:
        score = fallback_score
    triggers = explanation.get('trigger_words', [])
    features = explanation.get('features_used', {})
    
    # 2. Determine Theme (Color & Icon)
    if score > 0.8:
        color = 0xFF0000  # Red
        status = "🚨 Highly Toxic"
    elif score > 0.5:
        color = 0xFFA500  # Orange
        status = "⚠️ Suspicious"
    else:
        color = 0x00FF00  # Green
        status = "✅ Safe"

    # 3. Build the Embed Shell
    embed = discord.Embed(
        title=f"{status} (Confidence: {score:.1%})",
        description=f"Analysis for {user.mention}",
        color=color
    )
    
    # 4. Visual Progress Bar
    # Example: 0.85 -> "▓▓▓▓▓▓▓▓░░"
    filled = int(score * 10)
    bar = "▓" * filled + "░" * (10 - filled)
    embed.add_field(name="Toxicity Score", value=f"`{bar}` **{score:.1%}**", inline=False)

    # 5. Format Trigger Words (The "Why")
    # Filter for words that actually mattered (>5% impact)
    significant_triggers = [
        f"**{t['word']}** (+{t['score']:.0%})" 
        for t in triggers 
        if t['category'] == 'toxic' and t['score'] > 0.05
    ]
    
    if significant_triggers:
        # Join them: "**trash** (+85%), **stupid** (+12%)"
        embed.add_field(name="🔴 Trigger Words", value=", ".join(significant_triggers), inline=False)
    else:
        # If score is high but no words are toxic, it's the History/Context
        if score > 0.5:
            embed.add_field(
                name="⚠️ Context Warning", 
                value="Flagged due to **User History** or **Channel Pattern**.", 
                inline=False
            )

    # 6. Context Insights (Hidden Factors)
    # Check the 'context_frozen' section for red flags
    context = features.get('context_frozen', {})
    context_alerts = []
    
    if context.get('user_bad_ratio_7d', 0) > 0.2:
        context_alerts.append("• **High Violation Rate:** User has frequent toxic history.")
    if context.get('is_new_to_channel', 0) == 1:
        context_alerts.append("• **New Account:** Stricter filtering applied.")
    if context.get('channel_toxicity_ratio', 0) > 0.1:
        context_alerts.append("• **Heated Channel:** This channel is currently volatile.")

    if context_alerts:
        embed.add_field(name="📜 Context Factors", value="\n".join(context_alerts), inline=False)

    # 7. Original Message (Truncated)
    # Wrap in ||spoilers|| so we don't show toxic text openly
    display_text = original_text[:100] + "..." if len(original_text) > 100 else original_text
    embed.add_field(name="Analyzed Content", value=f"||{display_text}||", inline=False)
    
    embed.set_footer(text="Powered by LIME & Hybrid AI")
    return embed

# ═══════════════════════════════════════════════════════════════
# bot_service/cogs/moderation.py
# UPDATE: Add feedback buttons to !explain DM
# ═══════════════════════════════════════════════════════════════

# Add this View class to your moderation.py:

class FeedbackView(discord.ui.View):
    """
    Feedback buttons shown in !explain DM.
    User can dispute the model's decision.
    """
    
    def __init__(self, log_id: int, log_data: dict, db, user_id: str, server_id: str):
        super().__init__(timeout=None)  # Persistent buttons
        self.log_id = log_id
        self.log_data = log_data
        self.db = db
        self.user_id = str(user_id)
        self.server_id = str(server_id)
    
    @discord.ui.button(
        label="✅ Model was correct",
        style=discord.ButtonStyle.success,
        custom_id=f"feedback_correct"
    )
    async def feedback_correct(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button
    ):
        """User confirms model was correct"""
        await self.db.record_user_dispute(
            log_id=self.log_id,
            user_id=self.user_id,
            server_id=self.server_id,
            user_claimed_label=1, # Admitting it is Toxic
            platform='discord',
            dispute_reason="User admitted guilt"
        )
        await interaction.response.send_message(
            "✅ Thanks for confirming! This helps improve our AI.",
            ephemeral=True
        )
        
        # No database action needed - user agrees with model
        
        # Disable buttons
        for child in self.children:
            child.disabled = True
        await interaction.message.edit(view=self)
    
    @discord.ui.button(
        label="❌ Model was wrong",
        style=discord.ButtonStyle.danger,
        custom_id=f"feedback_wrong"
    )
    async def feedback_wrong(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button
    ):
        """User disputes the model's decision"""
        
        # Open modal to get dispute reason
        modal = DisputeReasonModal(
            log_id=self.log_id,
            log_data=self.log_data,
            db=self.db,
            server_id=self.server_id
        )
        await interaction.response.send_modal(modal)
        
        # Disable buttons after modal submission
        for child in self.children:
            child.disabled = True
        await interaction.message.edit(view=self)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Ensure only the person who got flagged can click the buttons."""
        if str(interaction.user.id) != self.user_id:
            await interaction.response.send_message("❌ This is not your report.", ephemeral=True)
            return False
        return True
    
class DisputeReasonModal(discord.ui.Modal, title="Dispute This Decision"):
    """
    Modal to collect why user thinks model was wrong.
    """
    
    reason = discord.ui.TextInput(
        label="Why was this flagged incorrectly?",
        style=discord.TextStyle.paragraph,
        placeholder="Example: This was sarcasm, not actually toxic\nExample: This is gaming slang, not harassment",
        required=False,
        max_length=500
    )
    
    def __init__(self, log_id: int, log_data: dict, db, server_id: str):
        super().__init__()
        self.log_id = log_id
        self.log_data = log_data
        self.db = db
        self.server_id = server_id
    
    async def on_submit(self, interaction: discord.Interaction):
        """Record the dispute"""
        
        user_id = str(interaction.user.id)
        
        # Determine what model predicted
        predicted_label = 1 if self.log_data['severity'] in ['LOW', 'MEDIUM', 'HIGH'] else 0
        
        # User claims it should be safe (opposite of model)
        user_claimed_label = 0 if predicted_label == 1 else 1
        
        try:
            # Record dispute in database
            feedback_id = await self.db.record_user_dispute(
                log_id=self.log_id,
                user_id=user_id,
                server_id=self.server_id,
                user_claimed_label=user_claimed_label,
                platform='discord',
                dispute_reason=self.reason.value or None
            )
            
            # Confirm to user
            await interaction.response.send_message(
                "📝 **Dispute recorded!**\n\n"
                "An admin will review this decision. If they agree with you, "
                "our AI will learn from this mistake.\n\n"
                "Thank you for helping improve our moderation!",
                ephemeral=True
            )
            
            logger.info(f"Dispute recorded: feedback_id={feedback_id}, log_id={self.log_id}, user={user_id}")
            
        except Exception as e:
            logger.error(f"Failed to record dispute: {e}", exc_info=True)
            await interaction.response.send_message(
                "❌ Failed to record dispute. Please contact an admin.",
                ephemeral=True
            )


# ═══════════════════════════════════════════════════════════════
# UPDATE YOUR explain_my_violation COMMAND
# ═══════════════════════════════════════════════════════════════

# In your existing explain_my_violation command, replace the DM section with:

@commands.command(name='whyflagged', aliases=['explain', 'why'])
@commands.cooldown(1, 15, commands.BucketType.user)  # Reduced cooldown
async def explain_my_violation(self, ctx):
    """
    Let users see why their last message was flagged + dispute it.
    """
    server_id = str(ctx.guild.id)
    user_id = str(ctx.author.id)
    
    status_msg = await ctx.send("🔍 Looking up your recent history...", delete_after=5)
    
    # Get most recent violation
    recent_log = await self.bot.db.get_latest_user_violation(
        server_id, user_id, platform='discord', hours=24 
    )
    
    if not recent_log:
        await ctx.send("✅ No violations found in the last 24 hours.", delete_after=10)
        return
    
    # Generate/get explanation
    explanation = recent_log.get('explanation')
    if isinstance(explanation, str):
        try:
            explanation = json.loads(explanation)
        except json.JSONDecodeError:
            explanation = None
    
    if not explanation:
        await status_msg.edit(content="⏳ Generating deep analysis... (approx 3s)")
        
        payload = {
            "text": recent_log['message'],
            "user_id": user_id,
            "channel_id": str(ctx.channel.id),
            "num_features": 6
        }
        
        try:
            async with self.bot.session.post(
                f"{self.bot.config.API_BASE_URL}/explain",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10.0),
                ssl=False
            ) as response:
                if response.status == 200:
                    explanation = await response.json()
                    await self.bot.db.update_log_explanation(recent_log['id'], explanation)
                else:
                    await ctx.send("❌ Analysis failed. Please ask a moderator.", delete_after=10)
                    return
        except Exception as e:
            logger.error(f"Explanation API failed: {e}")
            await ctx.send("❌ System busy. Try again later.", delete_after=10)
            return
    
    # Create explanation embed
    embed = create_explanation_embed(
        explanation=explanation,
        original_text=recent_log['message'],
        user=ctx.author,
        fallback_score=recent_log.get('toxicity_score', 0.0)
    )
    
    # Add timestamp footer
    ts = recent_log['timestamp']
    time_str = f"<t:{int(ts.timestamp())}:R>" if ts else "recently"
    embed.set_footer(text=f"Violation recorded {time_str}")
    
    # ═══════════════════════════════════════════════════════════
    # ADD FEEDBACK BUTTONS
    # ═══════════════════════════════════════════════════════════
    
    feedback_view = FeedbackView(
        log_id=recent_log['id'],
        log_data=recent_log,
        db=self.bot.db
    )
    
    # Send to DM with feedback buttons
    try:
        await ctx.author.send(
            content="**Was this flag correct?**",
            embed=embed,
            view=feedback_view
        )
        await ctx.message.delete()
        await status_msg.edit(content="✅ Explanation sent to your DMs! 📬", delete_after=10)
        
    except discord.Forbidden:
        # Fallback: send in channel (without buttons for privacy)
        await status_msg.edit(
            content=f"{ctx.author.mention} 🔒 I couldn't DM you. Here is your report:",
            delete_after=20
        )
        await ctx.send(embed=embed, delete_after=30)