"""
Admin commands for server configuration
Only administrators can use these commands
"""
import discord
from discord.ext import commands
from discord import app_commands
from typing import Optional

from shared.models import ServerConfig, ModerationAction
from shared.core.config_manager import ConfigManager

class AdminCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.config_manager: ConfigManager = bot.config_manager
    
    @app_commands.command(name="setup")
    @app_commands.default_permissions(administrator=True)
    async def setup_wizard(self, interaction: discord.Interaction):
        """Interactive setup wizard for moderation configuration"""
        
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message(
                "❌ You need Administrator permission to use this command.",
                ephemeral=True
            )
            return
        
        # Create setup modal
        class SetupModal(discord.ui.Modal, title="Moderation Setup Wizard"):
            strikes_timeout = discord.ui.TextInput(
                label="Strikes before timeout",
                placeholder="Default: 3",
                default="3",
                max_length=2,
                required=True
            )
            
            strikes_kick = discord.ui.TextInput(
                label="Strikes before kick",
                placeholder="Default: 5",
                default="5",
                max_length=2,
                required=True
            )
            
            strikes_ban = discord.ui.TextInput(
                label="Strikes before ban",
                placeholder="Default: 7",
                default="7",
                max_length=2,
                required=True
            )
            
            timeout_high = discord.ui.TextInput(
                label="High severity timeout (minutes)",
                placeholder="Default: 1440 (24 hours)",
                default="1440",
                max_length=5,
                required=True
            )
            
            def __init__(self, config_manager):
                super().__init__()
                self.config_manager = config_manager

            async def on_submit(self, modal_interaction: discord.Interaction):
                server_id = str(modal_interaction.guild.id)
                
                # Get current config
                config = await self.config_manager.get_config(
                    server_id,
                    modal_interaction.guild.name,
                    platform='discord'
                )
                
                # Update values
                try:
                    config.strikes_before_timeout = int(self.strikes_timeout.value)
                    config.strikes_before_kick = int(self.strikes_kick.value)
                    config.strikes_before_ban = int(self.strikes_ban.value)
                    config.timeout_duration_high = int(self.timeout_high.value)
                    
                    # Save
                    await self.config_manager.save_config(config, platform='discord')
                    
                    embed = discord.Embed(
                        title="✅ Configuration Saved",
                        description="Moderation settings have been updated!",
                        color=discord.Color.green()
                    )
                    
                    embed.add_field(
                        name="Strike System",
                        value=f"""
                        Timeout: {config.strikes_before_timeout} strikes
                        Kick: {config.strikes_before_kick} strikes
                        Ban: {config.strikes_before_ban} strikes
                        """,
                        inline=False
                    )
                    
                    embed.add_field(
                        name="High Severity Timeout",
                        value=f"{config.timeout_duration_high} minutes",
                        inline=False
                    )
                    
                    await modal_interaction.response.send_message(embed=embed, ephemeral=True)
                    
                except ValueError:
                    await modal_interaction.response.send_message(
                        "❌ Invalid input. Please enter numbers only.",
                        ephemeral=True
                    )
        
        await interaction.response.send_modal(SetupModal(self.config_manager))
        
    @app_commands.command(name="setlogchannel")
    @app_commands.describe(channel="Channel for moderation logs")
    @app_commands.default_permissions(administrator=True)
    async def set_log_channel(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel
    ):
        """Set the channel for moderation logs"""
        
        server_id = str(interaction.guild.id)
        config = await self.config_manager.get_config(server_id, interaction.guild.name, platform='discord')
        
        config.log_channel_id = str(channel.id)
        await self.config_manager.save_config(config, platform='discord')
        
        await interaction.response.send_message(
            f"✅ Moderation logs will be sent to {channel.mention}",
            ephemeral=True
        )
    
    @app_commands.command(name="setalertchannel")
    @app_commands.describe(channel="Channel for high-severity alerts")
    @app_commands.default_permissions(administrator=True)
    async def set_alert_channel(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel
    ):
        """Set the channel for high-severity alerts"""
        
        server_id = str(interaction.guild.id)
        config = await self.config_manager.get_config(server_id, interaction.guild.name, platform='discord')
        
        config.alert_channel_id = str(channel.id)
        await self.config_manager.save_config(config, platform='discord')
        
        await interaction.response.send_message(
            f"✅ High-severity alerts will be sent to {channel.mention}",
            ephemeral=True
        )
    
    @app_commands.command(name="exemptchannel")
    @app_commands.describe(channel="Channel to exempt from moderation")
    @app_commands.default_permissions(administrator=True)
    async def exempt_channel(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel
    ):
        """Exempt a channel from automatic moderation"""
        
        server_id = str(interaction.guild.id)
        config = await self.config_manager.get_config(server_id, interaction.guild.name, platform='discord')
        
        channel_id = str(channel.id)
        
        if channel_id in config.exempt_channel_ids:
            await interaction.response.send_message(
                f"⚠️ {channel.mention} is already exempt",
                ephemeral=True
            )
            return
        
        config.exempt_channel_ids.append(channel_id)
        await self.config_manager.save_config(config, platform='discord')
        
        await interaction.response.send_message(
            f"✅ {channel.mention} is now exempt from auto-moderation",
            ephemeral=True
        )
    
    @app_commands.command(name="unexemptchannel")
    @app_commands.describe(channel="Channel to remove exemption from")
    @app_commands.default_permissions(administrator=True)
    async def unexempt_channel(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel
    ):
        """Remove channel exemption from moderation"""
        
        server_id = str(interaction.guild.id)
        config = await self.config_manager.get_config(server_id, interaction.guild.name, platform='discord')
        
        channel_id = str(channel.id)
        
        if channel_id not in config.exempt_channel_ids:
            await interaction.response.send_message(
                f"⚠️ {channel.mention} is not currently exempt",
                ephemeral=True
            )
            return
        
        config.exempt_channel_ids.remove(channel_id)
        await self.config_manager.save_config(config, platform='discord')
        
        await interaction.response.send_message(
            f"✅ {channel.mention} is no longer exempt from auto-moderation",
            ephemeral=True
        )
    
    @app_commands.command(name="toggleautomod")
    @app_commands.default_permissions(administrator=True)
    async def toggle_automod(self, interaction: discord.Interaction):
        """Toggle automatic moderation on/off"""
        
        server_id = str(interaction.guild.id)
        config = await self.config_manager.get_config(server_id, interaction.guild.name, platform='discord')
        
        config.auto_moderate = not config.auto_moderate
        await self.config_manager.save_config(config, platform='discord')
        
        status = "enabled" if config.auto_moderate else "disabled"
        emoji = "✅" if config.auto_moderate else "⚠️"
        
        await interaction.response.send_message(
            f"{emoji} Automatic moderation is now **{status}**",
            ephemeral=True
        )
    
    @app_commands.command(name="resetconfig")
    @app_commands.default_permissions(administrator=True)
    async def reset_config(self, interaction: discord.Interaction):
        """Reset configuration to defaults"""
        
        # Confirmation button
        class ConfirmView(discord.ui.View):
            def __init__(self):
                super().__init__(timeout=30)
                self.value = None
            
            @discord.ui.button(label="Confirm Reset", style=discord.ButtonStyle.danger)
            async def confirm(
                self,
                button_interaction: discord.Interaction,
                button: discord.ui.Button
            ):
                self.value = True
                self.stop()
                await button_interaction.response.defer()
            
            @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
            async def cancel(
                self,
                button_interaction: discord.Interaction,
                button: discord.ui.Button
            ):
                self.value = False
                self.stop()
                await button_interaction.response.defer()
        
        view = ConfirmView()
        await interaction.response.send_message(
            "⚠️ This will reset ALL moderation settings to defaults. Are you sure?",
            view=view,
            ephemeral=True
        )
        
        await view.wait()
        
        if view.value:
            server_id = str(interaction.guild.id)
            
            # Create new default config
            config = ServerConfig(
                server_id=server_id,
                server_name=interaction.guild.name
            )
            
            await self.config_manager.save_config(config, platform='discord')
            
            await interaction.edit_original_response(
                content="✅ Configuration has been reset to defaults",
                view=None
            )
        else:
            await interaction.edit_original_response(
                content="❌ Reset cancelled",
                view=None
            )
            
    @app_commands.command(name="config")
    @app_commands.default_permissions(administrator=True)
    async def config_menu(self, interaction: discord.Interaction):        
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message(
                "❌ You need Administrator permission to use this command.",
                ephemeral=True
            )
            return
        
        await self._show_config_menu(interaction, interaction.guild.id)

    async def _show_config_menu(
        self, 
        interaction: discord.Interaction, 
        guild_id: int,
        edit: bool = False
    ):
        """Display the main configuration menu"""
        
        server_id = str(guild_id)
        config = await self.config_manager.get_config(
            server_id,
            interaction.guild.name,
            platform='discord'
        )
        
        # Create the main embed
        embed = discord.Embed(
            title="🛡️ Moderation Configuration Panel",
            description=(
                f"**Server:** {interaction.guild.name}\n"
                f"**Status:** {'✅ Active' if config.auto_moderate else '⚠️ Disabled'}\n\n"
                "Click a button below to configure that section."
            ),
            color=discord.Color.blue() if config.auto_moderate else discord.Color.orange()
        )
        
        # Quick overview
        embed.add_field(
            name="⚡ Strike System",
            value=f"Timeout: `{config.strikes_before_timeout}` • Kick: `{config.strikes_before_kick}` • Ban: `{config.strikes_before_ban}`",
            inline=False
        )
        
        embed.add_field(
            name="⏱️ Timeout Durations",
            value=f"Low: `{config.timeout_duration_low}m` • Med: `{config.timeout_duration_medium}m` • High: `{config.timeout_duration_high}m`",
            inline=False
        )
        
        embed.add_field(
            name="🎯 Detection Thresholds",
            value=f"Low: `{config.threshold_low}` • Medium: `{config.threshold_medium}` • High: `{config.threshold_high}`",
            inline=False
        )
        
        embed.add_field(
            name="⚙️ Actions",
            value=f"Low: `{config.low_severity_action}` • Medium: `{config.medium_severity_action}` • High: `{config.high_severity_action}`",
            inline=False
        )
        
        embed.set_footer(text="💡 Changes are saved instantly when you submit")
        
        # Create the button view
        view = ConfigMenuView(self.config_manager, config, self)
        
        if edit:
            await interaction.response.edit_message(embed=embed, view=view)
        else:
            await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
    
    
    @app_commands.command(name="quickset")
    @app_commands.describe(preset="Choose a preset configuration")
    @app_commands.choices(preset=[
        app_commands.Choice(name="😊 Lenient - Casual/Friendly servers", value="lenient"),
        app_commands.Choice(name="⚖️ Balanced - Recommended for most", value="balanced"),
        app_commands.Choice(name="⚔️ Strict - Large/Public servers", value="strict"),
    ])
    @app_commands.default_permissions(administrator=True)
    async def quick_preset(self, interaction: discord.Interaction, preset: str):
        """⚡ Instantly apply a preset configuration"""
        
        server_id = str(interaction.guild.id)
        config = await self.config_manager.get_config(
            server_id,
            interaction.guild.name,
            platform='discord'
        )
        
        if preset == "lenient":
            config.strikes_before_timeout = 5
            config.strikes_before_kick = 8
            config.strikes_before_ban = 10
            config.threshold_low = 0.4
            config.threshold_medium = 0.7
            config.threshold_high = 0.85
            config.timeout_duration_low = 5
            config.timeout_duration_medium = 30
            config.timeout_duration_high = 60
            desc = "**Lenient Mode:** More forgiving, higher thresholds"
            color = discord.Color.green()
        
        elif preset == "balanced":
            config.strikes_before_timeout = 3
            config.strikes_before_kick = 5
            config.strikes_before_ban = 7
            config.threshold_low = 0.3
            config.threshold_medium = 0.6
            config.threshold_high = 0.8
            config.timeout_duration_low = 10
            config.timeout_duration_medium = 60
            config.timeout_duration_high = 1440
            desc = "**Balanced Mode:** Recommended defaults"
            color = discord.Color.blue()
        
        else:  # strict
            config.strikes_before_timeout = 2
            config.strikes_before_kick = 3
            config.strikes_before_ban = 5
            config.threshold_low = 0.25
            config.threshold_medium = 0.5
            config.threshold_high = 0.7
            config.timeout_duration_low = 30
            config.timeout_duration_medium = 360
            config.timeout_duration_high = 2880
            desc = "**Strict Mode:** Zero tolerance, quick escalation"
            color = discord.Color.red()
        
        await self.config_manager.save_config(config, platform='discord')
        
        embed = discord.Embed(
            title=f"✅ {preset.title()} Preset Applied",
            description=desc,
            color=color
        )
        
        embed.add_field(
            name="Strike Thresholds",
            value=f"Timeout: {config.strikes_before_timeout} • Kick: {config.strikes_before_kick} • Ban: {config.strikes_before_ban}"
        )
        
        embed.add_field(
            name="Detection Sensitivity",
            value=f"Medium: {config.threshold_medium} • High: {config.threshold_high}"
        )
        
        embed.set_footer(text="Use /config to fine-tune individual settings")
        
        await interaction.response.send_message(embed=embed, ephemeral=True)
    
    @app_commands.command(name="toggle")
    @app_commands.describe(feature="Feature to toggle on/off")
    @app_commands.choices(feature=[
        app_commands.Choice(name="🤖 Auto-Moderation", value="auto_moderate"),
        app_commands.Choice(name="📧 DM Warnings", value="send_dm_warnings"),
        app_commands.Choice(name="👁️ Require Human Review", value="require_human_review"),
    ])
    @app_commands.default_permissions(administrator=True)
    async def toggle_feature(self, interaction: discord.Interaction, feature: str):
        """🔄 Quickly toggle features on/off"""
        
        server_id = str(interaction.guild.id)
        config = await self.config_manager.get_config(
            server_id,
            interaction.guild.name,
            platform='discord'
        )
        
        # Toggle the feature
        current = getattr(config, feature)
        setattr(config, feature, not current)
        await self.config_manager.save_config(config, platform='discord')
        
        new_state = getattr(config, feature)
        emoji = "✅" if new_state else "❌"
        status = "enabled" if new_state else "disabled"
        
        feature_names = {
            "auto_moderate": "Auto-Moderation",
            "send_dm_warnings": "DM Warnings",
            "require_human_review": "Human Review Requirement"
        }
        
        await interaction.response.send_message(
            f"{emoji} **{feature_names[feature]}** is now **{status}**",
            ephemeral=True
        )
    @app_commands.command(name="pardon")
    @app_commands.describe(
        user="User to pardon",
        reason="Why you're pardoning them"
    )
    @app_commands.default_permissions(administrator=True)
    async def pardon_user(
        self,
        interaction: discord.Interaction,
        user: discord.Member,
        reason: str = "Admin discretion"
    ):
        """
        Pardon a user - resets their strikes but keeps violation history.
        
        Use this for:
        - False positives (bot was wrong)
        - Reformed users (second chance)
        - Appeals (user contested the decision)
        """
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message(
                "❌ You need Administrator permission.",
                ephemeral=True
            )
            return
        
        await interaction.response.defer(ephemeral=True)
        
        server_id = str(interaction.guild.id)
        user_id = str(user.id)
        admin_id = str(interaction.user.id)
        
        # Get history before pardoning (so we can show it in embed)
        history = await self.bot.db.get_user_violation_history(
            server_id=server_id,
            user_id=user_id,
            platform='discord'
        )
        
        if history['active_strikes'] == 0:
            await interaction.followup.send(
                f"ℹ️ {user.mention} has **no active strikes** to pardon.\n"
                f"Lifetime violations on record: **{history['total_lifetime_violations']}**",
                ephemeral=True
            )
            return
        
        # Pardon the user
        result = await self.bot.db.pardon_user_violations(
            server_id=server_id,
            user_id=user_id,
            admin_id=admin_id,
            reason=reason,
            platform='discord'
        )
        
        if not result['success']:
            await interaction.followup.send(
                f"❌ Pardon failed: {result.get('reason', 'unknown error')}",
                ephemeral=True
            )
            return
        
        # Remove timeout if active
        timeout_removed = False
        if user.is_timed_out():
            try:
                await user.timeout(
                    None,
                    reason=f"Pardoned by {interaction.user.name}: {reason}"
                )
                timeout_removed = True
            except discord.Forbidden:
                pass  # Don't fail entire operation over permissions
        
        # Build embed
        embed = discord.Embed(
            title="✅ User Pardoned",
            description=f"{user.mention} has been given a fresh start.",
            color=discord.Color.green()
        )
        
        embed.add_field(
            name="Strikes",
            value=f"~~{result['previous_count']}~~ → **0**",
            inline=True
        )
        
        embed.add_field(
            name="Timeout",
            value="✅ Removed" if timeout_removed else "None active",
            inline=True
        )
        
        embed.add_field(
            name="Reason",
            value=reason,
            inline=False
        )
        
        embed.add_field(
            name="⚠️ Note",
            value=(
                f"Lifetime violations on record: **{history['total_lifetime_violations']}**\n"
                "History is kept for audit and ML purposes.\n"
                "The model will still factor in past behavior."
            ),
            inline=False
        )
        
        embed.set_footer(text=f"Pardoned by {interaction.user.name}")
        
        await interaction.followup.send(embed=embed, ephemeral=True)
        
        # Log pardon as a moderation event
        await self.bot.db.log_event(
            user_id=user_id,
            server_id=server_id,
            platform='discord',
            message=f"Pardoned by {interaction.user.name}: {reason}",
            score=0.0,
            severity='SAFE',
            action='PARDON',
            metadata={
                'pardoned_by': admin_id,
                'reason': reason,
                'previous_strikes': result['previous_count']
            }
        )
        
        # DM the user
        try:
            user_embed = discord.Embed(
                title="✅ Your Strikes Have Been Cleared",
                description=(
                    f"An administrator in **{interaction.guild.name}** "
                    "has pardoned your violations.\n\n"
                    "Your strike count has been reset to 0."
                ),
                color=discord.Color.green()
            )
            
            if reason != "Admin discretion":
                user_embed.add_field(name="Admin Note", value=reason)
            
            user_embed.set_footer(text="Please continue to follow community guidelines.")
            
            await user.send(embed=user_embed)
        except discord.Forbidden:
            pass  # DMs disabled, not a failure


    @app_commands.command(name="strikes")
    @app_commands.describe(user="User to check")
    @app_commands.default_permissions(moderate_members=True)
    async def view_strikes(
        self,
        interaction: discord.Interaction,
        user: discord.Member
    ):
        """
        View a user's strike history.
        Shows active strikes (for punishment) and lifetime violations (for context).
        """
        if not interaction.user.guild_permissions.moderate_members:
            await interaction.response.send_message(
                "❌ You need Moderate Members permission.",
                ephemeral=True
            )
            return
        
        await interaction.response.defer(ephemeral=True)
        
        history = await self.bot.db.get_user_violation_history(
            server_id=str(interaction.guild.id),
            user_id=str(user.id),
            platform='discord'
        )
        
        # Clean record
        if history['total_lifetime_violations'] == 0:
            embed = discord.Embed(
                title=f"📊 Strike History - {user.display_name}",
                description="✅ No violations on record.",
                color=discord.Color.green()
            )
            await interaction.followup.send(embed=embed, ephemeral=True)
            return
        
        # Determine color and status
        active = history['active_strikes']
        
        if history['is_pardoned'] and active == 0:
            color = discord.Color.blue()
            status = "🔵 Pardoned"
        elif active >= 5:
            color = discord.Color.red()
            status = "🔴 High Risk"
        elif active >= 3:
            color = discord.Color.orange()
            status = "🟠 At Risk"
        else:
            color = discord.Color.yellow()
            status = "🟡 Minor"
        
        embed = discord.Embed(
            title=f"📊 Strike History - {user.display_name}",
            description=f"**Status:** {status}",
            color=color
        )
        
        # Two key numbers side by side
        embed.add_field(
            name="Active Strikes",
            value=f"**{active}** / 7",
            inline=True
        )
        
        embed.add_field(
            name="Lifetime Violations",
            value=f"**{history['total_lifetime_violations']}**",
            inline=True
        )
        
        # Pardon details
        if history['is_pardoned']:
            pardoned_by = interaction.guild.get_member(
                int(history['pardoned_by'])
            ) if history['pardoned_by'] else None
            
            embed.add_field(
                name="✅ Pardon Info",
                value=(
                    f"By: {pardoned_by.mention if pardoned_by else 'Unknown'}\n"
                    f"Reason: {history['pardon_reason'] or 'No reason given'}\n"
                    f"When: <t:{int(history['pardoned_at'].timestamp())}:R>"
                ),
                inline=False
            )
        
        # Last 5 violations
        if history['recent_violations']:
            lines = []
            for v in history['recent_violations']:
                ts = int(v['timestamp'].timestamp())
                preview = v['message'][:40] + '...' if len(v['message']) > 40 else v['message']
                lines.append(f"• <t:{ts}:d> `{v['severity']}` - {preview}")
            
            embed.add_field(
                name="Recent Violations",
                value="\n".join(lines),
                inline=False
            )
        
        embed.set_thumbnail(url=user.display_avatar.url)
        embed.set_footer(text=f"Note: Lifetime violations are used by ML model regardless of pardons")
        
        await interaction.followup.send(embed=embed, ephemeral=True)



# ══════════════════════════════════════════════════════════════
# PERSISTENT BUTTON VIEW (DOESN'T TIMEOUT)
# ══════════════════════════════════════════════════════════════

class ConfigMenuView(discord.ui.View):
    def __init__(self, config_manager: ConfigManager, config: ServerConfig, cog):
        super().__init__(timeout=None)  # Never timeout
        self.config_manager = config_manager
        self.config = config
        self.cog = cog
    
    @discord.ui.button(
        label="Strike System",
        style=discord.ButtonStyle.primary,
        emoji="⚡",
        custom_id="config:strikes"
    )
    async def strikes_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(StrikeModal(self.config_manager, self.config, self.cog))
    
    @discord.ui.button(
        label="Timeouts",
        style=discord.ButtonStyle.primary,
        emoji="⏱️",
        custom_id="config:timeouts"
    )
    async def timeouts_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(TimeoutModal(self.config_manager, self.config, self.cog))
    
    @discord.ui.button(
        label="Thresholds",
        style=discord.ButtonStyle.primary,
        emoji="🎯",
        custom_id="config:thresholds"
    )
    async def thresholds_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(ThresholdModal(self.config_manager, self.config, self.cog))
    
    @discord.ui.button(
        label="Actions",
        style=discord.ButtonStyle.primary,
        emoji="⚙️",
        custom_id="config:actions"
    )
    # async def actions_button(self, interaction: discord.Interaction, button: discord.ui.Button):
    #     await interaction.response.send_modal(ActionModal(self.config_manager, self.config, self.cog))
    
    async def actions_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        view = ActionConfigView(self.config_manager, self.config, self)
        await interaction.response.send_message("⚙️ **Configure Actions**", view=view, ephemeral=True)

    @discord.ui.button(
        label="Features",
        style=discord.ButtonStyle.primary,
        emoji="🔧",
        custom_id="config:features"
    )
    
    # async def features_button(self, interaction: discord.Interaction, button: discord.ui.Button):
    #     await interaction.response.send_modal(FeatureModal(self.config_manager, self.config, self.cog))
    
    async def features_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        view = FeatureConfigView(self.config_manager, self.config, self)
        await interaction.response.send_message("🔧 **Toggle Features**", view=view, ephemeral=True)
    
    @discord.ui.button(
        label="Channels",
        style=discord.ButtonStyle.secondary,
        emoji="📢",
        custom_id="config:channels",
        row=1
    )
    async def channels_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(ChannelModal(self.config_manager, self.config, self.cog))
    
    @discord.ui.button(
        label="Refresh",
        style=discord.ButtonStyle.secondary,
        emoji="🔄",
        custom_id="config:refresh",
        row=1
    )
    async def refresh_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Reload config and refresh the menu
        server_id = str(interaction.guild.id)
        self.config = await self.config_manager.get_config(
            server_id,
            interaction.guild.name,
            platform='discord'
        )
        await self.cog._show_config_menu(interaction, interaction.guild.id, edit=True)
    
    @discord.ui.button(
        label="Reset to Defaults",
        style=discord.ButtonStyle.danger,
        emoji="⚠️",
        custom_id="config:reset",
        row=1
    )
    async def reset_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Confirmation view
        class ConfirmResetView(discord.ui.View):
            def __init__(self):
                super().__init__(timeout=30)
                self.value = None
            
            @discord.ui.button(label="Yes, Reset", style=discord.ButtonStyle.danger)
            async def confirm(self, button_interaction: discord.Interaction, button: discord.ui.Button):
                self.value = True
                self.stop()
            
            @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
            async def cancel(self, button_interaction: discord.Interaction, button: discord.ui.Button):
                self.value = False
                self.stop()
        
        confirm_view = ConfirmResetView()
        await interaction.response.send_message(
            "⚠️ **Are you sure?** This will reset ALL settings to defaults.",
            view=confirm_view,
            ephemeral=True
        )
        
        await confirm_view.wait()
        
        if confirm_view.value:
            # Reset config
            server_id = str(interaction.guild.id)
            default_config = ServerConfig(
                server_id=server_id,
                server_name=interaction.guild.name
            )
            await self.config_manager.save_config(default_config, platform='discord')
            
            await interaction.edit_original_response(
                content="✅ Configuration reset to defaults",
                view=None
            )
            
            # Refresh the main menu
            self.config = default_config
            await self.cog._show_config_menu(interaction, interaction.guild.id, edit=False)
        else:
            await interaction.edit_original_response(
                content="❌ Reset cancelled",
                view=None
            )


# ══════════════════════════════════════════════════════════════
# MODALS FOR EACH CATEGORY
# ══════════════════════════════════════════════════════════════

class StrikeModal(discord.ui.Modal, title="⚡ Strike System Configuration"):
    strikes_timeout = discord.ui.TextInput(
        label="Strikes before timeout",
        placeholder="Default: 3",
        max_length=2,
        required=True
    )
    
    strikes_kick = discord.ui.TextInput(
        label="Strikes before kick",
        placeholder="Default: 5",
        max_length=2,
        required=True
    )
    
    strikes_ban = discord.ui.TextInput(
        label="Strikes before ban",
        placeholder="Default: 7",
        max_length=2,
        required=True
    )
    
    decay_days = discord.ui.TextInput(
        label="Strike decay period (days)",
        placeholder="Default: 30 (strikes expire after X days)",
        max_length=3,
        required=True
    )
    
    def __init__(self, config_manager: ConfigManager, config: ServerConfig, cog):
        super().__init__()
        self.config_manager = config_manager
        self.config = config
        self.cog = cog
        
        # Pre-fill with current values
        self.strikes_timeout.default = str(config.strikes_before_timeout)
        self.strikes_kick.default = str(config.strikes_before_kick)
        self.strikes_ban.default = str(config.strikes_before_ban)
        self.decay_days.default = str(config.strike_decay_days)
    
    async def on_submit(self, interaction: discord.Interaction):
        try:
            # Update config
            self.config.strikes_before_timeout = int(self.strikes_timeout.value)
            self.config.strikes_before_kick = int(self.strikes_kick.value)
            self.config.strikes_before_ban = int(self.strikes_ban.value)
            self.config.strike_decay_days = int(self.decay_days.value)
            
            # Validate
            if not (self.config.strikes_before_timeout < self.config.strikes_before_kick < self.config.strikes_before_ban):
                await interaction.response.send_message(
                    "❌ Invalid: Strikes must increase (timeout < kick < ban)",
                    ephemeral=True
                )
                return
            
            # Save
            await self.config_manager.save_config(self.config, platform='discord')
            
            # Confirm
            await interaction.response.send_message(
                "✅ Strike system updated successfully!",
                ephemeral=True
            )
            
        except ValueError as e:
            await interaction.response.send_message(
                f"❌ Invalid input: Please enter numbers only.\n{e}",
                ephemeral=True
            )


class TimeoutModal(discord.ui.Modal, title="⏱️ Timeout Duration Configuration"):
    timeout_low = discord.ui.TextInput(
        label="Low severity timeout (minutes)",
        placeholder="Default: 10",
        max_length=5,
        required=True
    )
    
    timeout_medium = discord.ui.TextInput(
        label="Medium severity timeout (minutes)",
        placeholder="Default: 60",
        max_length=5,
        required=True
    )
    
    timeout_high = discord.ui.TextInput(
        label="High severity timeout (minutes)",
        placeholder="Default: 1440 (24 hours) • Max: 40320 (28 days)",
        max_length=5,
        required=True
    )
    
    def __init__(self, config_manager: ConfigManager, config: ServerConfig, cog):
        super().__init__()
        self.config_manager = config_manager
        self.config = config
        self.cog = cog
        
        self.timeout_low.default = str(config.timeout_duration_low)
        self.timeout_medium.default = str(config.timeout_duration_medium)
        self.timeout_high.default = str(config.timeout_duration_high)
    
    async def on_submit(self, interaction: discord.Interaction):
        try:
            self.config.timeout_duration_low = int(self.timeout_low.value)
            self.config.timeout_duration_medium = int(self.timeout_medium.value)
            self.config.timeout_duration_high = int(self.timeout_high.value)
            
            # Discord max timeout is 28 days (40320 minutes)
            if self.config.timeout_duration_high > 40320:
                await interaction.response.send_message(
                    "❌ Maximum timeout is 40320 minutes (28 days)",
                    ephemeral=True
                )
                return
            
            await self.config_manager.save_config(self.config, platform='discord')
            
            await interaction.response.send_message(
                "✅ Timeout durations updated successfully!",
                ephemeral=True
            )
            
        except ValueError as e:
            await interaction.response.send_message(
                f"❌ Invalid input: {e}",
                ephemeral=True
            )


class ThresholdModal(discord.ui.Modal, title="🎯 Detection Threshold Configuration"):
    threshold_low = discord.ui.TextInput(
        label="Low threshold (0.0 - 1.0)",
        placeholder="Default: 0.3 (Below this = safe)",
        max_length=4,
        required=True
    )
    
    threshold_medium = discord.ui.TextInput(
        label="Medium threshold (0.0 - 1.0)",
        placeholder="Default: 0.6 (Above this = medium severity)",
        max_length=4,
        required=True
    )
    
    threshold_high = discord.ui.TextInput(
        label="High threshold (0.0 - 1.0)",
        placeholder="Default: 0.8 (Above this = high severity)",
        max_length=4,
        required=True
    )
    
    def __init__(self, config_manager: ConfigManager, config: ServerConfig, cog):
        super().__init__()
        self.config_manager = config_manager
        self.config = config
        self.cog = cog
        
        self.threshold_low.default = str(config.threshold_low)
        self.threshold_medium.default = str(config.threshold_medium)
        self.threshold_high.default = str(config.threshold_high)
    
    async def on_submit(self, interaction: discord.Interaction):
        try:
            low = float(self.threshold_low.value)
            med = float(self.threshold_medium.value)
            high = float(self.threshold_high.value)
            
            # Validation
            if not all(0 <= x <= 1 for x in [low, med, high]):
                await interaction.response.send_message(
                    "❌ All thresholds must be between 0.0 and 1.0",
                    ephemeral=True
                )
                return
            
            if not (low < med < high):
                await interaction.response.send_message(
                    "❌ Thresholds must increase (low < medium < high)",
                    ephemeral=True
                )
                return
            
            self.config.threshold_low = low
            self.config.threshold_medium = med
            self.config.threshold_high = high
            
            await self.config_manager.save_config(self.config, platform='discord')
            
            await interaction.response.send_message(
                "✅ Detection thresholds updated successfully!",
                ephemeral=True
            )
            
        except ValueError as e:
            await interaction.response.send_message(
                f"❌ Invalid input: {e}",
                ephemeral=True
            )

class ActionConfigView(discord.ui.View):
    def __init__(self, config_manager, config, parent_view):
        super().__init__(timeout=180)
        self.config_manager = config_manager
        self.config = config
        self.parent_view = parent_view
        
        # Options available for every severity level
        options = [
            discord.SelectOption(label="⚠️ Warn", value="warn", emoji="⚠️"),
            discord.SelectOption(label="🗑️ Delete", value="delete", emoji="🗑️"),
            discord.SelectOption(label="⏱️ Timeout", value="timeout", emoji="⏱️"),
            discord.SelectOption(label="👢 Kick", value="kick", emoji="👢"),
            discord.SelectOption(label="🔨 Ban", value="ban", emoji="🔨"),
        ]

        # LOW SEVERITY DROPDOWN
        self.low_select = discord.ui.Select(
            placeholder=f"Low Severity Action (Current: {config.low_severity_action})",
            options=options,
            row=0
        )
        self.low_select.callback = self.on_low_change
        self.add_item(self.low_select)

        # MEDIUM SEVERITY DROPDOWN
        self.med_select = discord.ui.Select(
            placeholder=f"Medium Severity Action (Current: {config.medium_severity_action})",
            options=options,
            row=1
        )
        self.med_select.callback = self.on_med_change
        self.add_item(self.med_select)

        # HIGH SEVERITY DROPDOWN
        self.high_select = discord.ui.Select(
            placeholder=f"High Severity Action (Current: {config.high_severity_action})",
            options=options,
            row=2
        )
        self.high_select.callback = self.on_high_change
        self.add_item(self.high_select)

    async def on_low_change(self, interaction: discord.Interaction):
        self.config.low_severity_action = self.low_select.values[0]
        await self.save_and_update(interaction, "Low severity")

    async def on_med_change(self, interaction: discord.Interaction):
        self.config.medium_severity_action = self.med_select.values[0]
        await self.save_and_update(interaction, "Medium severity")

    async def on_high_change(self, interaction: discord.Interaction):
        self.config.high_severity_action = self.high_select.values[0]
        await self.save_and_update(interaction, "High severity")

    async def save_and_update(self, interaction: discord.Interaction, field_name: str):
        await self.config_manager.save_config(self.config, platform='discord')
        # Update placeholders to show new values
        self.low_select.placeholder = f"Low Severity Action (Current: {self.config.low_severity_action})"
        self.med_select.placeholder = f"Medium Severity Action (Current: {self.config.medium_severity_action})"
        self.high_select.placeholder = f"High Severity Action (Current: {self.config.high_severity_action})"
        await interaction.response.edit_message(view=self)

class FeatureConfigView(discord.ui.View):
    def __init__(self, config_manager, config, parent_view):
        super().__init__(timeout=180)
        self.config_manager = config_manager
        self.config = config
        self.parent_view = parent_view

        # Common True/False options
        bool_options = [
            discord.SelectOption(label="✅ Enabled", value="true"),
            discord.SelectOption(label="❌ Disabled", value="false")
        ]

        # AUTO MODERATE
        self.automod = discord.ui.Select(
            placeholder=f"Auto-Moderate (Current: {'✅' if config.auto_moderate else '❌'})",
            options=bool_options,
            row=0
        )
        self.automod.callback = self.on_automod_change
        self.add_item(self.automod)

        # DM WARNINGS
        self.dm_warn = discord.ui.Select(
            placeholder=f"DM Warnings (Current: {'✅' if config.send_dm_warnings else '❌'})",
            options=bool_options,
            row=1
        )
        self.dm_warn.callback = self.on_dm_change
        self.add_item(self.dm_warn)

        # REQUIRE REVIEW
        self.review = discord.ui.Select(
            placeholder=f"Require Human Review (Current: {'✅' if config.require_human_review else '❌'})",
            options=bool_options,
            row=2
        )
        self.review.callback = self.on_review_change
        self.add_item(self.review)

    async def on_automod_change(self, interaction: discord.Interaction):
        self.config.auto_moderate = (self.automod.values[0] == "true")
        await self.save_and_update(interaction)

    async def on_dm_change(self, interaction: discord.Interaction):
        self.config.send_dm_warnings = (self.dm_warn.values[0] == "true")
        await self.save_and_update(interaction)

    async def on_review_change(self, interaction: discord.Interaction):
        self.config.require_human_review = (self.review.values[0] == "true")
        await self.save_and_update(interaction)

    async def save_and_update(self, interaction: discord.Interaction):
        await self.config_manager.save_config(self.config, platform='discord')
        
        # Update placeholders visually
        self.automod.placeholder = f"Auto-Moderate (Current: {'✅' if self.config.auto_moderate else '❌'})"
        self.dm_warn.placeholder = f"DM Warnings (Current: {'✅' if self.config.send_dm_warnings else '❌'})"
        self.review.placeholder = f"Require Human Review (Current: {'✅' if self.config.require_human_review else '❌'})"
        
        await interaction.response.edit_message(view=self)

class ChannelModal(discord.ui.Modal, title="📢 Channel Configuration"):
    log_channel = discord.ui.TextInput(
        label="Log channel ID (optional)",
        placeholder="Right-click channel → Copy ID",
        max_length=20,
        required=False
    )
    
    alert_channel = discord.ui.TextInput(
        label="Alert channel ID (optional)",
        placeholder="For high-severity alerts",
        max_length=20,
        required=False
    )
    
    def __init__(self, config_manager: ConfigManager, config: ServerConfig, cog):
        super().__init__()
        self.config_manager = config_manager
        self.config = config
        self.cog = cog
        
        if config.log_channel_id:
            self.log_channel.default = config.log_channel_id
        if config.alert_channel_id:
            self.alert_channel.default = config.alert_channel_id
    
    async def on_submit(self, interaction: discord.Interaction):
        if self.log_channel.value:
            self.config.log_channel_id = self.log_channel.value.strip()
        
        if self.alert_channel.value:
            self.config.alert_channel_id = self.alert_channel.value.strip()
        
        await self.config_manager.save_config(self.config, platform='discord')
        
        await interaction.response.send_message(
            "✅ Channels updated successfully!",
            ephemeral=True
        )

    
async def setup(bot):
    await bot.add_cog(AdminCommands(bot))