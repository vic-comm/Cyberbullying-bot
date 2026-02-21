# ═══════════════════════════════════════════════════════════════
# bot_service/cogs/admin_review.py
# Admin feedback review system with bulk operations
# ═══════════════════════════════════════════════════════════════

import discord
from discord import app_commands
from discord.ext import commands
from typing import Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# REVIEW ITEM EMBED (displays one feedback item)
# ═══════════════════════════════════════════════════════════════
def create_review_item_embed(item: dict, index: int = 0) -> discord.Embed:
    """Create embed for a single review item"""
    
    # 1. Determine Color & Title based on Context
    score = item.get('toxicity_score', 0)
    
    if item.get('type') == 'uncertain':
        title = f"⚠️ Uncertain Prediction #{index + 1}"
        color = discord.Color.yellow()
        confidence = "Uncertain"
    else:
        title = f"📋 Dispute Review #{index + 1}"
        # For disputes, use red if high score, else orange
        color = discord.Color.red() if score > 0.7 else discord.Color.orange()
        confidence = "High" if score > 0.7 else "Medium"

    # 2. Initialize Embed (Using the variables we just set)
    embed = discord.Embed(
        title=title,
        color=color
    )
    
    # 3. Message Text
    text = item['text'][:500]
    if len(item['text']) > 500:
        text += "..."
    embed.add_field(
        name="Message",
        value=f"```{text}```",
        inline=False
    )
    
    # 4. Model Decision
    model_label = "🔴 Toxic" if item['predicted_label'] == 1 else "✅ Safe"
    embed.add_field(
        name="Model Says",
        value=f"{model_label} ({score:.1%})",
        inline=True
    )
    
    # 5. User Claim (Handle NULL for uncertain items)
    user_claim = item.get('user_claimed_label')
    if user_claim is not None:
        user_label = "🔴 Toxic" if user_claim == 1 else "✅ Safe"
        embed.add_field(name="User Claims", value=user_label, inline=True)
    else:
        # Uncertain items don't have a user claim yet
        embed.add_field(name="Status", value="❓ Needs Review", inline=True)
    
    # 6. Timing Info
    # Handle both 'hours_pending' (DB view) and raw calculation
    hours_val = item.get('hours_pending') or 0
    if hours_val < 1:
        time_str = f"{int(hours_val * 60)} min ago"
    elif hours_val < 24:
        time_str = f"{int(hours_val)} hours ago"
    else:
        time_str = f"{int(hours_val / 24)} days ago"
    
    embed.add_field(name="Age", value=time_str, inline=True)
    
    # 7. User History (Optional)
    violations = item.get('user_total_violations', 0)
    if violations > 0:
        embed.add_field(name="History", value=f"{violations} violations", inline=True)
    
    # Footer
    feedback_id = item.get('feedback_id', 'Pending')
    embed.set_footer(
        text=f"Feedback ID: {feedback_id} | Log ID: {item['log_id']} | User: {item['user_id']}"
    )
    
    return embed

# REVIEW CONTROLS (buttons for single item)

class ReviewItemView(discord.ui.View):
    """Interactive buttons for reviewing a single item"""
    
    def __init__(self, item: dict, admin_id: str, db, bot, server_id: str):
        super().__init__(timeout=600) # 10 minute timeout
        self.item = item
        self.admin_id = admin_id
        self.db = db
        self.bot = bot
        self.decision_made = False
        self.server_id = str(server_id)
    
    async def load_next_item(self, interaction: discord.Interaction):
        """Fetches the next item and updates the message."""
        # 1. Fetch the next item from the queue
        # We use offset=0 because the previous item is now 'reviewed' and gone from the queue
        queue = await self.db.get_review_queue(
            server_id=self.server_id,
            limit=1
        )

        if not queue:
            # Queue is empty!
            embed = discord.Embed(
                title="🎉 All Caught Up!",
                description="There are no more items to review.",
                color=discord.Color.green()
            )
            for child in self.children:
                child.disabled = True
            await interaction.edit_original_response(embed=embed, view=self)
            self.stop()
            return

        # 2. Update internal state with new item
        self.item = queue[0]
        
        # 3. Generate new Embed
        # Note: We pass index=0 because it's always the "next" item
        new_embed = create_review_item_embed(self.item, index=0)
        
        # 4. Re-enable buttons (in case they were disabled)
        for child in self.children:
            child.disabled = False
            
        # 5. Update the message
        await interaction.edit_original_response(embed=new_embed, view=self)

    async def handle_decision(self, interaction: discord.Interaction, decision: str):
        """Standard logic for handling Safe/Toxic/Skip"""
        if str(interaction.user.id) != self.admin_id:
            await interaction.response.send_message("❌ Only the reviewing admin can use these buttons", ephemeral=True)
            return

        await interaction.response.defer()

        # Logic for creating feedback if it's an "Uncertain" item
        if self.item.get('type') == 'uncertain':
            feedback_id = await self.db.record_user_dispute(
                log_id=self.item['log_id'],
                user_id=self.item['user_id'],
                server_id=self.server_id,
                platform='discord',
                user_claimed_label=0, 
                dispute_reason="Admin Proactive Review"
            )
            self.item['feedback_id'] = feedback_id

        # Logic for Skip (Don't save to DB, just move on)
        if decision == 'skip':
            await self.load_next_item(interaction)
            return

        # Save decision to DB
        await self.db.admin_review_feedback(
            feedback_id=self.item['feedback_id'],
            admin_id=self.admin_id,
            decision=decision,
            notes="Reviewed via continuous dashboard"
        )

        # Load the next one!
        await self.load_next_item(interaction)

    @discord.ui.button(label="✅ Model Correct", style=discord.ButtonStyle.success)
    async def agree_with_model(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_decision(interaction, 'agree_with_model')

    @discord.ui.button(label="❌ User Correct", style=discord.ButtonStyle.danger)
    async def agree_with_user(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_decision(interaction, 'agree_with_user')

    @discord.ui.button(label="⏭️ Skip", style=discord.ButtonStyle.secondary)
    async def skip_item(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_decision(interaction, 'skip')
    
    @discord.ui.button(label="🛑 Stop", style=discord.ButtonStyle.danger)
    async def stop_session(self, interaction: discord.Interaction, button: discord.ui.Button):
        if str(interaction.user.id) != self.admin_id: return
        await interaction.response.edit_message(content="🛑 Review session ended.", view=None, embed=None)
        self.stop()
    
    @discord.ui.button(label="⏭️ Skip", style=discord.ButtonStyle.secondary, custom_id="skip")
    async def skip_item(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Skip this item for now"""
        if str(interaction.user.id) != self.admin_id:
            await interaction.response.send_message("❌ Only the reviewing admin can use these buttons", ephemeral=True)
            return
        
        await interaction.response.send_message("⏭️ Skipped", ephemeral=True)
        self.stop()
    
    @discord.ui.button(label="📝 Add Note", style=discord.ButtonStyle.secondary, custom_id="add_note")
    async def add_note(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Add a note to this review"""
        if str(interaction.user.id) != self.admin_id:
            await interaction.response.send_message("❌ Only the reviewing admin can use these buttons", ephemeral=True)
            return
        
        # Open modal for note
        modal = ReviewNoteModal(self.item, self.admin_id, self.db)
        await interaction.response.send_modal(modal)


class ReviewNoteModal(discord.ui.Modal, title="Add Review Note"):
    """Modal for adding notes to a review"""
    
    note = discord.ui.TextInput(
        label="Admin Note",
        style=discord.TextStyle.paragraph,
        placeholder="Why did you make this decision?",
        required=True,
        max_length=500
    )
    
    decision = discord.ui.TextInput(
        label="Decision (model/user/custom)",
        style=discord.TextStyle.short,
        placeholder="model",
        required=True,
        max_length=10
    )
    
    def __init__(self, item: dict, admin_id: str, db):
        super().__init__()
        self.item = item
        self.admin_id = admin_id
        self.db = db
    
    async def on_submit(self, interaction: discord.Interaction):
        decision = self.decision.value.lower()
        
        if decision not in ['model', 'user', 'custom']:
            await interaction.response.send_message(
                "❌ Decision must be 'model', 'user', or 'custom'",
                ephemeral=True
            )
            return
        
        await self.db.admin_review_feedback(
            feedback_id=self.item['feedback_id'],
            admin_id=self.admin_id,
            decision=f'agree_with_{decision}',
            notes=self.note.value
        )
        
        await interaction.response.send_message(
            f"✅ Review recorded with note: {self.note.value[:50]}...",
            ephemeral=True
        )


# ═══════════════════════════════════════════════════════════════
# BULK REVIEW VIEW (for grouped items)
# ═══════════════════════════════════════════════════════════════

class BulkReviewView(discord.ui.View):
    """Bulk actions for multiple items"""
    
    def __init__(self, items: List[dict], admin_id: str, db, group_label: str):
        super().__init__(timeout=600)  # 10 minute timeout
        self.items = items
        self.admin_id = admin_id
        self.db = db
        self.group_label = group_label
    
    @discord.ui.button(label="✅ Approve All (Model Correct)", style=discord.ButtonStyle.success)
    async def bulk_approve_model(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Approve all - model was correct on all items"""
        if str(interaction.user.id) != self.admin_id:
            await interaction.response.send_message("❌ Only the reviewing admin can use these buttons", ephemeral=True)
            return
        
        await interaction.response.defer()
        
        feedback_ids = [item['feedback_id'] for item in self.items]
        count = await self.db.bulk_approve_model(feedback_ids, self.admin_id)
        
        await interaction.followup.send(
            f"✅ Approved {count} items - model was correct on all",
            ephemeral=True
        )
        
        # Disable buttons
        for child in self.children:
            child.disabled = True
        await interaction.message.edit(view=self)
        self.stop()
    
    @discord.ui.button(label="❌ Approve All (Users Correct)", style=discord.ButtonStyle.danger)
    async def bulk_approve_users(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Approve all - users were correct on all items"""
        if str(interaction.user.id) != self.admin_id:
            await interaction.response.send_message("❌ Only the reviewing admin can use these buttons", ephemeral=True)
            return
        
        await interaction.response.defer()
        
        feedback_ids = [item['feedback_id'] for item in self.items]
        count = await self.db.bulk_approve_users(feedback_ids, self.admin_id)
        
        await interaction.followup.send(
            f"✅ Approved {count} items - users were correct (false positives)",
            ephemeral=True
        )
        
        # Disable buttons
        for child in self.children:
            child.disabled = True
        await interaction.message.edit(view=self)
        self.stop()
    
    @discord.ui.button(label="📋 Review Individually", style=discord.ButtonStyle.secondary)
    async def review_individually(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Review each item one by one"""
        await interaction.response.send_message(
            "Use `/review next` to review items one at a time",
            ephemeral=True
        )
        self.stop()


# ═══════════════════════════════════════════════════════════════
# ADMIN REVIEW COG
# ═══════════════════════════════════════════════════════════════

class AdminReview(commands.Cog):
    """Admin commands for reviewing feedback and uncertain messages"""
    
    def __init__(self, bot):
        self.bot = bot
    
    # ───────────────────────────────────────────────────────────
    # /review - Main review dashboard
    # ───────────────────────────────────────────────────────────
    
    @app_commands.command(name="review")
    @app_commands.describe(
        view="How to view the queue",
        filter_user="Filter by specific user (optional)"
    )
    @app_commands.choices(view=[
        app_commands.Choice(name="📋 Next Item", value="next"),
        app_commands.Choice(name="👤 Group by User", value="by_user"),
        app_commands.Choice(name="📊 Summary Stats", value="stats"),
        app_commands.Choice(name="📜 Full List", value="list")
    ])
    @app_commands.default_permissions(administrator=True)
    async def review_queue(
        self,
        interaction: discord.Interaction,
        view: str = "next",
        filter_user: Optional[discord.Member] = None
    ):
        """
        Review disputed messages and uncertain predictions.
        
        Use this to:
        - Approve or override model decisions
        - Review user disputes
        - Handle uncertain predictions (0.3-0.7 confidence)
        """
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message(
                "❌ You need Administrator permission",
                ephemeral=True
            )
            return
        
        await interaction.response.defer(ephemeral=True)
        
        server_id = str(interaction.guild.id)
        admin_id = str(interaction.user.id)
        
        # ═══════════════════════════════════════════════════════
        # VIEW: Next Item (one-by-one review)
        # ═══════════════════════════════════════════════════════
        
        if view == "next":
            # Get next pending item
            queue = await self.bot.db.get_review_queue(
                server_id=server_id,
                limit=1
            )
            
            if not queue:
                await interaction.followup.send(
                    "✅ No items pending review! Queue is empty.",
                    ephemeral=True
                )
                return
            
            item = queue[0]
            
            # Create embed and controls
            embed = create_review_item_embed(item, index=0)
            view_controls = ReviewItemView(item, admin_id, self.bot.db, self.bot, server_id=server_id)
            
            await interaction.followup.send(
                embed=embed,
                view=view_controls,
                ephemeral=True
            )
        
        # ═══════════════════════════════════════════════════════
        # VIEW: Group by User (bulk operations)
        # ═══════════════════════════════════════════════════════
        
        elif view == "by_user":
            grouped = await self.bot.db.get_review_queue_grouped(
                server_id=server_id,
                group_by='user'
            )
            
            if not grouped:
                await interaction.followup.send(
                    "✅ No items pending review!",
                    ephemeral=True
                )
                return
            
            # Show groups
            embed = discord.Embed(
                title="📊 Review Queue - Grouped by User",
                description="Users with pending disputes:",
                color=discord.Color.blue()
            )
            
            for user_id, items in list(grouped.items())[:10]:  # Show top 10
                user = interaction.guild.get_member(int(user_id))
                user_name = user.mention if user else f"User {user_id[:8]}..."
                
                embed.add_field(
                    name=f"{user_name}",
                    value=f"{len(items)} disputes",
                    inline=True
                )
            
            embed.set_footer(text="Use /review user:@user to review a specific user's disputes")
            
            await interaction.followup.send(embed=embed, ephemeral=True)
        
        # ═══════════════════════════════════════════════════════
        # VIEW: Stats Summary
        # ═══════════════════════════════════════════════════════
        
        elif view == "stats":
            stats = await self.bot.db.get_feedback_stats(server_id)
            
            embed = discord.Embed(
                title="📊 Feedback Review Statistics",
                description=f"Last 30 days in {interaction.guild.name}",
                color=discord.Color.gold()
            )
            
            total = stats.get('total_disputes', 0)
            reviewed = stats.get('reviewed', 0)
            pending = stats.get('pending', 0)
            
            embed.add_field(
                name="Total Disputes",
                value=f"{total}",
                inline=True
            )
            
            embed.add_field(
                name="Reviewed",
                value=f"{reviewed} ({reviewed/total*100:.0f}%)" if total > 0 else "0",
                inline=True
            )
            
            embed.add_field(
                name="⏳ Pending",
                value=f"**{pending}**",
                inline=True
            )
            
            if reviewed > 0:
                model_correct = stats.get('model_correct', 0)
                user_correct = stats.get('user_correct', 0)
                
                embed.add_field(
                    name="Model Correct",
                    value=f"{model_correct} ({model_correct/reviewed*100:.0f}%)",
                    inline=True
                )
                
                embed.add_field(
                    name="User Correct (FP)",
                    value=f"{user_correct} ({user_correct/reviewed*100:.0f}%)",
                    inline=True
                )
                
                avg_hours = stats.get('avg_review_time_hours', 0)
                if avg_hours:
                    embed.add_field(
                        name="Avg Review Time",
                        value=f"{avg_hours:.1f} hours",
                        inline=True
                    )
            
            # Top disputing users
            top_users = await self.bot.db.get_top_disputing_users(server_id, limit=5)
            if top_users:
                users_text = ""
                for u in top_users:
                    member = interaction.guild.get_member(int(u['user_id']))
                    name = member.mention if member else f"User {u['user_id'][:8]}"
                    users_text += f"{name}: {u['dispute_count']} disputes\n"
                
                embed.add_field(
                    name="🔝 Top Disputing Users",
                    value=users_text,
                    inline=False
                )
            
            await interaction.followup.send(embed=embed, ephemeral=True)
        
        # ═══════════════════════════════════════════════════════
        # VIEW: Full List
        # ═══════════════════════════════════════════════════════
        
        elif view == "list":
            queue = await self.bot.db.get_review_queue(
                server_id=server_id,
                limit=20
            )
            
            if not queue:
                await interaction.followup.send(
                    "✅ No items pending review!",
                    ephemeral=True
                )
                return
            
            embed = discord.Embed(
                title="📋 Review Queue",
                description=f"{len(queue)} items pending (showing first 20)",
                color=discord.Color.blue()
            )
            
            for i, item in enumerate(queue[:10]):  # Show first 10 in embed
                user = interaction.guild.get_member(int(item['user_id']))
                user_name = user.mention if user else f"User {item['user_id'][:8]}"
                
                text_preview = item['text'][:50]
                if len(item['text']) > 50:
                    text_preview += "..."
                
                model = "🔴" if item['predicted_label'] == 1 else "✅"
                user_claim = "🔴" if item['user_claimed_label'] == 1 else "✅"
                
                embed.add_field(
                    name=f"#{i+1} - {user_name}",
                    value=f"{text_preview}\nModel: {model} | User: {user_claim}",
                    inline=False
                )
            
            embed.set_footer(text="Use /review next to review one at a time")
            
            await interaction.followup.send(embed=embed, ephemeral=True)
    
    # ───────────────────────────────────────────────────────────
    # /review_user - Review all disputes from one user
    # ───────────────────────────────────────────────────────────
    
    @app_commands.command(name="review_user")
    @app_commands.describe(user="User whose disputes to review")
    @app_commands.default_permissions(administrator=True)
    async def review_user_disputes(
        self,
        interaction: discord.Interaction,
        user: discord.Member
    ):
        """Review all pending disputes from a specific user (bulk operation)"""
        
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message(
                "❌ You need Administrator permission",
                ephemeral=True
            )
            return
        
        await interaction.response.defer(ephemeral=True)
        
        server_id = str(interaction.guild.id)
        user_id = str(user.id)
        admin_id = str(interaction.user.id)
        
        # Get all disputes from this user
        queue = await self.bot.db.get_review_queue(
            server_id=server_id,
            filter_by='user',
            user_id=user_id
        )
        
        if not queue:
            await interaction.followup.send(
                f"✅ {user.mention} has no pending disputes",
                ephemeral=True
            )
            return
        
        # Create summary embed
        embed = discord.Embed(
            title=f"📋 Reviewing Disputes from {user.display_name}",
            description=f"{len(queue)} pending disputes",
            color=discord.Color.orange()
        )
        
        # Show preview of disputes
        for i, item in enumerate(queue[:5]):
            text_preview = item['text'][:50]
            if len(item['text']) > 50:
                text_preview += "..."
            
            model = "🔴 Toxic" if item['predicted_label'] == 1 else "✅ Safe"
            user_claim = "🔴 Toxic" if item['user_claimed_label'] == 1 else "✅ Safe"
            
            embed.add_field(
                name=f"Dispute #{i+1}",
                value=f"{text_preview}\nModel: {model} → User: {user_claim}",
                inline=False
            )
        
        if len(queue) > 5:
            embed.add_field(
                name="...",
                value=f"+ {len(queue) - 5} more disputes",
                inline=False
            )
        
        # Bulk action buttons
        bulk_view = BulkReviewView(
            items=queue,
            admin_id=admin_id,
            db=self.bot.db,
            group_label=f"{user.display_name}'s disputes"
        )
        
        await interaction.followup.send(
            embed=embed,
            view=bulk_view,
            ephemeral=True
        )


async def setup(bot):
    await bot.add_cog(AdminReview(bot))