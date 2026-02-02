import discord
from discord.ext import commands
import logging
from dotenv import load_dotenv
import os

load_dotenv()

token = os.get_env('DISCORD_BOT_TOKEN')
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
intents = discord.Intents.default()

intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print("It is time")

# bot.run(token, log_handler=handler, log_level=logging.DEBUG)

@bot.event
async def on_member_join(member):
    await member.send(f"Welcome to the server {member.name}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return 
    
    if 'shit' in message.content.lower():
        await message.delete()
        await message.channel.send(f"{message.author.metion}-you'll have to leave this channel")
    
    await bot.process_commands(message)