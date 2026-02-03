import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import re
from collections import Counter


INPUT_FILE = "mlops/Cleaned_Final (8).csv"
OUTPUT_FILE = "data/training_data_with_history.parquet"
NUM_USERS = 5000
NUM_CHANNELS = 50
DATE_RANGE_DAYS = 365

BAD_WORDS = ["trash", "kill", "die", "ugly", "stupid", "idiot", "hate", "loser", "dumb"]
PERSONAL_PRONOUNS = ["you", "your", "you're", "ur", "u"]

# ==========================================
# 2. LOAD & SHUFFLE
# ==========================================
print(f"Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f"Dataset Shape: {df.shape}")
df = df.rename(columns={'Message': 'text', 'Label': 'label'})

# ==========================================
# 3. REALISTIC USER PERSONALITIES
# ==========================================
print("Synthesizing User Personas...")

user_ids = np.arange(1000, 1000 + NUM_USERS)

# Create diverse user archetypes
trolls = np.random.choice(user_ids, size=int(NUM_USERS * 0.08), replace=False)  # 8% trolls
aggressors = np.random.choice(
    np.setdiff1d(user_ids, trolls), 
    size=int(NUM_USERS * 0.12), 
    replace=False
)  # 12% occasionally aggressive
normals = np.setdiff1d(user_ids, np.concatenate([trolls, aggressors]))

def assign_user_realistic(label):
    """Assign users based on realistic behavior patterns"""
    if label == 1:
        rand = random.random()
        if rand < 0.60:  # 60% from trolls
            return np.random.choice(trolls)
        elif rand < 0.85:  # 25% from aggressors
            return np.random.choice(aggressors)
        else:  # 15% from normally safe users (everyone can have bad days)
            return np.random.choice(normals)
    else:
        rand = random.random()
        if rand < 0.85:  # 85% from normal users
            return np.random.choice(normals)
        elif rand < 0.95:  # 10% from aggressors (they're not always toxic)
            return np.random.choice(aggressors)
        else:  # 5% from trolls (occasional normal messages)
            return np.random.choice(trolls)

df['user_id'] = df['label'].apply(assign_user_realistic)

# ==========================================
# 4. REALISTIC TIMESTAMPS
# ==========================================
print("Synthesizing Timestamps with Activity Patterns...")

end_date = datetime.now()
start_date = end_date - timedelta(days=DATE_RANGE_DAYS)

# Weight towards evening hours (more activity 6PM-11PM)
def generate_realistic_timestamp():
    random_day = random.randint(0, DATE_RANGE_DAYS - 1)
    
    # Peak hours: 18-23 (6PM-11PM) - 50% of messages
    # Normal hours: 9-18 (9AM-6PM) - 30% of messages
    # Off hours: 0-9, 23-24 - 20% of messages
    rand = random.random()
    if rand < 0.5:
        hour = random.randint(18, 23)
    elif rand < 0.8:
        hour = random.randint(9, 17)
    else:
        hour = random.choice(list(range(0, 9)) + [23])
    
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    
    return start_date + timedelta(days=random_day, hours=hour, minutes=minute, seconds=second)

df['timestamp'] = [generate_realistic_timestamp() for _ in range(len(df))]

# Sort by user and time for rolling calculations
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# Assign channels (some channels are more toxic)
toxic_channels = np.random.choice(NUM_CHANNELS, size=int(NUM_CHANNELS * 0.2), replace=False)
def assign_channel(label):
    if label == 1 and random.random() < 0.4:
        return np.random.choice(toxic_channels)
    return np.random.randint(0, NUM_CHANNELS)

df['channel_id'] = df['label'].apply(assign_channel)

# ==========================================
# 5. ENHANCED MESSAGE-LEVEL FEATURES
# ==========================================
print("Computing Enhanced Message Features...")

# Basic features
df['msg_len'] = df['text'].astype(str).apply(len)
df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
df['caps_ratio'] = df['text'].astype(str).apply(
    lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
)

# Punctuation patterns (excessive punctuation often indicates emotion)
df['exclamation_count'] = df['text'].astype(str).apply(lambda x: x.count('!'))
df['question_count'] = df['text'].astype(str).apply(lambda x: x.count('?'))
df['ellipsis_count'] = df['text'].astype(str).apply(lambda x: x.count('...'))

# Personal targeting (you/your indicates directed harassment)
df['personal_pronoun_count'] = df['text'].astype(str).apply(
    lambda x: sum(1 for word in x.lower().split() if word in PERSONAL_PRONOUNS)
)

# Toxicity indicators
df['slur_count'] = df['text'].astype(str).apply(
    lambda x: sum(1 for w in BAD_WORDS if w in x.lower())
)

# Character repetition (e.g., "hahahaha", "nooooo")
df['char_repetition'] = df['text'].astype(str).apply(
    lambda x: len(re.findall(r'(.)\1{2,}', x.lower()))
)

# All caps words (shouting)
df['all_caps_words'] = df['text'].astype(str).apply(
    lambda x: sum(1 for word in x.split() if word.isupper() and len(word) > 1)
)

# ==========================================
# 6. USER HISTORICAL FEATURES (Rolling Windows)
# ==========================================
print("Computing User Behavioral History...")

df_indexed = df.set_index('timestamp')
grouped = df_indexed.groupby('user_id')

# 7-day window
df['user_7d_msg_count'] = grouped['label'].rolling('7D', closed='left').count().values
df['user_7d_bad_count'] = grouped['label'].rolling('7D', closed='left').sum().values

# 30-day window (longer-term behavior)
df['user_30d_msg_count'] = grouped['label'].rolling('30D', closed='left').count().values
df['user_30d_bad_count'] = grouped['label'].rolling('30D', closed='left').sum().values

# Message velocity (messages per day in last 7 days)
df['user_msg_velocity'] = df['user_7d_msg_count'] / 7

# Ratios
df['user_bad_ratio_7d'] = df['user_7d_bad_count'] / df['user_7d_msg_count']
df['user_bad_ratio_30d'] = df['user_30d_bad_count'] / df['user_30d_msg_count']

# Trend: Is user getting worse? (compare 7d to 30d ratio)
df['user_toxicity_trend'] = df['user_bad_ratio_7d'] - df['user_bad_ratio_30d']

# Clean up NaNs
df.fillna(0, inplace=True)
# ==========================================
# 7. CHANNEL-LEVEL FEATURES
# ==========================================
print("Computing Channel Context Features...")

# FIX: Sort by Channel -> Time so the rolling window works strictly in order
df = df.sort_values(['channel_id', 'timestamp'])
df_indexed = df.set_index('timestamp')

grouped_channel = df_indexed.groupby('channel_id')

# Channel toxicity in last 24 hours
df['channel_24h_msg_count'] = grouped_channel['label'].rolling('24H', closed='left').count().values
df['channel_24h_bad_count'] = grouped_channel['label'].rolling('24H', closed='left').sum().values

# Calculate ratio and handle division by zero
df['channel_toxicity_ratio'] = df['channel_24h_bad_count'] / df['channel_24h_msg_count']
df['channel_toxicity_ratio'] = df['channel_toxicity_ratio'].fillna(0)

# ==========================================
# 8. INTERACTION FEATURES
# ==========================================
print("Computing Interaction Features...")

# FIX: Re-sort back to User -> Time to calculate user-specific gaps
df = df.reset_index() # Bring timestamp back as column
df = df.sort_values(['user_id', 'timestamp'])

# Time since user's last message (in hours)
# Now that it's sorted by user, .diff() calculates the gap between *this* msg and their *previous* msg
df['hours_since_last_msg'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
df['hours_since_last_msg'] = df['hours_since_last_msg'].fillna(24)  # Default for first message

# Is this user new to the channel? (first 3 messages in channel)
df['user_channel_msg_count'] = df.groupby(['user_id', 'channel_id']).cumcount() + 1
df['is_new_to_channel'] = (df['user_channel_msg_count'] <= 3).astype(int)

# ==========================================
# 9. SAVE
# ==========================================
print("Saving Feature-Rich Dataset...")

# Final cleanup
df.fillna(0, inplace=True)
df = df.reset_index(drop=True)

df.to_parquet(OUTPUT_FILE, index=False)

print(f"\nDone! Saved to {OUTPUT_FILE}")
print(f"\nSample of features:")
feature_cols = ['text', 'label', 'user_id', 'msg_len', 'caps_ratio', 'personal_pronoun_count',
                'user_7d_bad_count', 'user_bad_ratio_7d', 'user_toxicity_trend', 
                'channel_toxicity_ratio', 'hours_since_last_msg']
print(df[feature_cols].head(10))

print(f"\nFeature summary:")
print(f"Total features: {len(df.columns)}")
print(f"Message-level features: 10")
print(f"User historical features: 8")
print(f"Channel context features: 4")
print(f"Interaction features: 3")