import numpy as np
import re

BAD_WORDS = ["trash", "kill", "die", "ugly", "stupid", "idiot", "hate", "loser", "dumb"]
PERSONAL_PRONOUNS = ["you", "your", "you're", "ur", "u"]

def calculate_text_features(text: str) -> dict:
    """
    Computes static text features for a single string.
    Logic strictly mirrors 'generate_data.py' to prevent Training-Serving Skew.
    """
    # Handle edge cases
    text = str(text) if text is not None else ""
    if not text:
        return _get_empty_features()
        
    # Pre-computation
    words_case_sensitive = text.split()
    words_lower = text.lower().split()
    text_lower = text.lower()
    msg_len = len(text)
    
    # 1. Basic Stats
    word_count = len(words_case_sensitive)
    
    # 2. Character Patterns
    # Caps Ratio: sum(1 for c in x if c.isupper()) / len(x)
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / msg_len if msg_len > 0 else 0.0
    
    # 3. Punctuation Patterns
    exclamation_count = text.count('!')
    question_count = text.count('?')  # Renamed from question_mark_count to match training
    ellipsis_count = text.count('...')
    
    # 4. Linguistic Features
    # Personal Pronouns (Targeting): sum(1 for word in x.lower().split() if word in PERSONAL_PRONOUNS)
    personal_pronoun_count = sum(1 for w in words_lower if w in PERSONAL_PRONOUNS)
    
    # Character Repetition: len(re.findall(r'(.)\1{2,}', x.lower()))
    # Matches 3 or more repeated characters (e.g., "nooo")
    char_repetition = len(re.findall(r'(.)\1{2,}', text_lower))
    
    # All Caps Words: sum(1 for word in x.split() if word.isupper() and len(word) > 1)
    # Checks original case, ignores single letters like 'I' or 'A'
    all_caps_words = sum(1 for w in words_case_sensitive if w.isupper() and len(w) > 1)

    # 5. Toxicity Signals
    # Slur Count: sum(1 for w in BAD_WORDS if w in x.lower())
    # Note: Training logic checked if bad word is IN the text string, not exact word match
    slur_count = sum(1 for w in BAD_WORDS if w in text_lower)

    return {
        "msg_len": msg_len,
        "word_count": word_count,
        "caps_ratio": float(caps_ratio),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "ellipsis_count": ellipsis_count,
        "personal_pronoun_count": personal_pronoun_count,
        "char_repetition": char_repetition,
        "all_caps_words": all_caps_words,
        "slur_count": slur_count
    }

def _get_empty_features():
    """Returns 0/0.0 for all features to prevent errors on empty input."""
    return {
        "msg_len": 0, "word_count": 0, "caps_ratio": 0.0,
        "exclamation_count": 0, "question_count": 0, "ellipsis_count": 0,
        "personal_pronoun_count": 0, "char_repetition": 0, 
        "all_caps_words": 0, "slur_count": 0
    }