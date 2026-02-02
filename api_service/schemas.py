# schemas.py - Created for Cyberbullying Bot Project
from pydantic import BaseModel, Field
from typing import Optional

class PredictionRequest(BaseModel):
    text: str = Field(..., example="You are trash at this game", min_length=1)
    user_id: str = Field(..., example="123456789")
    
    msg_len: int = Field(0, description="Length of message")
    caps_ratio: float = Field(0.0, description="Ratio of uppercase letters")
    personal_pronoun_count: int = Field(0, description="Count of you/your")
    slur_count: int = Field(0, description="Count of bad words")
    user_bad_ratio_7d: float = Field(0.0, description="User's toxicity ratio last 7 days")
    user_toxicity_trend: float = Field(0.0, description="Is user getting worse?")
    channel_toxicity_ratio: float = Field(0.0, description="How toxic is the channel?")
    hours_since_last_msg: float = Field(24.0, description="Hours since last active")
    is_new_to_channel: int = Field(0, description="1 if new user, 0 otherwise")

class PredictionResponse(BaseModel):
    is_toxic: bool
    confidence: float
    model_version: str
    processing_time_ms: float