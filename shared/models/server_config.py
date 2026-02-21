from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Optional
import json

class ModerationAction(Enum):
    NONE = "none"
    WARN = "warn"
    DELETE = "delete"
    TIMEOUT = "timeout"
    KICK = "kick"
    BAN = "ban"

class SeverityLevel(Enum):
    """Severity levels for violations"""
    SAFE = "SAFE"
    UNCERTAIN = "UNCERTAIN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

@dataclass
class ServerConfig:
    """Configuration for a Discord server's moderation settings"""
    
    server_id: str
    server_name: str = "Unknown Server"
    platform: str = "discord"
    
    strikes_before_timeout: int = 3
    strikes_before_kick: int = 5
    strikes_before_ban: int = 7
    strike_decay_days: int = 30  # Strikes expire after X days
    
    low_severity_action: str = "warn"      # WARN, DELETE, TIMEOUT
    medium_severity_action: str = "delete"  # DELETE, TIMEOUT
    high_severity_action: str = "timeout"   # TIMEOUT, KICK, BAN
    
    timeout_duration_low: int = 10        # 10 minutes
    timeout_duration_medium: int = 60     # 1 hour
    timeout_duration_high: int = 1440     # 24 hours
    
    threshold_low: float = 0.3      # Below this = safe
    threshold_medium: float = 0.6   # Above this = medium severity
    threshold_high: float = 0.8     # Above this = high severity
    uncertainty_min: float = 0.45   # Uncertainty range start
    uncertainty_max: float = 0.55   # Uncertainty range end
    
    auto_moderate: bool = True              # Enable automatic moderation
    require_human_review: bool = False      # Flag all for review before action
    send_dm_warnings: bool = True           # DM users about violations
    delete_after_timeout: bool = True       # Delete warning messages after X seconds
    warning_delete_delay: int = 30          # Seconds before deleting warnings
    
    log_channel_id: Optional[str] = None    # Channel for mod logs
    alert_channel_id: Optional[str] = None  # Channel for high-severity alerts
    
    exempt_role_ids: List[str] = None       # Roles immune to auto-mod
    exempt_channel_ids: List[str] = None    # Channels to ignore
    monitored_channel_ids: List[str] = None # Only moderate these (if set)
    
    warning_message_template: str = (
        "{mention}, please keep conversations respectful. "
        "Further violations may result in action."
    )
    
    timeout_message_template: str = (
        "🚫 {mention} has been timed out for {duration}. "
        "Reason: {reason}"
    )
    
    strike_message_template: str = (
        "🚫 {mention}, your message violated community guidelines. "
        "**Strike {count}/{max}**"
    )
    
    def __post_init__(self):
        """Initialize mutable defaults"""
        if self.exempt_role_ids is None:
            self.exempt_role_ids = []
        if self.exempt_channel_ids is None:
            self.exempt_channel_ids = []
        if self.monitored_channel_ids is None:
            self.monitored_channel_ids = []
    
    def to_dict(self) -> dict:
        """Convert to dictionary for database storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ServerConfig':
        if 'config_data' in data and isinstance(data['config_data'], dict):
            flat_data = data.copy()
            nested = flat_data.pop('config_data')
            flat_data.update(nested)
        else:
            flat_data = data

        valid_keys = cls.__dataclass_fields__.keys()
        clean_data = {k: v for k, v in flat_data.items() if k in valid_keys}
        
        return cls(**clean_data)
        