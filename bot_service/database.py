import aiosqlite
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

class ViolationLevel(Enum):
    SAFE = "SAFE"
    UNCERTAIN = "UNCERTAIN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

@dataclass
class ViolationRecord:
    user_id: str
    violations: int
    last_offense_time: datetime
    severity_history: List[str]

class DatabaseManager:
    """Async database manager with connection pooling"""

    def __init__(self, db_path: str = 'bot_memory.db'):
        self.db_path = db_path
        self._conn = Optional[aiosqlite.Connection] = None

    async def init_db(self):
        """Initialize database schema"""
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row

        await self._conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                violations INTEGER DEFAULT 0,
                last_offense_time TIMESTAMP,
                first_offense_time TIMESTAMP,
                severity_history TEXT DEFAULT '[]',
                account_created TIMESTAMP
            )
        ''')

        await self._conn.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                toxicity_score REAL,
                severity TEXT,
                action_taken TEXT,
                timestamp TIMESTAMP NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')

        await self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_logs_user_time 
            ON logs(user_id, timestamp DESC)
        ''')

        await self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_logs_severity 
            ON logs(severity, timestamp DESC)
        ''')

        await self._conn.commit()

    async def close(self):
        if self._conn:
            await self._conn.close()

    async def add_violation(self, user_id: str, severity: ViolationLevel) -> int:
        cursor = await self._conn.execute(
            "SELECT violations, severity_history FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = await cursor.fetchone()
        
        now = datetime.now()
        
        if row:
            new_count = row['violations'] + 1
            severity_history = json.loads(row['severity_history'])
            severity_history.append(severity.value)
            
            await self._conn.execute('''
                UPDATE users 
                SET violations = ?,
                    last_offense_time = ?,
                    severity_history = ?
                WHERE user_id = ?
            ''', (new_count, now, json.dumps(severity_history), user_id))
        else:
            new_count = 1
            await self._conn.execute('''
                INSERT INTO users (user_id, violations, first_offense_time, last_offense_time, severity_history)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, 1, now, now, json.dumps([severity.value])))
        
        await self._conn.commit()
        return new_count
    
    async def log_event(self, user_id: str,message: str, score: float, severity: str, action: str, 
                        metadata: Optional[Dict[str, Any]] = None):
        await self._conn.execute('''
            INSERT INTO logs (user_id, message, toxicity_score, severity, action_taken, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            message[:500],
            score,
            severity,
            action,
            datetime.now(),
            json.dumps(metadata or {})
        ))
        await self._conn.commit()

    async def get_user_violations(self, user_id: str) -> Dict[str, Any]:
        cursor = await self._conn.execute('''
            SELECT violations, last_offense_time, severity_history 
            FROM users 
            WHERE user_id = ?
        ''', (user_id,))
        
        row = await cursor.fetchone()
        
        if row:
            return {
                'count': row['violations'],
                'last_offense': row['last_offense_time'],
                'severity_history': json.loads(row['severity_history'])
            }
        return {'count': 0, 'last_offense': None, 'severity_history': []}
    
    async def clear_violations(self, user_id: str):
        await self._conn.execute('''
            UPDATE users 
            SET violations = 0, 
                severity_history = '[]'
            WHERE user_id = ?
        ''', (user_id,))
        await self._conn.commit()

    async def get_moderation_stats(self, days: int = 7) -> Dict[str, int]:
        cutoff = datetime.now() - timedelta(days=days)
        
        cursor = await self._conn.execute('''
            SELECT 
                COUNT(*) as total_violations,
                COUNT(DISTINCT user_id) as unique_users,
                SUM(CASE WHEN action_taken LIKE 'DELETE%' THEN 1 ELSE 0 END) as deleted_messages,
                SUM(CASE WHEN action_taken LIKE 'TIMEOUT%' THEN 1 ELSE 0 END) as timeouts,
                SUM(CASE WHEN severity = 'UNCERTAIN' OR action_taken = 'FLAGGED_REVIEW' THEN 1 ELSE 0 END) as pending_review
            FROM logs
            WHERE timestamp > ?
        ''', (cutoff,))
        
        row = await cursor.fetchone()
        
        return {
            'total_violations': row['total_violations'] or 0,
            'unique_users': row['unique_users'] or 0,
            'deleted_messages': row['deleted_messages'] or 0,
            'timeouts': row['timeouts'] or 0,
            'pending_review': row['pending_review'] or 0
        }
    
    async def get_pending_reviews(self, limit: int = 50) -> List[Dict[str, Any]]:
        cursor = await self._conn.execute('''
            SELECT user_id, message, toxicity_score, timestamp, metadata
            FROM logs
            WHERE severity = 'UNCERTAIN' OR action_taken = 'FLAGGED_REVIEW'
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = await cursor.fetchall()
        
        return [
            {
                'user_id': row['user_id'],
                'message': row['message'],
                'score': row['toxicity_score'],
                'timestamp': row['timestamp'],
                'metadata': json.loads(row['metadata'])
            }
            for row in rows
        ]