import asyncpg
import json
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

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
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.pool: Optional[asyncpg.Pool] = None

    async def init_db(self):        
        # Create connection pool
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20,
            command_timeout=60,
            statement_cache_size=0
        )
        
        logger.info("✅ Connected to Supabase (PostgreSQL)")
        
        await self._create_tables()
        logger.info("✅ Database schema initialized")

    async def _create_tables(self):
        """Create tables, views, functions, and apply schema migrations if they don't exist"""
        async with self.pool.acquire() as conn:
            
            # ═══════════════════════════════════════════════════════════════
            # 1. CORE TABLES
            # ═══════════════════════════════════════════════════════════════
            
            # Users table (legacy)
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    violations INTEGER DEFAULT 0,
                    last_offense_time TIMESTAMP,
                    first_offense_time TIMESTAMP,
                    severity_history JSONB DEFAULT '[]'::jsonb,
                    account_created TIMESTAMP DEFAULT NOW()
                )
            ''')

            # Logs table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    server_id TEXT,
                    platform TEXT NOT NULL DEFAULT 'discord',
                    message TEXT NOT NULL,
                    toxicity_score REAL,
                    severity TEXT,
                    action_taken TEXT,
                    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    explanation JSONB
                )
            ''')

            # Server configs
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS server_configs (
                    server_id TEXT NOT NULL,
                    platform TEXT NOT NULL DEFAULT 'discord',
                    server_name TEXT,
                    config_data JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (server_id, platform)
                )
            ''')

            # Server-specific user violations
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS server_user_violations (
                    id SERIAL PRIMARY KEY,
                    server_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    platform TEXT NOT NULL DEFAULT 'discord',
                    violation_count INTEGER DEFAULT 0,
                    last_violation_time TIMESTAMP,
                    first_violation_time TIMESTAMP,
                    severity_history JSONB DEFAULT '[]'::jsonb,
                    UNIQUE(server_id, user_id, platform)
                )
            ''')

            # ═══════════════════════════════════════════════════════════════
            # 2. SCHEMA MIGRATIONS (e.g., Adding Pardon Columns safely)
            # ═══════════════════════════════════════════════════════════════
            
            await conn.execute('''
                ALTER TABLE server_user_violations
                ADD COLUMN IF NOT EXISTS pardoned BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS pardoned_at TIMESTAMP,
                ADD COLUMN IF NOT EXISTS pardoned_by TEXT,
                ADD COLUMN IF NOT EXISTS pardon_reason TEXT;
            ''')

            # ═══════════════════════════════════════════════════════════════
            # 3. FEEDBACK SYSTEM (Tables, Views, Functions)
            # ═══════════════════════════════════════════════════════════════
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    log_id INTEGER NOT NULL REFERENCES logs(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL,
                    server_id TEXT NOT NULL,
                    platform TEXT DEFAULT 'discord',
                    predicted_label INTEGER NOT NULL,
                    predicted_score REAL,
                    predicted_severity TEXT,
                    user_claimed_label INTEGER NOT NULL,
                    dispute_reason TEXT,
                    disputed_at TIMESTAMP DEFAULT NOW(),
                    admin_reviewed BOOLEAN DEFAULT FALSE,
                    admin_decision TEXT,
                    final_label INTEGER,
                    reviewed_by TEXT,
                    reviewed_at TIMESTAMP,
                    admin_notes TEXT,
                    used_in_training BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(log_id)
                );

                -- Admin Review Queue View
                CREATE OR REPLACE VIEW admin_review_queue AS
                SELECT 
                    f.id as feedback_id, f.log_id, f.user_id, f.server_id, f.platform, f.disputed_at,
                    l.message as text, l.toxicity_score, l.severity, l.action_taken, l.timestamp as message_timestamp,
                    f.predicted_label, f.predicted_score, f.user_claimed_label, f.dispute_reason,
                    f.admin_reviewed, f.admin_decision, f.final_label, f.reviewed_by, f.reviewed_at,
                    EXTRACT(EPOCH FROM (NOW() - f.disputed_at)) / 3600 AS hours_pending,
                    (SELECT COUNT(*) FROM server_user_violations v WHERE v.user_id = f.user_id AND v.server_id = f.server_id AND v.platform = f.platform) as user_total_violations
                FROM feedback f JOIN logs l ON f.log_id = l.id WHERE f.admin_reviewed = FALSE ORDER BY f.disputed_at ASC;

                -- Uncertain Messages View
                CREATE OR REPLACE VIEW uncertain_messages AS
                SELECT 
                    l.id as log_id, l.user_id, l.server_id, l.platform, l.message as text, l.toxicity_score, l.severity, l.action_taken, l.timestamp,
                    EXISTS(SELECT 1 FROM feedback f WHERE f.log_id = l.id) as has_feedback,
                    EXTRACT(EPOCH FROM (NOW() - l.timestamp)) / 3600 AS hours_ago
                FROM logs l WHERE l.toxicity_score >= 0.3 AND l.toxicity_score <= 0.7 AND l.severity IN ('LOW', 'MEDIUM', 'UNCERTAIN') AND l.timestamp > NOW() - INTERVAL '7 days' ORDER BY l.timestamp DESC;

                -- Function: Get pending review count
                CREATE OR REPLACE FUNCTION get_pending_review_count(p_server_id TEXT) RETURNS INTEGER AS $$
                BEGIN
                    RETURN (SELECT COUNT(*) FROM feedback WHERE server_id = p_server_id AND admin_reviewed = FALSE) + 
                           (SELECT COUNT(*) FROM uncertain_messages WHERE server_id = p_server_id AND has_feedback = FALSE);
                END;
                $$ LANGUAGE plpgsql;

                -- Function: Record user dispute
                CREATE OR REPLACE FUNCTION record_user_dispute(
                    p_log_id INTEGER, p_user_id TEXT, p_server_id TEXT, p_platform TEXT, p_user_claimed_label INTEGER, p_dispute_reason TEXT DEFAULT NULL
                ) RETURNS INTEGER AS $$
                DECLARE v_feedback_id INTEGER; v_log RECORD;
                BEGIN
                    SELECT * INTO v_log FROM logs WHERE id = p_log_id;
                    IF NOT FOUND THEN RAISE EXCEPTION 'Log ID % not found', p_log_id; END IF;
                    INSERT INTO feedback (log_id, user_id, server_id, platform, predicted_label, predicted_score, predicted_severity, user_claimed_label, dispute_reason) 
                    VALUES (p_log_id, p_user_id, p_server_id, p_platform, CASE WHEN v_log.severity IN ('LOW', 'MEDIUM', 'HIGH') THEN 1 ELSE 0 END, v_log.toxicity_score, v_log.severity, p_user_claimed_label, p_dispute_reason)
                    ON CONFLICT (log_id) DO UPDATE SET user_claimed_label = EXCLUDED.user_claimed_label, dispute_reason = EXCLUDED.dispute_reason, disputed_at = NOW() RETURNING id INTO v_feedback_id;
                    RETURN v_feedback_id;
                END;
                $$ LANGUAGE plpgsql;

                -- Function: Admin review decision
                CREATE OR REPLACE FUNCTION admin_review_feedback(
                    p_feedback_id INTEGER, p_admin_id TEXT, p_decision TEXT, p_final_label INTEGER DEFAULT NULL, p_notes TEXT DEFAULT NULL
                ) RETURNS BOOLEAN AS $$
                DECLARE v_feedback RECORD;
                BEGIN
                    SELECT * INTO v_feedback FROM feedback WHERE id = p_feedback_id;
                    IF NOT FOUND THEN RAISE EXCEPTION 'Feedback ID % not found', p_feedback_id; END IF;
                    IF p_decision = 'agree_with_model' THEN p_final_label := v_feedback.predicted_label;
                    ELSIF p_decision = 'agree_with_user' THEN p_final_label := v_feedback.user_claimed_label;
                    ELSIF p_final_label IS NULL THEN RAISE EXCEPTION 'Custom decision requires final_label'; END IF;
                    UPDATE feedback SET admin_reviewed = TRUE, admin_decision = p_decision, final_label = p_final_label, reviewed_by = p_admin_id, reviewed_at = NOW(), admin_notes = p_notes WHERE id = p_feedback_id;
                    RETURN TRUE;
                END;
                $$ LANGUAGE plpgsql;

                -- Function: Bulk approve model
                CREATE OR REPLACE FUNCTION bulk_approve_model(p_feedback_ids INTEGER[], p_admin_id TEXT) RETURNS INTEGER AS $$
                DECLARE v_count INTEGER;
                BEGIN
                    UPDATE feedback SET admin_reviewed = TRUE, admin_decision = 'agree_with_model', final_label = predicted_label, reviewed_by = p_admin_id, reviewed_at = NOW() WHERE id = ANY(p_feedback_ids) AND admin_reviewed = FALSE;
                    GET DIAGNOSTICS v_count = ROW_COUNT;
                    RETURN v_count;
                END;
                $$ LANGUAGE plpgsql;
            ''')

            # ═══════════════════════════════════════════════════════════════
            # 4. INDEXES
            # ═══════════════════════════════════════════════════════════════
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_logs_platform_user_time ON logs(platform, user_id, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_logs_server_time ON logs(server_id, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_logs_severity ON logs(severity, timestamp DESC) WHERE severity IN ('LOW', 'MEDIUM', 'HIGH');
                CREATE INDEX IF NOT EXISTS idx_server_violations_lookup ON server_user_violations(platform, server_id, user_id);
                CREATE INDEX IF NOT EXISTS idx_server_configs_lookup ON server_configs(platform, server_id);
                CREATE INDEX IF NOT EXISTS idx_logs_user_lookup ON logs(user_id);
                CREATE INDEX IF NOT EXISTS idx_feedback_review_status ON feedback(admin_reviewed, created_at);
                CREATE INDEX IF NOT EXISTS idx_feedback_server ON feedback(server_id, admin_reviewed);
                CREATE INDEX IF NOT EXISTS idx_feedback_training ON feedback(used_in_training, admin_reviewed);
            ''')

            logger.info("✅ Tables, Views, Functions, and Indexes verified/created")

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("✅ Database connection pool closed")

    # ========== SERVER CONFIG METHODS ==========
    
    async def get_server_config(
        self, 
        server_id: str, 
        platform: str = 'discord'
    ) -> Optional[Dict[str, Any]]:
        """Get server configuration for a specific platform"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT * FROM server_configs 
                WHERE server_id = $1 AND platform = $2
            ''', server_id, platform)
            
            if row:
                config_data = json.loads(row['config_data'])
                
                config_data['server_id'] = server_id
                config_data['server_name'] = row['server_name']
                return config_data
            return None
    
    async def save_server_config(
        self, 
        config_dict: Dict[str, Any],
        platform: str = 'discord'
    ):
        """
        Save config using Atomic Upsert and JSONB.
        This handles both INSERT and UPDATE in one safe command.
        """
        # 1. Extract the Primary Key fields
        server_id = config_dict.pop('server_id')
        
        # 2. Extract standard columns (if any exist outside JSON)
        # In our optimized schema, 'server_name' is a column, rest is JSON
        server_name = config_dict.pop('server_name', 'Unknown')
        config_json_str = json.dumps(config_dict)

        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO server_configs (server_id, platform, server_name, config_data, updated_at)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (server_id, platform)
                DO UPDATE SET
                    config_data = EXCLUDED.config_data,
                    server_name = EXCLUDED.server_name,
                    updated_at = NOW()
            ''', server_id, platform, server_name, config_json_str)
            
            logger.debug(f"Saved config for server {server_id} ({platform})")
        
    async def get_server_user_violations(
        self, 
        server_id: str, 
        user_id: str,
        platform: str = 'discord'
    ) -> Dict[str, Any]:
        """Get user's violation count for specific server/platform"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT violation_count, last_violation_time, 
                       severity_history, first_violation_time
                FROM server_user_violations
                WHERE server_id = $1 AND user_id = $2 AND platform = $3
            ''', server_id, user_id, platform)
            
            if row:
                history = row['severity_history']
                if isinstance(history, str):
                    import json
                    try:
                        history = json.loads(history)
                    except:
                        history = []
                return {
                    'count': row['violation_count'],
                    'last_offense': row['last_violation_time'],
                    'first_offense': row['first_violation_time'],
                    'severity_history': row['severity_history'] if row['severity_history'] else []
                }
            return {
                'count': 0, 
                'last_offense': None, 
                'first_offense': None, 
                'severity_history': []
            }
    
    async def clear_server_violations(
        self, 
        server_id: str, 
        user_id: str,
        platform: str = 'discord'
    ):
        """Clear violations for user in specific server/platform"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                UPDATE server_user_violations
                SET violation_count = 0,
                    severity_history = '[]'::jsonb
                WHERE server_id = $1 AND user_id = $2 AND platform = $3
            ''', server_id, user_id, platform)
    
    # ========== LOG METHODS ==========
    
    async def log_event(
        self,
        user_id: str,
        message: str,
        score: float,
        severity: str,
        action: str,
        server_id: Optional[str] = None,
        platform: str = 'discord',
        metadata: Optional[Dict[str, Any]] = None,
        explanation: Optional[Dict[str, Any]] = None
    ) -> int:
        """Log moderation event and return log ID"""
        async with self.pool.acquire() as conn:
            log_id = await conn.fetchval('''
                INSERT INTO logs (
                    user_id, server_id, platform, message, toxicity_score, 
                    severity, action_taken, timestamp, metadata, explanation
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
            ''', 
                user_id,
                server_id,
                platform,
                message[:1000],  
                score,
                severity,
                action,
                datetime.now(),
                json.dumps(metadata or {}),
                json.dumps(explanation) if explanation else None
            )
            
            return log_id
    
    async def update_log_explanation(self, log_id: int, explanation: Dict[str, Any]):
        """Update explanation for an existing log entry"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                UPDATE logs 
                SET explanation = $1
                WHERE id = $2
            ''', json.dumps(explanation), log_id)
            
            logger.debug(f"Updated explanation for log {log_id}")
    
    async def get_log_by_message(
        self, 
        user_id: str, 
        server_id: str, 
        platform: str = 'discord',
        timestamp_window: int = 60
    ) -> Optional[int]:
        """
        Find a log entry within the last N seconds
        Used to link background explanation to the violation
        """
        async with self.pool.acquire() as conn:
            cutoff = datetime.now() - timedelta(seconds=timestamp_window)
            
            log_id = await conn.fetchval('''
                SELECT id FROM logs
                WHERE user_id = $1 
                  AND server_id = $2 
                  AND platform = $3
                  AND timestamp > $4
                ORDER BY timestamp DESC
                LIMIT 1
            ''', user_id, server_id, platform, cutoff)
            
            return log_id
    
    async def get_latest_user_violation(
        self, 
        server_id: str, 
        user_id: str,
        platform: str = 'discord',
        hours: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        Get user's most recent violation for explanation
        """
        async with self.pool.acquire() as conn:
            cutoff = datetime.now() - timedelta(hours=hours)
            
            row = await conn.fetchrow('''
                SELECT id, message, severity, action_taken, 
                       toxicity_score, timestamp, explanation
                FROM logs
                WHERE server_id = $1 
                  AND user_id = $2 
                  AND platform = $3
                  AND timestamp > $4
                  AND severity IN ('LOW', 'MEDIUM', 'HIGH')
                ORDER BY timestamp DESC
                LIMIT 1
            ''', server_id, user_id, platform, cutoff)
            
            if row:
                return {
                    'id': row['id'],
                    'message': row['message'],
                    'severity': row['severity'],
                    'action_taken': row['action_taken'],
                    'toxicity_score': row['toxicity_score'],
                    'timestamp': row['timestamp'],
                    'explanation': row['explanation']  # Already JSONB
                }
            
            return None
    
    # ========== STATS & MONITORING ==========
    
    async def get_moderation_stats(
        self, 
        days: int = 7, 
        server_id: Optional[str] = None,
        platform: str = 'discord'
    ) -> Dict[str, int]:
        """Get moderation statistics"""
        async with self.pool.acquire() as conn:
            cutoff = datetime.now() - timedelta(days=days)
            
            query = '''
                SELECT 
                    COUNT(*) as total_violations,
                    COUNT(DISTINCT user_id) as unique_users,
                    SUM(CASE WHEN action_taken LIKE 'DELETE%' THEN 1 ELSE 0 END) as deleted_messages,
                    SUM(CASE WHEN action_taken LIKE 'TIMEOUT%' THEN 1 ELSE 0 END) as timeouts,
                    SUM(CASE WHEN severity = 'UNCERTAIN' OR action_taken = 'FLAGGED_REVIEW' THEN 1 ELSE 0 END) as pending_review
                FROM logs
                WHERE timestamp > $1 AND platform = $2
            '''
            
            params = [cutoff, platform]
            
            if server_id:
                query += " AND server_id = $3"
                params.append(server_id)
            
            row = await conn.fetchrow(query, *params)
            
            return {
                'total_violations': row['total_violations'] or 0,
                'unique_users': row['unique_users'] or 0,
                'deleted_messages': row['deleted_messages'] or 0,
                'timeouts': row['timeouts'] or 0,
                'pending_review': row['pending_review'] or 0
            }
    
    async def get_pending_reviews(
        self, 
        limit: int = 50, 
        server_id: Optional[str] = None,
        platform: str = 'discord'
    ) -> List[Dict[str, Any]]:
        """Get messages pending human review"""
        async with self.pool.acquire() as conn:
            query = '''
                SELECT user_id, server_id, platform, message, 
                       toxicity_score, timestamp, metadata
                FROM logs
                WHERE (severity = 'UNCERTAIN' OR action_taken = 'FLAGGED_REVIEW')
                  AND platform = $1
            '''
            
            params = [platform]
            
            if server_id:
                query += " AND server_id = $2"
                params.append(server_id)
            
            query += f" ORDER BY timestamp DESC LIMIT ${len(params)+1}"
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            
            return [
                {
                    'user_id': row['user_id'],
                    'server_id': row['server_id'],
                    'platform': row['platform'],
                    'message': row['message'],
                    'score': row['toxicity_score'],
                    'timestamp': row['timestamp'],
                    'metadata': row['metadata']
                }
                for row in rows
            ]
        
    async def add_violation(
            self, 
            user_id: str, 
            server_id: str, 
            severity: str,
            platform: str = 'discord'
        ) -> int:
            """
            Records a violation. Updates both global history AND server-specific strikes.
            Returns the new SERVER-SPECIFIC violation count (for punishment logic).
            """
            now = datetime.now()
            
            async with self.pool.acquire() as conn:
                # 1. Update Global History (Legacy/Backup)
                # We keep this so you can track a user's behavior across ALL servers
                # Note: We use json.dumps([severity]) to ensure it appends as a list
                await conn.execute('''
                    INSERT INTO users (user_id, violations, first_offense_time, last_offense_time, severity_history)
                    VALUES ($1, 1, $2, $2, $3)
                    ON CONFLICT (user_id)
                    DO UPDATE SET
                        violations = users.violations + 1,
                        last_offense_time = $2,
                        severity_history = users.severity_history || $3::jsonb
                ''', user_id, now, json.dumps([severity]))

                # 2. Update Server-Specific Strikes (The Real Logic)
                # This includes the "Relapse Logic" (pardoned = FALSE)
                new_server_count = await conn.fetchval('''
                    INSERT INTO server_user_violations (server_id, user_id, platform, violation_count, severity_history)
                    VALUES ($1, $2, $3, 1, $4)
                    ON CONFLICT (server_id, user_id, platform)
                    DO UPDATE SET
                        violation_count = server_user_violations.violation_count + 1,
                        last_violation_time = NOW(),
                        severity_history = server_user_violations.severity_history || $4::jsonb,
                        
                        -- ⚠️ CRITICAL: Revoke pardon if they relapse
                        pardoned = FALSE,
                        pardoned_at = NULL,
                        pardoned_by = NULL,
                        pardon_reason = NULL
                        
                    RETURNING violation_count
                ''', server_id, user_id, platform, json.dumps([severity]))
                
                return new_server_count  
          
    async def get_user_violations(self, user_id: str) -> Dict[str, Any]:
        """Legacy method - global violations"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT violations, last_offense_time, severity_history 
                FROM users 
                WHERE user_id = $1
            ''', user_id)
            
            if row:
                return {
                    'count': row['violations'],
                    'last_offense': row['last_offense_time'],
                    'severity_history': row['severity_history'] if row['severity_history'] else []
                }
            return {
                'count': 0, 
                'last_offense': None, 
                'severity_history': []
            }
    
    async def clear_violations(self, user_id: str):
        """Legacy method - clear global violations"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                UPDATE users 
                SET violations = 0, 
                    severity_history = '[]'::jsonb
                WHERE user_id = $1
            ''', user_id)

    async def pardon_user_violations(
        self,
        server_id: str,
        user_id: str,
        admin_id: str,
        reason: str = "Admin discretion",
        platform: str = 'discord'
    ) -> Dict[str, Any]:
        """
        Pardon a user's violations.
        
        WHAT THIS DOES:
        - Resets active_strikes to 0 (no more punishment)
        - Marks as pardoned (audit trail)
        - Keeps all violation records (ML features unaffected)
        """
        async with self.pool.acquire() as conn:
            # Get current state before pardoning
            current = await conn.fetchrow('''
                SELECT violation_count
                FROM server_user_violations
                WHERE server_id = $1
                AND user_id = $2
                AND platform = $3
            ''', server_id, user_id, platform)
            
            # Nothing to pardon
            if not current or current['violation_count'] == 0:
                return {
                    'success': False,
                    'reason': 'no_active_strikes',
                    'previous_count': 0
                }
            
            previous_count = current['violation_count']
            
            # Pardon: reset strikes, keep history
            await conn.execute('''
                UPDATE server_user_violations
                SET
                    violation_count = 0,
                    pardoned        = TRUE,
                    pardoned_at     = NOW(),
                    pardoned_by     = $1,
                    pardon_reason   = $2
                WHERE server_id = $3
                AND user_id   = $4
                AND platform  = $5
            ''', admin_id, reason, server_id, user_id, platform)
            
            return {
                'success': True,
                'previous_count': previous_count,
                'admin_id': admin_id,
                'reason': reason
            }


    async def get_user_violation_history(
        self,
        server_id: str,
        user_id: str,
        platform: str = 'discord'
    ) -> Dict[str, Any]:
        """
        Get full violation history for admin review.
        
        Returns:
        - active_strikes:          What matters for punishment (respects pardons)
        - total_lifetime_violations: Real behavior count (ignores pardons)
        - recent_violations:       Last 5 logs for context
        """
        async with self.pool.acquire() as conn:
            # Current strike status
            current = await conn.fetchrow('''
                SELECT
                    violation_count,
                    pardoned,
                    pardoned_at,
                    pardoned_by,
                    pardon_reason,
                    last_violation_time,
                    first_violation_time
                FROM server_user_violations
                WHERE server_id = $1
                AND user_id   = $2
                AND platform  = $3
            ''', server_id, user_id, platform)
            
            # Total lifetime violations (ML uses this - no pardon filter)
            total = await conn.fetchval('''
                SELECT COUNT(*)
                FROM logs
                WHERE server_id = $1
                AND user_id   = $2
                AND platform  = $3
                AND severity IN ('LOW', 'MEDIUM', 'HIGH')
            ''', server_id, user_id, platform)
            
            # Recent violations for context
            recent = await conn.fetch('''
                SELECT
                    id,
                    message,
                    severity,
                    timestamp,
                    action_taken
                FROM logs
                WHERE server_id = $1
                AND user_id   = $2
                AND platform  = $3
                AND severity IN ('LOW', 'MEDIUM', 'HIGH')
                ORDER BY timestamp DESC
                LIMIT 5
            ''', server_id, user_id, platform)
            
            return {
                # For punishment decisions
                'active_strikes': current['violation_count'] if current else 0,
                'is_pardoned': current['pardoned'] if current else False,
                'pardoned_at': current['pardoned_at'] if current else None,
                'pardoned_by': current['pardoned_by'] if current else None,
                'pardon_reason': current['pardon_reason'] if current else None,
                
                # For full context
                'total_lifetime_violations': total or 0,
                'last_violation': current['last_violation_time'] if current else None,
                'recent_violations': [dict(r) for r in recent]
            }
    async def get_log(self, log_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a specific log by ID (Useful for appeals)"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT id, message, explanation, timestamp, severity
                FROM logs
                WHERE id = $1
            ''', log_id)
            
            if row:
                data = dict(row)
                
                # 🔴 FIX: Parse explanation if string
                if isinstance(data.get('explanation'), str):
                    import json
                    try:
                        data['explanation'] = json.loads(data['explanation'])
                    except:
                        data['explanation'] = None
                        
                return data
            return None
        
    async def record_user_dispute(
        self,
        log_id: int,
        user_id: str,
        server_id: str,
        user_claimed_label: int,  # 0=safe, 1=toxic
        platform: str = 'discord',
        dispute_reason: str = None
    ) -> int:
        """
        Record when a user disputes a moderation decision.
        
        Args:
            log_id: ID of the log entry being disputed
            user_id: User who is disputing
            server_id: Server where dispute occurred
            user_claimed_label: What user claims (0=safe, 1=toxic)
            platform: Platform (discord/slack/whatsapp)
            dispute_reason: Optional text reason from user
        
        Returns:
            feedback_id: ID of created/updated feedback entry
        """
        async with self.pool.acquire() as conn:
            feedback_id = await conn.fetchval(
                'SELECT record_user_dispute($1, $2, $3, $4, $5, $6)',
                log_id,
                user_id,
                server_id,
                platform,
                user_claimed_label,
                dispute_reason
            )
            return feedback_id

    # ───────────────────────────────────────────────────────────────
    # ADMIN REVIEW QUEUE
    # ───────────────────────────────────────────────────────────────

    async def get_pending_review_count(
        self,
        server_id: str
    ) -> int:
        """Get count of items pending admin review"""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                'SELECT get_pending_review_count($1)',
                server_id
            )
            return count or 0


    async def get_review_queue(
        self,
        server_id: str,
        limit: int = 50,
        offset: int = 0,
        filter_by: str = None,
        user_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get a unified queue of items needing review:
        1. User Disputes (High Priority)
        2. Uncertain Predictions (Proactive Review)
        """
        async with self.pool.acquire() as conn:
            # We construct a query that standardizes columns from both sources
            query = '''
            WITH combined_queue AS (
                -- 1. USER DISPUTES
                SELECT 
                    'dispute' as type,
                    f.id as feedback_id,
                    f.log_id,
                    f.user_id,
                    f.server_id,
                    l.message as text,
                    l.toxicity_score,
                    f.predicted_label,
                    f.user_claimed_label,
                    f.dispute_reason,
                    f.disputed_at as sort_time
                FROM feedback f
                JOIN logs l ON f.log_id = l.id
                WHERE f.server_id = $1 AND f.admin_reviewed = FALSE
                
                UNION ALL
                
                -- 2. UNCERTAIN MESSAGES (No feedback ID yet)
                SELECT 
                    'uncertain' as type,
                    NULL as feedback_id, -- Placeholder
                    l.id as log_id,
                    l.user_id,
                    l.server_id,
                    l.message as text,
                    l.toxicity_score,
                    CASE WHEN l.severity IN ('LOW','MEDIUM','HIGH') THEN 1 ELSE 0 END as predicted_label,
                    NULL as user_claimed_label, -- No user claim
                    'Model confidence is low (' || ROUND(l.toxicity_score::numeric * 100, 1) || '%)' as dispute_reason,
                    l.timestamp as sort_time
                FROM logs l
                WHERE l.server_id = $1
                  AND l.toxicity_score BETWEEN 0.3 AND 0.7
                  -- Exclude items that already have feedback (don't show duplicates)
                  AND NOT EXISTS (SELECT 1 FROM feedback f WHERE f.log_id = l.id)
            )
            SELECT * FROM combined_queue
            ORDER BY 
                CASE WHEN type = 'dispute' THEN 0 ELSE 1 END, -- Show disputes first
                sort_time ASC
            LIMIT $2 OFFSET $3
            '''
            
            rows = await conn.fetch(query, server_id, limit, offset)
            return [dict(row) for row in rows]

    async def get_uncertain_messages(
        self,
        server_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get uncertain messages (0.3-0.7 confidence) for admin review"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM uncertain_messages
                WHERE server_id = $1
                AND has_feedback = FALSE
                ORDER BY timestamp DESC
                LIMIT $2
            ''', server_id, limit)
            
            return [dict(row) for row in rows]


    async def get_review_queue_grouped(
        self,
        server_id: str,
        group_by: str = 'user'  # 'user' | 'channel' | 'time'
    ) -> Dict[str, List[Dict]]:
        """
        Get review queue grouped for bulk operations.
        
        Returns dict like:
        {
            'user_123': [item1, item2, item3],
            'user_456': [item4, item5]
        }
        """
        async with self.pool.acquire() as conn:
            if group_by == 'user':
                rows = await conn.fetch('''
                    SELECT 
                        user_id,
                        json_agg(
                            json_build_object(
                                'feedback_id', feedback_id,
                                'log_id', log_id,
                                'text', text,
                                'toxicity_score', toxicity_score,
                                'predicted_label', predicted_label,
                                'user_claimed_label', user_claimed_label,
                                'disputed_at', disputed_at
                            ) ORDER BY disputed_at DESC
                        ) as items
                    FROM admin_review_queue
                    WHERE server_id = $1
                    GROUP BY user_id
                    ORDER BY COUNT(*) DESC
                ''', server_id)
                
                return {row['user_id']: row['items'] for row in rows}
            
            # Add other grouping options as needed
            return {}


    # ───────────────────────────────────────────────────────────────
    # ADMIN REVIEW ACTIONS
    # ───────────────────────────────────────────────────────────────

    async def admin_review_feedback(
        self,
        feedback_id: int,
        admin_id: str,
        decision: str,  # 'agree_with_model' | 'agree_with_user' | 'custom'
        final_label: int = None,
        notes: str = None
    ) -> bool:
        """
        Record admin's review decision on a feedback item.
        """
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                'SELECT admin_review_feedback($1, $2, $3, $4, $5)',
                feedback_id,
                admin_id,
                decision,
                final_label,
                notes
            )
            return result


    async def bulk_approve_model(
        self,
        feedback_ids: List[int],
        admin_id: str
    ) -> int:
        """
        Bulk approve: admin agrees with model on multiple items.
        
        Returns count of items approved.
        """
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                'SELECT bulk_approve_model($1, $2)',
                feedback_ids,
                admin_id
            )
            return count or 0


    async def bulk_approve_users(
        self,
        feedback_ids: List[int],
        admin_id: str
    ) -> int:
        """
        Bulk approve: admin agrees with users on multiple items.
        """
        async with self.pool.acquire() as conn:
            # Use similar logic to bulk_approve_model but agree with user
            await conn.execute('''
                UPDATE feedback SET
                    admin_reviewed = TRUE,
                    admin_decision = 'agree_with_user',
                    final_label = user_claimed_label,
                    reviewed_by = $2,
                    reviewed_at = NOW()
                WHERE id = ANY($1)
                AND admin_reviewed = FALSE
            ''', feedback_ids, admin_id)
            
            return len(feedback_ids)


    # ───────────────────────────────────────────────────────────────
    # STATS & ANALYTICS
    # ───────────────────────────────────────────────────────────────

    async def get_feedback_stats(
        self,
        server_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get feedback statistics for admin dashboard"""
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow('''
                SELECT
                    COUNT(*) as total_disputes,
                    SUM(CASE WHEN admin_reviewed THEN 1 ELSE 0 END) as reviewed,
                    SUM(CASE WHEN NOT admin_reviewed THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN admin_decision = 'agree_with_model' THEN 1 ELSE 0 END) as model_correct,
                    SUM(CASE WHEN admin_decision = 'agree_with_user' THEN 1 ELSE 0 END) as user_correct,
                    AVG(EXTRACT(EPOCH FROM (reviewed_at - disputed_at)) / 3600) as avg_review_time_hours
                FROM feedback
                WHERE server_id = $1
                AND disputed_at > NOW() - ($2 || ' days')::INTERVAL
            ''', server_id, str(days))
            
            return dict(stats) if stats else {}


    async def get_top_disputing_users(
        self,
        server_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get users with most disputes (potential bad actors)"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT 
                    user_id,
                    COUNT(*) as dispute_count,
                    SUM(CASE WHEN admin_decision = 'agree_with_user' THEN 1 ELSE 0 END) as correct_disputes,
                    SUM(CASE WHEN admin_decision = 'agree_with_model' THEN 1 ELSE 0 END) as wrong_disputes,
                    MAX(disputed_at) as last_dispute
                FROM feedback
                WHERE server_id = $1
                AND disputed_at > NOW() - INTERVAL '30 days'
                GROUP BY user_id
                ORDER BY COUNT(*) DESC
                LIMIT $2
            ''', server_id, limit)
            
            return [dict(row) for row in rows]


































