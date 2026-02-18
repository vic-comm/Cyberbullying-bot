import asyncio
import asyncpg
import os
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# THE SQL SCHEMA (Exactly what you provided)
# ═══════════════════════════════════════════════════════════════
FEEDBACK_SCHEMA = """
-- 1. FEEDBACK TABLE
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
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_feedback_log_id ON feedback(log_id);
CREATE INDEX IF NOT EXISTS idx_feedback_review_status ON feedback(admin_reviewed, created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_server ON feedback(server_id, admin_reviewed);
CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id, disputed_at);
CREATE INDEX IF NOT EXISTS idx_feedback_training ON feedback(used_in_training, admin_reviewed);
CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_unique_log ON feedback(log_id);

-- 2. ADMIN REVIEW QUEUE VIEW
CREATE OR REPLACE VIEW admin_review_queue AS
SELECT 
    f.id as feedback_id,
    f.log_id,
    f.user_id,
    f.server_id,
    f.platform,
    f.disputed_at,
    l.message as text,
    l.toxicity_score,
    l.severity,
    l.action_taken,
    l.timestamp as message_timestamp,
    f.predicted_label,
    f.predicted_score,
    f.user_claimed_label,
    f.dispute_reason,
    f.admin_reviewed,
    f.admin_decision,
    f.final_label,
    f.reviewed_by,
    f.reviewed_at,
    EXTRACT(EPOCH FROM (NOW() - f.disputed_at)) / 3600 AS hours_pending,
    (
        SELECT COUNT(*) 
        FROM server_user_violations v 
        WHERE v.user_id = f.user_id 
        AND v.server_id = f.server_id
        AND v.platform = f.platform
    ) as user_total_violations
FROM feedback f
JOIN logs l ON f.log_id = l.id
WHERE f.admin_reviewed = FALSE
ORDER BY f.disputed_at ASC;

-- 3. UNCERTAIN MESSAGES VIEW
CREATE OR REPLACE VIEW uncertain_messages AS
SELECT 
    l.id as log_id,
    l.user_id,
    l.server_id,
    l.platform,
    l.message as text,
    l.toxicity_score,
    l.severity,
    l.action_taken,
    l.timestamp,
    EXISTS(SELECT 1 FROM feedback f WHERE f.log_id = l.id) as has_feedback,
    EXTRACT(EPOCH FROM (NOW() - l.timestamp)) / 3600 AS hours_ago
FROM logs l
WHERE l.toxicity_score >= 0.3 
  AND l.toxicity_score <= 0.7
  AND l.severity IN ('LOW', 'MEDIUM', 'UNCERTAIN')
  AND l.timestamp > NOW() - INTERVAL '7 days'
ORDER BY l.timestamp DESC;

-- 4. FUNCTIONS

-- Get pending review count
CREATE OR REPLACE FUNCTION get_pending_review_count(p_server_id TEXT)
RETURNS INTEGER AS $$
BEGIN
    RETURN (
        SELECT COUNT(*) FROM feedback WHERE server_id = p_server_id AND admin_reviewed = FALSE
    ) + (
        SELECT COUNT(*) FROM uncertain_messages WHERE server_id = p_server_id AND has_feedback = FALSE
    );
END;
$$ LANGUAGE plpgsql;

-- Record user dispute
CREATE OR REPLACE FUNCTION record_user_dispute(
    p_log_id INTEGER,
    p_user_id TEXT,
    p_server_id TEXT,
    p_platform TEXT,
    p_user_claimed_label INTEGER,
    p_dispute_reason TEXT DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
    v_feedback_id INTEGER;
    v_log RECORD;
BEGIN
    SELECT * INTO v_log FROM logs WHERE id = p_log_id;
    IF NOT FOUND THEN RAISE EXCEPTION 'Log ID % not found', p_log_id; END IF;
    
    INSERT INTO feedback (
        log_id, user_id, server_id, platform, predicted_label, predicted_score, 
        predicted_severity, user_claimed_label, dispute_reason
    ) VALUES (
        p_log_id, p_user_id, p_server_id, p_platform,
        CASE WHEN v_log.severity IN ('LOW', 'MEDIUM', 'HIGH') THEN 1 ELSE 0 END,
        v_log.toxicity_score, v_log.severity, p_user_claimed_label, p_dispute_reason
    )
    ON CONFLICT (log_id) DO UPDATE SET
        user_claimed_label = EXCLUDED.user_claimed_label,
        dispute_reason = EXCLUDED.dispute_reason,
        disputed_at = NOW()
    RETURNING id INTO v_feedback_id;
    return v_feedback_id;
END;
$$ LANGUAGE plpgsql;

-- Admin review decision
CREATE OR REPLACE FUNCTION admin_review_feedback(
    p_feedback_id INTEGER,
    p_admin_id TEXT,
    p_decision TEXT,
    p_final_label INTEGER DEFAULT NULL,
    p_notes TEXT DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    v_feedback RECORD;
BEGIN
    SELECT * INTO v_feedback FROM feedback WHERE id = p_feedback_id;
    IF NOT FOUND THEN RAISE EXCEPTION 'Feedback ID % not found', p_feedback_id; END IF;
    
    IF p_decision = 'agree_with_model' THEN p_final_label := v_feedback.predicted_label;
    ELSIF p_decision = 'agree_with_user' THEN p_final_label := v_feedback.user_claimed_label;
    ELSIF p_final_label IS NULL THEN RAISE EXCEPTION 'Custom decision requires final_label';
    END IF;
    
    UPDATE feedback SET
        admin_reviewed = TRUE, admin_decision = p_decision, final_label = p_final_label,
        reviewed_by = p_admin_id, reviewed_at = NOW(), admin_notes = p_notes
    WHERE id = p_feedback_id;
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Bulk approve
CREATE OR REPLACE FUNCTION bulk_approve_model(p_feedback_ids INTEGER[], p_admin_id TEXT)
RETURNS INTEGER AS $$
DECLARE v_count INTEGER;
BEGIN
    UPDATE feedback SET
        admin_reviewed = TRUE, admin_decision = 'agree_with_model', final_label = predicted_label,
        reviewed_by = p_admin_id, reviewed_at = NOW()
    WHERE id = ANY(p_feedback_ids) AND admin_reviewed = FALSE;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$ LANGUAGE plpgsql;
"""

# ═══════════════════════════════════════════════════════════════
# MIGRATION FUNCTION
# ═══════════════════════════════════════════════════════════════

async def init_database():
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("❌ DATABASE_URL is not set!")
        return

    logger.info("🔄 Connecting to Database...")
    try:
        conn = await asyncpg.connect(db_url)
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return

    try:
        # 1. Ensure LOGS table exists (Dependencies first!)
        logger.info("🛠 Checking base tables...")
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id SERIAL PRIMARY KEY,
                user_id TEXT,
                server_id TEXT,
                platform TEXT,
                message TEXT,
                toxicity_score REAL,
                severity TEXT,
                action_taken TEXT,
                timestamp TIMESTAMP DEFAULT NOW(),
                metadata JSONB
            );
            
            CREATE TABLE IF NOT EXISTS server_user_violations (
                id SERIAL PRIMARY KEY,
                user_id TEXT,
                server_id TEXT,
                platform TEXT,
                violation_count INTEGER DEFAULT 0,
                last_violation_at TIMESTAMP
            );
        ''')

        # 2. Add Pardon Columns (Safe Migration)
        logger.info("🔄 Applying Pardon columns...")
        await conn.execute('''
            ALTER TABLE server_user_violations
            ADD COLUMN IF NOT EXISTS pardoned BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS pardoned_at TIMESTAMP,
            ADD COLUMN IF NOT EXISTS pardoned_by TEXT,
            ADD COLUMN IF NOT EXISTS pardon_reason TEXT;
        ''')

        # 3. Apply Feedback Schema
        logger.info("🔄 Applying Feedback System Schema...")
        await conn.execute(FEEDBACK_SCHEMA)
        
        logger.info("✅ Database Initialization Complete!")
        
    except Exception as e:
        logger.error(f"❌ Database Init Failed: {e}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(init_database())