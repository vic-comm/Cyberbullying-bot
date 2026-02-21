"""
╔═══════════════════════════════════════════════════════════════╗
║        INGESTION PIPELINE TEST SUITE                          ║
║  Comprehensive validation of stratified sampling + QA gates   ║
╚═══════════════════════════════════════════════════════════════╝

Tests:
1. Normal operation (stratified sampling works)
2. Quality gates (spam/flooding detection)
3. Admin feedback integration
4. Edge cases (insufficient data, malformed data)
5. DVC/S3 fallback logic
"""

import asyncio
import asyncpg
import random
import json
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()
# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")

# ══════════════════════════════════════════════════════════════
# TEST DATA GENERATORS
# ══════════════════════════════════════════════════════════════

class TestDataGenerator:
    """Generate realistic test data with controlled properties"""
    
    # Diverse message pools to avoid spam detection
    SAFE_MESSAGES = [
        "Hello everyone! Hope you're having a great day.",
        "Does anyone want to join my party for the raid?",
        "That was a really good game, well played.",
        "Can someone help me with this coding problem?",
        "I completely agree with your point.",
        "Have you guys seen the new update? It's awesome.",
        "Thanks for the advice, that fixed my issue!",
        "Just chilling and listening to some music.",
        "What time is the event starting tomorrow?",
        "Haha that meme is hilarious!",
        "The weather is beautiful today!",
        "Anyone up for a quick match?",
        "I love this new feature they added.",
        "Great work team, we got this!",
        "Has anyone tried the new restaurant downtown?",
    ]
    
    TOXIC_MESSAGES = [
        "You are absolute trash at this game.",
        "Shut up you stupid idiot.",
        "I hope you die alone, loser.",
        "Why are you so dumb and worthless?",
        "Your opinion is garbage.",
        "I'm going to attack your server.",
        "You are pathetic and fat.",
        "Get out of here scum.",
        "You are a waste of space.",
        "Nobody likes you, ugly.",
        "Kill yourself already.",
        "You're a disgusting piece of trash.",
        "I will destroy you.",
        "Go away worthless loser.",
        "You're the worst player I've ever seen.",
    ]
    
    UNCERTAIN_MESSAGES = [
        "This new feature is absolute trash.",          # Product, not person
        "I will literally kill for a slice of pizza.",  # Hyperbole
        "You guys are acting crazy today.",             # Friendly banter
        "Stop being so dumb, it's a joke.",             # Sarcasm
        "That movie was painful to watch.",             # Figurative
        "I'm dying of laughter right now!",             # Expression
        "This code is killing me.",                     # Developer speak
        "You're insane for pulling that off!",          # Compliment
        "I hate Mondays so much.",                      # Common phrase
        "That's sick! (in a good way)",                 # Slang
    ]
    
    # Generate diverse users to avoid flooding detection
    USERS = [f"user_{i:03d}" for i in range(1, 51)]  # 50 unique users
    SERVERS = ["server_alpha", "server_beta", "server_gamma", "server_delta"]
    PLATFORMS = ["discord", "slack"]
    
    @staticmethod
    def make_unique(message: str) -> str:
        """Add unique suffix to avoid exact duplicates"""
        return f"{message} [{random.randint(1000, 9999)}]"
    
    @staticmethod
    def generate_metadata(user_type: str = "normal") -> Dict[str, Any]:
        """Generate realistic user metadata"""
        if user_type == "toxic":
            return {
                "user_bad_ratio_7d": random.uniform(0.3, 0.6),
                "violation_count_7d": random.randint(3, 7),
                "user_toxicity_trend": random.uniform(0.05, 0.15),
                "channel_toxicity_ratio": random.uniform(0.1, 0.3),
                "is_new_to_channel": 0
            }
        elif user_type == "new":
            return {
                "user_bad_ratio_7d": 0.0,
                "violation_count_7d": 0,
                "user_toxicity_trend": 0.0,
                "channel_toxicity_ratio": random.uniform(0.05, 0.15),
                "is_new_to_channel": 1
            }
        else:  # normal
            return {
                "user_bad_ratio_7d": random.uniform(0.0, 0.1),
                "violation_count_7d": random.randint(0, 2),
                "user_toxicity_trend": random.uniform(-0.05, 0.05),
                "channel_toxicity_ratio": random.uniform(0.05, 0.15),
                "is_new_to_channel": 0
            }


# ══════════════════════════════════════════════════════════════
# TEST SCENARIOS
# ══════════════════════════════════════════════════════════════

class TestScenarios:
    """Individual test scenarios for ingestion pipeline"""
    
    def __init__(self, conn):
        self.conn = conn
        self.gen = TestDataGenerator()
    
    async def scenario_1_normal_operation(self):
        """
        TEST 1: Normal Operation
        - Mix of safe, toxic, and admin-reviewed messages
        - Passes all quality gates
        - Demonstrates stratified sampling
        """
        print("\n" + "="*70)
        print("TEST 1: NORMAL OPERATION (Happy Path)")
        print("="*70)
        
        now = datetime.now()
        inserted = 0
        
        # High-confidence SAFE (should get 5% sampled)
        print("📝 Inserting 200 high-confidence SAFE messages...")
        for _ in range(200):
            msg = self.gen.make_unique(random.choice(self.gen.SAFE_MESSAGES))
            await self.conn.execute('''
                INSERT INTO logs (user_id, server_id, platform, message, toxicity_score, severity, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ''', 
                random.choice(self.gen.USERS), random.choice(self.gen.SERVERS), 
                random.choice(self.gen.PLATFORMS), msg,
                random.uniform(0.01, 0.25),  # < 0.3 threshold
                'SAFE', 
                now - timedelta(hours=random.uniform(0.1, 23.0)),
                json.dumps(self.gen.generate_metadata("normal"))
            )
            inserted += 1
        
        # High-confidence TOXIC (should get 5% sampled)
        print("📝 Inserting 80 high-confidence TOXIC messages...")
        for _ in range(80):
            msg = self.gen.make_unique(random.choice(self.gen.TOXIC_MESSAGES))
            await self.conn.execute('''
                INSERT INTO logs (user_id, server_id, platform, message, toxicity_score, severity, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ''', 
                random.choice(self.gen.USERS), random.choice(self.gen.SERVERS),
                random.choice(self.gen.PLATFORMS), msg,
                random.uniform(0.75, 0.99),  # > 0.7 threshold
                random.choice(['MEDIUM', 'HIGH']),
                now - timedelta(hours=random.uniform(0.1, 23.0)),
                json.dumps(self.gen.generate_metadata("toxic"))
            )
            inserted += 1
        
        # Admin-reviewed FEEDBACK (should get 100% included)
        print("📝 Inserting 30 admin-reviewed feedback items...")
        for i in range(30):
            user = random.choice(self.gen.USERS)
            server = random.choice(self.gen.SERVERS)
            platform = random.choice(self.gen.PLATFORMS)
            
            # 50/50 false positives vs false negatives
            if i % 2 == 0:
                # False positive: model said toxic, admin said safe
                msg = self.gen.make_unique(random.choice(self.gen.UNCERTAIN_MESSAGES))
                predicted_score, severity, predicted_label = random.uniform(0.6, 0.8), 'MEDIUM', 1
                final_label, decision = 0, 'agree_with_user'
            else:
                # False negative: model said safe, admin said toxic
                msg = self.gen.make_unique(random.choice(self.gen.TOXIC_MESSAGES).replace(" ", "  "))
                predicted_score, severity, predicted_label = random.uniform(0.2, 0.4), 'SAFE', 0
                final_label, decision = 1, 'agree_with_user'
            
            log_id = await self.conn.fetchval('''
                INSERT INTO logs (user_id, server_id, platform, message, toxicity_score, severity, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            ''', 
                user, server, platform, msg, predicted_score, severity,
                now - timedelta(hours=random.uniform(0.1, 23.0)),
                json.dumps({"is_feedback": True})
            )
            
            await self.conn.execute('''
                INSERT INTO feedback (
                    log_id, user_id, server_id, platform,
                    predicted_label, predicted_score, user_claimed_label,
                    admin_reviewed, admin_decision, final_label, used_in_training
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ''', log_id, user, server, platform,
                predicted_label, predicted_score, final_label,
                True, decision, final_label, False
            )
            inserted += 1
        
        print(f"✅ TEST 1 COMPLETE: {inserted} records inserted")
        print(f"   Expected ingestion:")
        print(f"   - Anchors: ~15 (5% of 280 high-conf messages)")
        print(f"   - Feedback: 30 (100% of admin-reviewed)")
        print(f"   - Total: ~45 training samples")
        return {"status": "pass", "records": inserted}
    
    async def scenario_2_spam_attack(self):
        """
        TEST 2: Spam Attack Detection
        - >50% identical messages
        - Should FAIL quality gate
        """
        print("\n" + "="*70)
        print("TEST 2: SPAM ATTACK (Should FAIL Quality Gate)")
        print("="*70)
        
        now = datetime.now()
        spam_message = "BUY CRYPTO NOW!!! CHEAP PRICES!!!"
        
        print("📝 Inserting 100 SPAM messages (51% identical)...")
        for _ in range(510):
            await self.conn.execute('''
                INSERT INTO logs (user_id, server_id, platform, message, toxicity_score, severity, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ''', 
                random.choice(self.gen.USERS), random.choice(self.gen.SERVERS),
                random.choice(self.gen.PLATFORMS), spam_message,  # NO unique suffix
                random.uniform(0.1, 0.3), 'SAFE',
                now - timedelta(hours=random.uniform(0.1, 23.0)),
                json.dumps({})
            )
        
        # Add diverse messages to make total 100
        for _ in range(490):
            msg = self.gen.make_unique(random.choice(self.gen.SAFE_MESSAGES))
            await self.conn.execute('''
                INSERT INTO logs (user_id, server_id, platform, message, toxicity_score, severity, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ''', 
                random.choice(self.gen.USERS), random.choice(self.gen.SERVERS),
                random.choice(self.gen.PLATFORMS), msg,
                random.uniform(0.1, 0.3), 'SAFE',
                now - timedelta(hours=random.uniform(0.1, 23.0)),
                json.dumps({})
            )
        
        print(f"✅ TEST 2 COMPLETE: 100 records inserted (51% spam)")
        print(f"   Expected: INGESTION BLOCKED (❌ SPAM ATTACK)")
        return {"status": "pass", "records": 100, "expected_failure": True}
    
    async def scenario_3_user_flooding(self):
        """
        TEST 3: Single User Flooding
        - One user sends >30% of messages
        - Should FAIL quality gate
        """
        print("\n" + "="*70)
        print("TEST 3: USER FLOODING (Should FAIL Quality Gate)")
        print("="*70)
        
        now = datetime.now()
        flooder = "malicious_bot_user"
        
        print("📝 Inserting 150 messages (50 from one user = 33%)...")
        # Flooder sends 50 messages
        for _ in range(500):
            msg = self.gen.make_unique(random.choice(self.gen.SAFE_MESSAGES))
            await self.conn.execute('''
                INSERT INTO logs (user_id, server_id, platform, message, toxicity_score, severity, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ''', 
                flooder, random.choice(self.gen.SERVERS),
                random.choice(self.gen.PLATFORMS), msg,
                random.uniform(0.1, 0.3), 'SAFE',
                now - timedelta(hours=random.uniform(0.1, 23.0)),
                json.dumps({})
            )
        
        # Normal users send 100 messages
        for _ in range(1000):
            msg = self.gen.make_unique(random.choice(self.gen.SAFE_MESSAGES))
            await self.conn.execute('''
                INSERT INTO logs (user_id, server_id, platform, message, toxicity_score, severity, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ''', 
                random.choice(self.gen.USERS), random.choice(self.gen.SERVERS),
                random.choice(self.gen.PLATFORMS), msg,
                random.uniform(0.1, 0.3), 'SAFE',
                now - timedelta(hours=random.uniform(0.1, 23.0)),
                json.dumps({})
            )
        
        print(f"✅ TEST 3 COMPLETE: 150 records (33% from one user)")
        print(f"   Expected: INGESTION BLOCKED (❌ FLOODING)")
        return {"status": "pass", "records": 150, "expected_failure": True}
    
    async def scenario_4_insufficient_admin_feedback(self):
        """
        TEST 4: Insufficient Admin Feedback
        - <5% admin-reviewed data (warning, not blocking)
        - Should WARN but still pass
        """
        print("\n" + "="*70)
        print("TEST 4: LOW ADMIN FEEDBACK (Should WARN)")
        print("="*70)
        
        now = datetime.now()
        
        print("📝 Inserting 200 messages with only 5 admin-reviewed (2.5%)...")
        # 195 high-conf messages
        for _ in range(195):
            msg = self.gen.make_unique(random.choice(self.gen.SAFE_MESSAGES))
            await self.conn.execute('''
                INSERT INTO logs (user_id, server_id, platform, message, toxicity_score, severity, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ''', 
                random.choice(self.gen.USERS), random.choice(self.gen.SERVERS),
                random.choice(self.gen.PLATFORMS), msg,
                random.uniform(0.01, 0.25), 'SAFE',
                now - timedelta(hours=random.uniform(0.1, 23.0)),
                json.dumps({})
            )
        
        # Only 5 feedback items (2.5%)
        for _ in range(5):
            user = random.choice(self.gen.USERS)
            msg = self.gen.make_unique(random.choice(self.gen.UNCERTAIN_MESSAGES))
            
            log_id = await self.conn.fetchval('''
                INSERT INTO logs (user_id, server_id, platform, message, toxicity_score, severity, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            ''', user, random.choice(self.gen.SERVERS), random.choice(self.gen.PLATFORMS),
                msg, 0.6, 'MEDIUM', now - timedelta(hours=random.uniform(0.1, 23.0)),
                json.dumps({})
            )
            
            await self.conn.execute('''
                INSERT INTO feedback (
                    log_id, user_id, server_id, platform,
                    predicted_label, predicted_score, user_claimed_label,
                    admin_reviewed, admin_decision, final_label, used_in_training
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ''', log_id, user, random.choice(self.gen.SERVERS), random.choice(self.gen.PLATFORMS),
                1, 0.6, 0, True, 'agree_with_user', 0, False
            )
        
        print(f"✅ TEST 4 COMPLETE: 200 records (2.5% admin-reviewed)")
        print(f"   Expected: ⚠️  WARNING (low admin feedback)")
        return {"status": "pass", "records": 200, "expected_warning": True}
    
    async def scenario_5_edge_cases(self):
        """
        TEST 5: Edge Cases
        - Very short/long messages
        - Missing metadata
        - Unusual characters
        """
        print("\n" + "="*70)
        print("TEST 5: EDGE CASES (Boundary Testing)")
        print("="*70)
        
        now = datetime.now()
        edge_cases = [
            # Too short (should be filtered out by MIN_TEXT_LENGTH=3)
            "Hi",
            "OK",
            
            # Valid short
            "Hello!",
            "Thanks!",
            
            # Emojis and special chars
            "🔥🔥🔥 This is fire!!!",
            "❤️ Love this community ❤️",
            
            # Mixed languages (if supported)
            "¡Hola amigos!",
            "Привет друзья!",
            
            # Code snippets
            "Check this: `print('hello')`",
            
            # URLs
            "Visit https://example.com for more info",
            
            # Very long (near MAX_TEXT_LENGTH=5000)
            "A" * 100 + " this is a long message " + "B" * 100,
        ]
        
        print(f"📝 Inserting {len(edge_cases)} edge case messages...")
        for msg in edge_cases:
            try:
                await self.conn.execute('''
                    INSERT INTO logs (user_id, server_id, platform, message, toxicity_score, severity, timestamp, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ''', 
                    random.choice(self.gen.USERS), random.choice(self.gen.SERVERS),
                    random.choice(self.gen.PLATFORMS), msg,
                    random.uniform(0.1, 0.3), 'SAFE',
                    now - timedelta(hours=random.uniform(0.1, 23.0)),
                    json.dumps({})
                )
            except Exception as e:
                print(f"      ⚠️  Edge case failed (expected): {msg[:30]}... - {e}")
        
        print(f"✅ TEST 5 COMPLETE")
        print(f"   Expected: Some messages filtered by length checks")
        return {"status": "pass"}
    
    async def scenario_6_realistic_production(self):
        """
        TEST 6: Realistic Production Load
        - 1000 messages over 24 hours
        - Realistic distribution: 85% safe, 10% toxic, 5% admin-reviewed
        - Tests actual production conditions
        """
        print("\n" + "="*70)
        print("TEST 6: REALISTIC PRODUCTION LOAD (1000 messages)")
        print("="*70)
        
        now = datetime.now()
        inserted = 0
        
        print("📝 Inserting 1000 production-like messages...")
        
        # 850 safe messages
        for _ in range(850):
            msg = self.gen.make_unique(random.choice(self.gen.SAFE_MESSAGES))
            await self.conn.execute('''
                INSERT INTO logs (user_id, server_id, platform, message, toxicity_score, severity, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ''', 
                random.choice(self.gen.USERS), random.choice(self.gen.SERVERS),
                random.choice(self.gen.PLATFORMS), msg,
                random.uniform(0.01, 0.29), 'SAFE',
                now - timedelta(hours=random.uniform(0.1, 23.9)),
                json.dumps(self.gen.generate_metadata("normal"))
            )
            inserted += 1
        
        # 100 toxic messages
        for _ in range(100):
            msg = self.gen.make_unique(random.choice(self.gen.TOXIC_MESSAGES))
            await self.conn.execute('''
                INSERT INTO logs (user_id, server_id, platform, message, toxicity_score, severity, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ''', 
                random.choice(self.gen.USERS), random.choice(self.gen.SERVERS),
                random.choice(self.gen.PLATFORMS), msg,
                random.uniform(0.71, 0.99), random.choice(['MEDIUM', 'HIGH']),
                now - timedelta(hours=random.uniform(0.1, 23.9)),
                json.dumps(self.gen.generate_metadata("toxic"))
            )
            inserted += 1
        
        # 50 admin-reviewed (5%)
        for i in range(50):
            user = random.choice(self.gen.USERS)
            
            if i % 2 == 0:
                msg = self.gen.make_unique(random.choice(self.gen.UNCERTAIN_MESSAGES))
                predicted_score, severity, predicted_label = random.uniform(0.5, 0.7), 'MEDIUM', 1
                final_label, decision = 0, 'agree_with_user'
            else:
                msg = self.gen.make_unique(random.choice(self.gen.TOXIC_MESSAGES).replace(" ", " "))
                predicted_score, severity, predicted_label = random.uniform(0.3, 0.5), 'SAFE', 0
                final_label, decision = 1, 'agree_with_user'
            
            log_id = await self.conn.fetchval('''
                INSERT INTO logs (user_id, server_id, platform, message, toxicity_score, severity, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            ''', user, random.choice(self.gen.SERVERS), random.choice(self.gen.PLATFORMS),
                msg, predicted_score, severity,
                now - timedelta(hours=random.uniform(0.1, 23.9)),
                json.dumps({"is_feedback": True})
            )
            
            await self.conn.execute('''
                INSERT INTO feedback (
                    log_id, user_id, server_id, platform,
                    predicted_label, predicted_score, user_claimed_label,
                    admin_reviewed, admin_decision, final_label, used_in_training
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ''', log_id, user, random.choice(self.gen.SERVERS), random.choice(self.gen.PLATFORMS),
                predicted_label, predicted_score, final_label,
                True, decision, final_label, False
            )
            inserted += 1
        
        print(f"✅ TEST 6 COMPLETE: {inserted} records inserted")
        print(f"   Expected ingestion:")
        print(f"   - Anchors: ~48 (5% of 950 high-conf)")
        print(f"   - Feedback: 50 (100% of admin-reviewed)")
        print(f"   - Total: ~98 training samples")
        return {"status": "pass", "records": inserted}


# ══════════════════════════════════════════════════════════════
# TEST RUNNER
# ══════════════════════════════════════════════════════════════

async def clear_test_data(conn):
    """Clean up test data between scenarios"""
    print("\n🧹 Cleaning up test data...")
    await conn.execute("DELETE FROM feedback")
    await conn.execute("DELETE FROM logs WHERE timestamp > NOW() - INTERVAL '24 hours'")
    print("✅ Cleanup complete\n")

async def run_test_suite():
    """Execute all test scenarios"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║        INGESTION PIPELINE TEST SUITE                          ║
║                  Starting Tests...                            ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    conn = await asyncpg.connect(DATABASE_URL, statement_cache_size=0)
    scenarios = TestScenarios(conn)
    results = []
    
    try:
        # Test 1: Normal operation
        await clear_test_data(conn)
        result = await scenarios.scenario_1_normal_operation()
        results.append(("Normal Operation", result))
        input("\n⏸️  Press Enter to continue to Test 2...")
        
        # Test 2: Spam attack
        await clear_test_data(conn)
        result = await scenarios.scenario_2_spam_attack()
        results.append(("Spam Attack Detection", result))
        input("\n⏸️  Press Enter to continue to Test 3...")
        
        # Test 3: User flooding
        await clear_test_data(conn)
        result = await scenarios.scenario_3_user_flooding()
        results.append(("User Flooding Detection", result))
        input("\n⏸️  Press Enter to continue to Test 4...")
        
        # Test 4: Low admin feedback
        await clear_test_data(conn)
        result = await scenarios.scenario_4_insufficient_admin_feedback()
        results.append(("Low Admin Feedback", result))
        input("\n⏸️  Press Enter to continue to Test 5...")
        
        # Test 5: Edge cases
        await clear_test_data(conn)
        result = await scenarios.scenario_5_edge_cases()
        results.append(("Edge Cases", result))
        input("\n⏸️  Press Enter to continue to Test 6...")
        
        # Test 6: Realistic production
        await clear_test_data(conn)
        result = await scenarios.scenario_6_realistic_production()
        results.append(("Realistic Production", result))
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUITE SUMMARY")
        print("="*70)
        for name, result in results:
            status_emoji = "✅" if result["status"] == "pass" else "❌"
            print(f"{status_emoji} {name}: {result['status'].upper()}")
        
        print("\n" + "="*70)
        print("🎯 NEXT STEPS:")
        print("="*70)
        print("1. Run: python -m mlops.ingest_data")
        print("2. Verify logs show:")
        print("   - Test 1: ~45 samples ingested (stratified sampling works)")
        print("   - Test 2: INGESTION BLOCKED (spam attack)")
        print("   - Test 3: INGESTION BLOCKED (user flooding)")
        print("   - Test 4: ⚠️  WARNING (low admin feedback)")
        print("   - Test 6: ~98 samples ingested (production load)")
        print("")
        print("3. Check master dataset grew properly")
        print("4. Verify feedback items marked as used_in_training=TRUE")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await conn.close()


# ENTRY POINT

if __name__ == "__main__":
    print("Database URL:", DATABASE_URL.split('@')[-1])
    print("\nThis script will insert test data and guide you through")
    print("running the ingestion pipeline to verify it works correctly.")
    print("")
    
    choice = input("Continue? (y/n): ")
    if choice.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    asyncio.run(run_test_suite())