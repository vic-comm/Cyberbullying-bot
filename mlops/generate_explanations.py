import sqlite3
import json
from api_service.explainer import ToxicityExplainer

def generate_batch_explanations():
    # Connect to the Bot's database
    conn = sqlite3.connect('bot_memory.db')
    cursor = conn.cursor()
    
    # 1. Find "Missed" Explanations
    # Select messages that were punished (ACTION NOT NULL) 
    # but still have no explanation (metadata is empty or missing 'trigger_words')
    cursor.execute("""
        SELECT id, message 
        FROM logs 
        WHERE severity IN ('LOW', 'MEDIUM', 'HIGH') 
        AND (metadata IS NULL OR metadata NOT LIKE '%trigger_words%')
        LIMIT 500
    """)
    rows = cursor.fetchall()

    # 2. Initialize Explainer (Heavy Load)
    # We load the model here locally because this is a batch script, not the live bot.
    explainer = ToxicityExplainer(load_model())

    for row_id, text in rows:
        # Generate the explanation (Heavy calculation)
        result = explainer.explain(text)
        
        # 3. Patch the Database
        # We update the JSON metadata column with the new explanation
        cursor.execute("""
            UPDATE logs 
            SET metadata = json_patch(ifnull(metadata, '{}'), ?)
            WHERE id = ?
        """, (json.dumps({"explanation": result}), row_id))

    conn.commit()
    print("Batch job complete.")