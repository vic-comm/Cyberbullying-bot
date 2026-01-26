import pandas as pd
import numpy as np
import torch
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import fbeta_score, classification_report, make_scorer
from transformers import DistilBertTokenizer, DistilBertModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt


DATA_PATH = "../data/training_data_with_history.parquet"
ARTIFACTS_DIR = "../api_service/artifacts"

# OPTIMIZED: Increased from 64 to 128 for better performance
SVD_COMPONENTS = 128  # Sweet spot: 91-94% variance, good F2-score
RANDOM_STATE = 42

# Features to use (Must match what generate_data.py created)
TABULAR_FEATURES = [
    'msg_len', 'caps_ratio', 'personal_pronoun_count', 'slur_count',
    'user_bad_ratio_7d', 'user_toxicity_trend',
    'channel_toxicity_ratio', 'hours_since_last_msg', 'is_new_to_channel'
]

# Ensure artifacts dir exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ==========================================
# 1. COMPONENT OPTIMIZATION (Optional)
# ==========================================
def analyze_components(vectors_768, component_range=[64, 96, 128, 160, 196]):
    """
    Analyze variance retention for different component counts
    """
    print("\n📊 Analyzing SVD Component Retention...")
    results = []
    
    for n_comp in component_range:
        svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
        svd.fit(vectors_768)
        variance = svd.explained_variance_ratio_.sum()
        results.append({'components': n_comp, 'variance': variance})
        print(f"   {n_comp} components → {variance*100:.2f}% variance retained")
    
    return pd.DataFrame(results)

# ==========================================
# 2. HELPER: BATCHED VECTORIZATION
# ==========================================
def get_bert_embeddings(text_list, batch_size=64):
    """
    Generates DistilBERT embeddings in batches to prevent Memory Errors.
    Returns a numpy array of shape (N, 768).
    """
    print("   -> Loading DistilBERT Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    model.eval()  # Set to evaluation mode
    
    all_embeddings = []
    total = len(text_list)
    
    print(f"   -> Vectorizing {total} messages on {device}...")
    
    for i in range(0, total, batch_size):
        batch_text = text_list[i : i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_text, 
            padding=True, 
            truncation=True, 
            max_length=128, # Short context for chat is usually enough
            return_tensors="pt"
        ).to(device)
        
        # Inference (No Gradient Calculation needed)
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Take [CLS] token (first token) as the sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
        
        if i % 1000 == 0 and i > 0:
            print(f"      Processed {i}/{total}...")
    
    # Clear CUDA cache to free memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
            
    return np.vstack(all_embeddings)

# ==========================================
# 3. THRESHOLD OPTIMIZATION
# ==========================================
def optimize_threshold(model, X_val, y_val, beta=2):
    """
    Find optimal classification threshold for F-beta score
    """
    print("\n🎯 Optimizing Classification Threshold...")
    
    # Get probability predictions
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Test thresholds
    thresholds = np.arange(0.1, 0.9, 0.01)
    f_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f_score = fbeta_score(y_val, y_pred, beta=beta)
        f_scores.append(f_score)
    
    # Find optimal
    optimal_idx = np.argmax(f_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f_score = f_scores[optimal_idx]
    
    print(f"   Optimal Threshold: {optimal_threshold:.3f}")
    print(f"   F{beta}-Score at Optimal: {optimal_f_score:.4f}")
    
    # Compare to default 0.5
    default_pred = (y_proba >= 0.5).astype(int)
    default_f_score = fbeta_score(y_val, default_pred, beta=beta)
    improvement = ((optimal_f_score - default_f_score) / default_f_score) * 100
    print(f"   Default (0.5) F{beta}-Score: {default_f_score:.4f}")
    print(f"   Improvement: +{improvement:.2f}%")
    
    return optimal_threshold, optimal_f_score

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def run_training(analyze_svd=False):
    print("1. Loading Data...")
    df = pd.read_parquet(DATA_PATH)
    
    # Optional: Sample data for faster debugging if dataset is huge
    # df = df.sample(50000, random_state=RANDOM_STATE)
    
    print(f"   Shape: {df.shape}")
    class_dist = df['label'].value_counts(normalize=True)
    print(f"   Class Balance:")
    print(f"      Safe (0): {class_dist[0]*100:.2f}%")
    print(f"      Toxic (1): {class_dist[1]*100:.2f}%")
    
    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = (df['label'] == 0).sum() / (df['label'] == 1).sum()
    print(f"   Recommended scale_pos_weight: {scale_pos_weight:.2f}")

    # --------------------------------------
    # A. PROCESS TEXT (DistilBERT + SVD)
    # --------------------------------------
    print("\n2. Text Processing...")
    
    # 2a. Vectorize
    raw_text = df['text'].astype(str).tolist()
    vectors_768 = get_bert_embeddings(raw_text)
    
    # 2b. Optional: Analyze components
    if analyze_svd:
        component_analysis = analyze_components(vectors_768)
        component_analysis.to_csv(f"{ARTIFACTS_DIR}/component_analysis.csv", index=False)
    
    # 2c. Dimensionality Reduction (SVD)
    print(f"   -> Reducing dimensions (768 -> {SVD_COMPONENTS}) with TruncatedSVD...")
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
    vectors_reduced = svd.fit_transform(vectors_768)
    
    variance_retained = svd.explained_variance_ratio_.sum()
    print(f"   -> Variance Retained: {variance_retained*100:.2f}%")
    
    if variance_retained < 0.90:
        print(f"   ⚠️  WARNING: Only {variance_retained*100:.2f}% variance retained.")
        print(f"      Consider increasing SVD_COMPONENTS to 128 or 160 for better performance.")
    
    # Save SVD transformer for production inference
    joblib.dump(svd, f"{ARTIFACTS_DIR}/svd_transformer.pkl")

    # --------------------------------------
    # B. PROCESS METADATA (Scaling)
    # --------------------------------------
    print("\n3. Tabular Processing...")
    tabular_data = df[TABULAR_FEATURES].values
    
    scaler = StandardScaler()
    tabular_scaled = scaler.fit_transform(tabular_data)
    
    # Save Scaler
    joblib.dump(scaler, f"{ARTIFACTS_DIR}/scaler.pkl")

    # --------------------------------------
    # C. COMBINE FEATURES
    # --------------------------------------
    print("\n4. Feature Fusion...")
    # Stack [SVD_Vectors + Scaled_Features]
    X = np.hstack([vectors_reduced, tabular_scaled])
    y = df['label'].values
    
    print(f"   Final Input Shape: {X.shape}")
    print(f"   - Text features (SVD): {vectors_reduced.shape[1]}")
    print(f"   - Behavioral features: {tabular_scaled.shape[1]}")

    # Split with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    # Further split temp into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"   Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # --------------------------------------
    # D. MODEL TOURNAMENT
    # --------------------------------------
    print("\n5. Model Training & Evaluation (Target: F2-Score)...")
    
    # Optimized hyperparameters based on your use case
    models = {
        "XGBoost": XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1
        )
    }
    
    best_score = 0
    best_model_name = ""
    best_model = None
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        
        # Train with early stopping for tree models
        if name in ["XGBoost", "LightGBM"]:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        # Predict on validation set
        preds = model.predict(X_val)
        
        # Calculate F2 Score (Weights Recall 2x higher than Precision)
        f2 = fbeta_score(y_val, preds, beta=2)
        print(f"   -> {name} F2 Score (Val): {f2:.4f}")
        print(classification_report(y_val, preds, target_names=['Safe', 'Toxic']))
        
        if f2 > best_score:
            best_score = f2
            best_model_name = name
            best_model = model

    # --------------------------------------
    # E. THRESHOLD OPTIMIZATION
    # --------------------------------------
    optimal_threshold, optimized_f2 = optimize_threshold(best_model, X_val, y_val, beta=2)
    
    # Save optimal threshold
    joblib.dump(optimal_threshold, f"{ARTIFACTS_DIR}/optimal_threshold.pkl")

    # --------------------------------------
    # F. FINAL TEST EVALUATION
    # --------------------------------------
    print(f"\n6. Final Test Set Evaluation...")
    print(f"   Using {best_model_name} with threshold={optimal_threshold:.3f}")
    
    # Test with optimal threshold
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
    
    test_f2 = fbeta_score(y_test, y_test_pred, beta=2)
    
    print(f"\n   🎯 Final Test F2-Score: {test_f2:.4f}")
    print("\n" + "="*60)
    print(classification_report(y_test, y_test_pred, target_names=['Safe', 'Toxic']))
    print("="*60)

    # --------------------------------------
    # G. SAVE ARTIFACTS
    # --------------------------------------
    print(f"\n7. Saving Model Artifacts...")
    print(f"   🏆 WINNER: {best_model_name} (F2: {test_f2:.4f})")
    
    # Save model
    joblib.dump(best_model, f"{ARTIFACTS_DIR}/model_v1.pkl")
    
    # Save metadata
    metadata = {
        'model_type': best_model_name,
        'svd_components': SVD_COMPONENTS,
        'variance_retained': float(variance_retained),
        'f2_score_val': float(best_score),
        'f2_score_test': float(test_f2),
        'optimal_threshold': float(optimal_threshold),
        'scale_pos_weight': float(scale_pos_weight),
        'feature_count': int(X.shape[1]),
        'training_samples': int(X_train.shape[0]),
        'tabular_features': TABULAR_FEATURES
    }
    
    import json
    with open(f"{ARTIFACTS_DIR}/model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Saved to {ARTIFACTS_DIR}/")
    print(f"   - model_v1.pkl")
    print(f"   - svd_transformer.pkl")
    print(f"   - scaler.pkl")
    print(f"   - optimal_threshold.pkl")
    print(f"   - model_metadata.json")
    print("\n✅ Training Pipeline Complete.")

if __name__ == "__main__":
    # Set analyze_svd=True to test different component counts
    run_training(analyze_svd=False)