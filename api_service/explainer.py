import hashlib
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
import logging

logger = logging.getLogger(__name__)

class ToxicityExplainer:
    # Define which features are calculated from text (will change with LIME perturbations)
    TEXT_DERIVED_FEATURES = {'msg_len', 'caps_ratio', 'personal_pronoun_count', 'slur_count'}
    
    def __init__(self, model_pipeline, feature_calculator, model_tabular_features, model_int_features):
        self.pipeline = model_pipeline
        self.feature_calculator = feature_calculator
        self.model_tabular_features = model_tabular_features
        self.model_int_features = model_int_features
        
        # Initialize LIME
        self.explainer = LimeTextExplainer(
            class_names=['safe', 'toxic'],
            split_expression=r'\W+',
            bow=False
        )
        self._cache = {}

    def explain(self, text: str, all_features: dict, num_features: int = 6):
        """
        Generate LIME explanation for text toxicity.
        
        Args:
            text: The message to explain
            all_features: Dictionary with ALL features (text + user + channel)
            num_features: Number of top words to include in explanation
            
        Returns:
            Dictionary with trigger words and context information
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._cache:
            logger.debug(f"Cache hit for text_hash: {text_hash}")
            return self._cache[text_hash]

        # ---------------------------------------------------------
        # 1. DEFINE PREDICTOR INSIDE (Thread-safe closure)
        # ---------------------------------------------------------
        def custom_predictor(texts):
            """
            Predictor for LIME that:
            - Recalculates text-derived features for each variation
            - Keeps user/channel context features frozen
            """
            rows = []
            for t in texts:
                # A. Recalculate dynamic text features (these change as words are removed)
                dynamic_features = self.feature_calculator(t)
                
                # B. Start building the row
                row = {'text': t}
                row.update(dynamic_features)
                
                # C. Add frozen context features (user history, channel stats)
                # These stay CONSTANT across all LIME perturbations
                for feature in self.model_tabular_features:
                    if feature not in row:  # Don't override dynamic features
                        row[feature] = all_features.get(feature, 0)
                
                rows.append(row)
            
            # D. Create batch DataFrame (much faster than individual predictions)
            input_df = pd.DataFrame(rows)
            
            # E. Enforce data types (crucial for XGBoost/sklearn)
            for feature in self.model_tabular_features:
                if feature in input_df.columns:
                    if feature in self.model_int_features:
                        input_df[feature] = input_df[feature].astype('int64')
                    else:
                        input_df[feature] = input_df[feature].astype('float64')

            # F. Batch prediction
            try:
                _, confidences = self.pipeline.predict(input_df)
            except Exception as e:
                logger.error(f"Prediction failed in explainer: {e}")
                # Return neutral probabilities on error
                return np.array([[0.5, 0.5]] * len(texts))
            
            # G. Format for LIME: [[safe_prob, toxic_prob], ...]
            probas = []
            for score in confidences:
                score = float(score)
                # Assuming score is probability of toxic class
                probas.append([1.0 - score, score])
                
            return np.array(probas)

        # ---------------------------------------------------------
        # 2. RUN LIME
        # ---------------------------------------------------------
        try:
            exp = self.explainer.explain_instance(
                text, 
                custom_predictor,
                num_features=num_features, 
                num_samples=500
            )
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {
                "error": "Explanation generation failed",
                "text_hash": text_hash,
                "trigger_words": [],
                "toxic_probability": 0.0
            }

        # ---------------------------------------------------------
        # 3. FORMAT OUTPUT
        # ---------------------------------------------------------
        trigger_words = [
            {
                "word": w, 
                "score": round(float(s), 4), 
                "category": "toxic" if s > 0 else "safe"
            } 
            for w, s in exp.as_list(label=1)  # Label 1 = toxic
        ]
        
        result = {
            "text_hash": text_hash,
            "trigger_words": trigger_words,
            "toxic_probability": round(float(exp.predict_proba[1]), 4),
            "features_used": {
                "text_derived": {
                    k: all_features.get(k, 0) 
                    for k in self.TEXT_DERIVED_FEATURES 
                    if k in all_features
                },
                "context_frozen": {
                    k: v for k, v in all_features.items() 
                    if k not in self.TEXT_DERIVED_FEATURES and k != 'text'
                }
            }
        }
        
        self._cache[text_hash] = result
        logger.info(f"Generated explanation for text_hash: {text_hash}")
        return result