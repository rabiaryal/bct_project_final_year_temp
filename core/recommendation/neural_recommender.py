"""
Neural Dual-Encoder Recommendation System
Integrates trained neural networks with the production chatbot system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
try:
    from tensorflow import keras
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import json

class NeuralRecommender:
    """
    Neural dual-encoder recommendation system
    Uses trained query and item encoders for fast similarity computation
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.query_encoder = None
        self.item_encoder = None
        self.scaler = None
        self.encoder = None
        self.item_features = []
        self.query_input_cols = []
        self.cat_feature_names = []
        self.item_embeddings = None
        
        if TENSORFLOW_AVAILABLE:
            self._prepare_data()
            self._load_or_train_models()
        else:
            print("⚠️ TensorFlow not available. Neural recommendations disabled.")
    
    def _prepare_data(self):
        """Prepare data for neural network training/inference"""
        print("🔧 Preparing neural network data...")
        
        # Preprocessing for neural networks
        NUM_COLS = ['Fee', 'TotalSeats', 'PassPercentage', 'Rating']
        self.scaler = MinMaxScaler()
        
        # Create a copy for processing
        df_processed = self.df.copy()
        df_processed[NUM_COLS] = self.scaler.fit_transform(df_processed[NUM_COLS])
        
        CAT_COLS = ['Location', 'Type', 'CourseName']
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        cat_encoded = self.encoder.fit_transform(df_processed[CAT_COLS])
        self.cat_feature_names = self.encoder.get_feature_names_out(CAT_COLS)
        
        df_cat_encoded = pd.DataFrame(cat_encoded, columns=self.cat_feature_names)
        df_processed = pd.concat([df_processed.reset_index(drop=True), df_cat_encoded], axis=1)
        
        self.item_features = [col for col in df_processed.columns if col in NUM_COLS or col in self.cat_feature_names]
        self.query_input_cols = ['query_fee_norm'] + [col + '_query' for col in self.cat_feature_names]
        
        # Store processed dataframe
        self.df_processed = df_processed
        
        print(f"✓ Neural data prepared: {len(self.item_features)} item features, {len(self.query_input_cols)} query features")
    
    def _load_or_train_models(self):
        """Load existing models or train new ones"""
        try:
            # Always train fresh models for compatibility
            print("🏗️ Training fresh neural encoders for best compatibility...")
            self._train_models()
            
        except Exception as e:
            print(f"❌ Neural training failed: {e}")
            self.query_encoder = None
            self.item_encoder = None
    
    def _train_models(self):
        """Train neural dual-encoder models"""
        if not TENSORFLOW_AVAILABLE:
            return
            
        try:
            # Generate training data
            training_data = self._generate_training_data()
            
            if len(training_data) == 0:
                print("❌ No training data generated. Skipping neural training.")
                return
            
            # Prepare training arrays
            query_data, item_data, labels = self._prepare_training_arrays(training_data)
            
            # Build models
            query_dim = len(self.query_input_cols)
            item_dim = len(self.item_features)
            
            self.query_encoder, self.item_encoder, training_model = self._build_dual_encoder(
                query_dim, item_dim, embedding_dim=64
            )
            
            # Train the model
            print("🏋️ Training neural encoders...")
            history = training_model.fit(
                [query_data, item_data], labels,
                epochs=3,  # Reduced epochs for faster training
                batch_size=32,
                validation_split=0.2,
                verbose=0  # Silent training
            )
            
            # Don't save models for compatibility
            print("✅ Neural encoders trained successfully")
            
            # Precompute item embeddings
            self._precompute_item_embeddings()
            
        except Exception as e:
            print(f"❌ Neural training failed: {e}")
            self.query_encoder = None
            self.item_encoder = None
    
    def _generate_training_data(self):
        """Generate positive and negative training pairs"""
        import random
        
        training_data = []
        
        for idx, item in self.df_processed.iterrows():
            # Positive pair
            positive_pair = {
                'query_fee_norm': item['Fee'],  # Normalized fee
                'item_index': idx,
                'label': 1.0
            }
            
            # Add categorical query features
            for col in self.cat_feature_names:
                positive_pair[f'{col}_query'] = item[col]
            
            training_data.append(positive_pair)
            
            # Negative pairs
            negative_indices = random.sample(range(len(self.df_processed)), min(3, len(self.df_processed)))
            for neg_idx in negative_indices:
                if neg_idx != idx:
                    negative_pair = {
                        'query_fee_norm': item['Fee'],
                        'item_index': neg_idx,
                        'label': 0.0
                    }
                    for col in self.cat_feature_names:
                        negative_pair[f'{col}_query'] = item[col]
                    training_data.append(negative_pair)
        
        return pd.DataFrame(training_data)
    
    def _prepare_training_arrays(self, df_train):
        """Convert training data to numpy arrays"""
        item_indices = df_train['item_index'].values
        item_data = self.df_processed.loc[item_indices, self.item_features].values
        
        query_cols = [col for col in df_train.columns if col in self.query_input_cols]
        query_data = df_train[query_cols].values
        labels = df_train['label'].values
        
        return query_data, item_data, labels
    
    def _build_dual_encoder(self, query_dim, item_dim, embedding_dim=64):
        """Build dual-encoder neural network"""
        print("🏗️ Building neural dual-encoder...")
        
        # Query encoder
        query_input = keras.Input(shape=(query_dim,), name="query_features")
        x = keras.layers.Dense(128, activation="relu")(query_input)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        query_embedding = keras.layers.Dense(embedding_dim, name="query_embedding")(x)
        query_encoder = keras.Model(query_input, query_embedding)
        
        # Item encoder
        item_input = keras.Input(shape=(item_dim,), name="item_features")
        y = keras.layers.Dense(128, activation="relu")(item_input)
        y = keras.layers.Dropout(0.3)(y)
        y = keras.layers.Dense(64, activation="relu")(y)
        item_embedding = keras.layers.Dense(embedding_dim, name="item_embedding")(y)
        item_encoder = keras.Model(item_input, item_embedding)
        
        # Training model
        similarity = keras.layers.Dot(axes=1, normalize=True)([query_embedding, item_embedding])
        training_model = keras.Model([query_input, item_input], similarity)
        
        training_model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        
        print("✓ Neural model built")
        return query_encoder, item_encoder, training_model
    
    def _precompute_item_embeddings(self):
        """Precompute embeddings for all items for fast inference"""
        if self.item_encoder is None:
            return
        
        print("🔄 Precomputing item embeddings...")
        item_data = self.df_processed[self.item_features].values
        self.item_embeddings = self.item_encoder.predict(item_data, verbose=0)
        print(f"✅ Precomputed {len(self.item_embeddings)} item embeddings")
    
    def _parse_query_to_features(self, prompt: str, intent: str, entities: Dict) -> Optional[np.ndarray]:
        """Convert natural language query to neural network features"""
        try:
            # Initialize query features
            query_features = {}
            
            # Fee preference (normalized)
            fee_preference = 0.5  # Default to middle range
            
            # Extract fee preferences from prompt
            if 'cheap' in prompt.lower() or 'affordable' in prompt.lower() or 'low cost' in prompt.lower():
                fee_preference = 0.2  # Low fee preference
            elif 'expensive' in prompt.lower() or 'premium' in prompt.lower() or 'high end' in prompt.lower():
                fee_preference = 0.8  # High fee preference
            elif 'moderate' in prompt.lower() or 'average' in prompt.lower():
                fee_preference = 0.5  # Medium fee preference
            
            query_features['query_fee_norm'] = fee_preference
            
            # Location preferences
            for col in self.cat_feature_names:
                query_features[f'{col}_query'] = 0.0
            
            # Set location preference
            if entities.get('location_mentioned'):
                location = entities['location_mentioned']
                location_col = f'Location_{location}'
                if location_col in self.cat_feature_names:
                    query_features[f'{location_col}_query'] = 1.0
            elif 'kathmandu' in prompt.lower():
                if 'Location_Kathmandu' in self.cat_feature_names:
                    query_features['Location_Kathmandu_query'] = 1.0
            elif 'pokhara' in prompt.lower():
                if 'Location_Pokhara' in self.cat_feature_names:
                    query_features['Location_Pokhara_query'] = 1.0
            
            # Course preferences
            if entities.get('course_mentioned'):
                course = entities['course_mentioned']
                course_col = f'CourseName_{course}'
                if course_col in self.cat_feature_names:
                    query_features[f'{course_col}_query'] = 1.0
            elif 'engineering' in prompt.lower():
                for col in self.cat_feature_names:
                    if 'CourseName' in col and 'Engineering' in col:
                        query_features[f'{col}_query'] = 1.0
                        break
            elif 'bba' in prompt.lower() or 'business' in prompt.lower():
                for col in self.cat_feature_names:
                    if 'CourseName' in col and 'BBA' in col:
                        query_features[f'{col}_query'] = 1.0
                        break
            
            # Type preferences
            if 'private' in prompt.lower():
                if 'Type_Private' in self.cat_feature_names:
                    query_features['Type_Private_query'] = 1.0
            elif 'public' in prompt.lower() or 'government' in prompt.lower():
                if 'Type_Public' in self.cat_feature_names:
                    query_features['Type_Public_query'] = 1.0
            
            # Convert to array in correct order
            query_array = np.array([query_features.get(col, 0.0) for col in self.query_input_cols])
            
            return query_array.reshape(1, -1)
        
        except Exception as e:
            print(f"❌ Query parsing error: {e}")
            return None
    
    def rank_colleges(self, prompt: str, intent: str, entities: Dict, k: int = 5) -> List[Dict]:
        """
        Use neural encoders to rank colleges based on query
        """
        if not TENSORFLOW_AVAILABLE or self.query_encoder is None or self.item_embeddings is None:
            print("⚠️ Neural encoders not available, falling back to basic ranking")
            return self._fallback_ranking(prompt, k)
        
        try:
            print(f"🧠 Using neural dual-encoder for recommendations")
            
            # Parse query to neural features
            query_features = self._parse_query_to_features(prompt, intent, entities)
            
            if query_features is None:
                return self._fallback_ranking(prompt, k)
            
            # Get query embedding
            query_embedding = self.query_encoder.predict(query_features, verbose=0)
            
            # Calculate similarities with all items
            similarities = np.dot(self.item_embeddings, query_embedding.T).flatten()
            
            # Get top k items
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for i, idx in enumerate(top_indices):
                college_data = self.df.iloc[idx]
                similarity_score = similarities[idx]
                
                result = {
                    'college_data': college_data.to_dict(),
                    'neural_similarity': float(similarity_score),
                    'rank': i + 1,
                    'ranking_explanation': f"Neural similarity: {similarity_score:.3f}",
                    'weights_used': {'neural': 1.0}
                }
                results.append(result)
            
            print(f"✅ Neural ranking completed: {len(results)} colleges ranked")
            return results
        
        except Exception as e:
            print(f"❌ Neural ranking failed: {e}")
            return self._fallback_ranking(prompt, k)
    
    def _fallback_ranking(self, prompt: str, k: int) -> List[Dict]:
        """Simple fallback ranking when neural methods fail"""
        # Simple ranking by rating
        df_sorted = self.df.sort_values('Rating', ascending=False)
        
        results = []
        for i, (_, row) in enumerate(df_sorted.head(k).iterrows()):
            result = {
                'college_data': row.to_dict(),
                'neural_similarity': 0.0,
                'rank': i + 1,
                'ranking_explanation': f"Ranked by rating: {row['Rating']:.1f}/5.0",
                'weights_used': {'rating': 1.0}
            }
            results.append(result)
        
        return results

# Factory function for integration
def create_neural_recommender(df: pd.DataFrame) -> NeuralRecommender:
    """Create neural recommender instance"""
    return NeuralRecommender(df)