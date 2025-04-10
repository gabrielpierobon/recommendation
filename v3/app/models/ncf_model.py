"""
Neural Collaborative Filtering (NCF) Model
Implements a deep learning based recommendation model that combines matrix factorization
with multi-layer perceptron for enhanced recommendation accuracy.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Flatten, Concatenate, Input, Multiply, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NCFRecommender:
    """Neural Collaborative Filtering based recommender that integrates with the main recommendation engine."""
    
    def __init__(self, recommendation_engine, embedding_size=32, mlp_layers=[64, 32, 16], dropout_rate=0.2):
        """Initialize NCF recommender.
        
        Args:
            recommendation_engine: Parent recommendation engine instance
            embedding_size (int): Size of embedding vectors
            mlp_layers (list): List of layer sizes for MLP
            dropout_rate (float): Dropout rate for regularization
        """
        self.engine = recommendation_engine
        self.embedding_size = embedding_size
        self.mlp_layers = mlp_layers
        self.dropout_rate = dropout_rate
        
        # Initialize encoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Fit encoders on all users and items
        self.user_encoder.fit(self.engine.users_df['user_id'])
        self.item_encoder.fit(self.engine.items_df['item_id'])
        
        # Get number of users and items
        self.num_users = len(self.user_encoder.classes_)
        self.num_items = len(self.item_encoder.classes_)
        
        # Build model
        self.model = self._build_model()
        
    def _build_model(self):
        """Build and compile the NCF model architecture."""
        # Input layers
        user_input = Input(shape=(1,), name='user_input')
        item_input = Input(shape=(1,), name='item_input')

        # GMF path
        gmf_user_embedding = Embedding(self.num_users, self.embedding_size, name='gmf_user_embedding')(user_input)
        gmf_item_embedding = Embedding(self.num_items, self.embedding_size, name='gmf_item_embedding')(item_input)
        
        gmf_user_latent = Flatten()(gmf_user_embedding)
        gmf_item_latent = Flatten()(gmf_item_embedding)
        gmf_vector = Multiply()([gmf_user_latent, gmf_item_latent])

        # MLP path
        mlp_user_embedding = Embedding(self.num_users, self.embedding_size, name='mlp_user_embedding')(user_input)
        mlp_item_embedding = Embedding(self.num_items, self.embedding_size, name='mlp_item_embedding')(item_input)
        
        mlp_user_latent = Flatten()(mlp_user_embedding)
        mlp_item_latent = Flatten()(mlp_item_embedding)
        mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])

        # Build MLP layers with dropout
        for i, layer_size in enumerate(self.mlp_layers):
            mlp_vector = Dense(layer_size, activation='relu', name=f'mlp_layer_{i}')(mlp_vector)
            mlp_vector = Dropout(self.dropout_rate, name=f'dropout_{i}')(mlp_vector)

        # Combine GMF and MLP paths
        predict_vector = Concatenate()([gmf_vector, mlp_vector])
        prediction = Dense(1, activation='sigmoid', name='prediction')(predict_vector)

        model = Model(inputs=[user_input, item_input], outputs=prediction)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        return model

    def train(self, validation_split=0.2, epochs=20, batch_size=256):
        """Train the NCF model using the engine's rating data."""
        # Prepare training data
        user_ids = self.user_encoder.transform(self.engine.ratings_df['user_id'])
        item_ids = self.item_encoder.transform(self.engine.ratings_df['item_id'])
        ratings = (self.engine.ratings_df['rating'] >= 4).astype(int)
        
        # Create model checkpoint callback
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'ncf_model_best.h5')
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
        ]
        
        history = self.model.fit(
            [user_ids, item_ids],
            ratings,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def predict(self, user_ids, item_ids, batch_size=1024):
        """Generate predictions for user-item pairs."""
        # Transform IDs using encoders
        encoded_users = self.user_encoder.transform(user_ids)
        encoded_items = self.item_encoder.transform(item_ids)
        
        return self.model.predict(
            [encoded_users, encoded_items],
            batch_size=batch_size
        )

    def get_recommendations(self, user_id, num_recommendations=10, exclude_rated=True):
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID to generate recommendations for
            num_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude items the user has already rated
            
        Returns:
            DataFrame containing recommended items with scores
        """
        # Get all items
        all_items = self.engine.items_df['item_id'].values
        
        # Exclude rated items if requested
        if exclude_rated:
            rated_items = self.engine.ratings_df[
                self.engine.ratings_df['user_id'] == user_id
            ]['item_id'].values
            all_items = np.setdiff1d(all_items, rated_items)
        
        # Create user array
        user_array = np.full_like(all_items, user_id)
        
        # Get predictions
        predictions = self.predict(user_array, all_items)
        
        # Create recommendations DataFrame
        recommendations = pd.DataFrame({
            'item_id': all_items,
            'score': predictions.flatten()
        })
        
        # Sort by score and get top N
        recommendations = recommendations.sort_values('score', ascending=False)
        recommendations = recommendations.head(num_recommendations)
        
        # Merge with item details
        recommendations = recommendations.merge(
            self.engine.items_df,
            on='item_id',
            how='left'
        )
        
        return recommendations

    def save(self, filepath):
        """Save the model and encoders."""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, recommendation_engine, filepath):
        """Load a saved model."""
        instance = cls(recommendation_engine)
        instance.model = load_model(filepath)
        return instance 