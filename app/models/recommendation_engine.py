import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from collections import defaultdict
import random

class RecommendationEngine:
    """
    Recommendation Engine class that implements collaborative filtering.
    Phase 2 implementation according to the requirements:
    - Item-based collaborative filtering
    - User-based collaborative filtering
    - Content-based filtering using product metadata
    - Handling cold-start problem with popular items
    """
    
    def __init__(self):
        """Initialize recommendation engine with data from Excel files"""
        # Look for data files in the workspace root's data directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        
        # Define paths to Excel files
        ratings_file = os.path.join(data_dir, 'ratings.xlsx')
        items_file = os.path.join(data_dir, 'items.xlsx')
        users_file = os.path.join(data_dir, 'users.xlsx')
        
        # Load data from Excel files
        print(f"Loading data from {data_dir}")
        self.ratings_df = pd.read_excel(ratings_file)
        self.items_df = pd.read_excel(items_file)
        self.users_df = pd.read_excel(users_file)
        
        # Create user-item matrix for collaborative filtering
        self.user_item_matrix = self._create_user_item_matrix()
        
        # Pre-compute item similarity matrix
        self.item_similarity = self._compute_item_similarity()
        
        # Pre-compute user similarity matrix
        self.user_similarity = self._compute_user_similarity()
        
        # Pre-compute popular items
        self.popular_items = self._compute_popular_items()
        
        print("Data loaded successfully!")
        print(f"Users: {len(self.users_df)}")
        print(f"Items: {len(self.items_df)}")
        print(f"Ratings: {len(self.ratings_df)}")
    
    def _create_user_item_matrix(self):
        """Create a user-item matrix from ratings dataframe"""
        return self.ratings_df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
    
    def _compute_item_similarity(self):
        """Compute item-item similarity matrix using cosine similarity"""
        # Transpose user-item matrix to get item-user matrix
        item_user_matrix = self.user_item_matrix.T
        
        # Compute cosine similarity between items
        return pd.DataFrame(
            cosine_similarity(item_user_matrix),
            index=item_user_matrix.index,
            columns=item_user_matrix.index
        )
    
    def _compute_user_similarity(self):
        """Compute user-user similarity matrix using cosine similarity"""
        # Compute cosine similarity between users
        return pd.DataFrame(
            cosine_similarity(self.user_item_matrix),
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
    
    def _compute_popular_items(self):
        """Compute popular items based on average rating and number of ratings"""
        # Group by item_id and compute mean rating and count
        item_stats = self.ratings_df.groupby('item_id').agg({
            'rating': ['mean', 'count']
        })
        
        # Flatten the column hierarchy
        item_stats.columns = ['mean_rating', 'rating_count']
        
        # Sort by rating count and mean rating
        return item_stats.sort_values(['rating_count', 'mean_rating'], ascending=False)
    
    def recommend_for_user(self, user_id, num_recommendations=5):
        """
        Generate recommendations for a user using user-based collaborative filtering
        """
        user_id = int(user_id)
        
        # Handle cold start problem
        if user_id not in self.user_item_matrix.index:
            return self.recommend_popular_items(num_recommendations)
        
        # Get items already rated by the user
        user_rated_items = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['item_id'])
        
        # Get similar users
        if user_id in self.user_similarity.index:
            similar_users = self.user_similarity[user_id].sort_values(ascending=False).index[1:11]  # top 10 similar users
        else:
            # If user not in similarity matrix, use all users
            similar_users = self.user_similarity.index
        
        # Calculate recommendation scores
        recommendations = defaultdict(float)
        
        for similar_user in similar_users:
            # Skip if it's the same user
            if similar_user == user_id:
                continue
                
            # Get similarity score
            similarity = self.user_similarity.loc[user_id, similar_user]
            
            # Get items rated by the similar user
            similar_user_ratings = self.ratings_df[self.ratings_df['user_id'] == similar_user]
            
            for _, row in similar_user_ratings.iterrows():
                item_id = row['item_id']
                
                # Skip items already rated by the user
                if item_id in user_rated_items:
                    continue
                
                # Add weighted rating to recommendations
                recommendations[item_id] += similarity * row['rating']
        
        # Convert recommendations to list and sort
        recommendations = [
            {
                'item_id': int(item_id),  # Convert numpy.int64 to Python int
                'score': float(score),    # Convert numpy.float64 to Python float
                **self.get_item_details(item_id)
            }
            for item_id, score in sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Return top N recommendations
        return recommendations[:num_recommendations]
    
    def recommend_similar_items(self, item_id, num_recommendations=5):
        """
        Generate recommendations for similar items using item-based collaborative filtering
        """
        item_id = int(item_id)
        
        # Handle case where item doesn't exist
        if item_id not in self.item_similarity.index:
            return self.recommend_popular_items(num_recommendations)
        
        # Get similarity scores for the item
        item_similarities = self.item_similarity[item_id].sort_values(ascending=False)
        
        # Skip the first item (which is the item itself)
        similar_items = item_similarities.index[1:num_recommendations+1]
        
        # Create recommendations list
        recommendations = [
            {
                'item_id': int(similar_item),  # Convert numpy.int64 to Python int
                'score': float(item_similarities[similar_item]),  # Convert numpy.float64 to Python float
                **self.get_item_details(similar_item)
            }
            for similar_item in similar_items
        ]
        
        return recommendations
    
    def recommend_popular_items(self, num_recommendations=5):
        """
        Recommend popular items (for cold start problem)
        """
        popular_items = self.popular_items.head(num_recommendations)
        
        # Create recommendations list
        recommendations = [
            {
                'item_id': int(item_id),  # Convert numpy.int64 to Python int
                'score': float(row['mean_rating']),  # Convert numpy.float64 to Python float
                'popularity': int(row['rating_count']),  # Convert numpy.int64 to Python int
                **self.get_item_details(item_id)
            }
            for item_id, row in popular_items.iterrows()
        ]
        
        return recommendations
    
    def get_item_details(self, item_id):
        """Get details for a specific item"""
        item_id = int(item_id)  # Convert to Python int if needed
        item = self.items_df[self.items_df['item_id'] == item_id]
        
        if len(item) == 0:
            return {
                'name': f'Unknown Item {item_id}',
                'category': 'Unknown',
                'brand': 'Unknown',
                'price': 0.0,
                'release_date': None
            }
        
        return {
            'name': item['name'].values[0],
            'category': item['category'].values[0],
            'brand': item['brand'].values[0],
            'price': float(item['price'].values[0]),  # Convert numpy.float64 to Python float
            'release_date': item['release_date'].values[0] if 'release_date' in item.columns else None
        }
    
    def get_all_items(self):
        """Get all items with their details"""
        items = []
        for _, item in self.items_df.iterrows():
            items.append({
                'item_id': int(item['item_id']),  # Convert numpy.int64 to Python int
                'name': item['name'],
                'category': item['category'],
                'brand': item['brand'],
                'price': float(item['price']),  # Convert numpy.float64 to Python float
                'release_date': item['release_date'] if 'release_date' in self.items_df.columns else None
            })
        return items
    
    def get_all_users(self):
        """Get all users"""
        users = []
        for _, user in self.users_df.iterrows():
            user_data = {
                'user_id': int(user['user_id']),  # Convert numpy.int64 to Python int
                'country': user['country'],
                'age_group': user['age_group'],
                'gender': user['gender']
            }
            
            # Add registration_date if it exists
            if 'registration_d' in self.users_df.columns:
                user_data['registration_date'] = user['registration_d']
            
            users.append(user_data)
        return users 