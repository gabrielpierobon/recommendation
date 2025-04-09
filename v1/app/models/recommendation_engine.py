import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from collections import defaultdict
import random
from datetime import datetime

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
        """Initialize recommendation engine with data from CSV files"""
        # Look for data files in the workspace root's data directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        
        # Define paths to CSV files
        ratings_file = os.path.join(data_dir, 'ratings.csv')
        items_file = os.path.join(data_dir, 'items.csv')
        users_file = os.path.join(data_dir, 'users.csv')
        
        # Load data from CSV files with proper data types
        print(f"Loading data from {data_dir}")
        
        # Manually read and parse CSV files to avoid header issues
        try:
            # === RATINGS DATA ===
            with open(os.path.join(data_dir, 'ratings.csv'), 'r', encoding='utf-8') as f:
                # Print first 10 lines for debugging
                print("First 10 lines of ratings.csv:")
                lines = []
                for i in range(10):
                    line = f.readline().strip()
                    print(f"Line {i+1}: {line}")
                    lines.append(line)
                
                # Reset file pointer
                f.seek(0)
                
                # Process file properly
                header = f.readline().strip()  # Skip header
                print(f"Header: {header}")
                
                ratings_data = []
                line_num = 1
                for line in f:
                    line_num += 1
                    line = line.strip()
                    
                    # Skip duplicate header lines
                    if line == header or line == "user_id,item_id,rating,timestamp":
                        print(f"Skipping duplicate header at line {line_num}")
                        continue
                        
                    try:
                        parts = line.split(',')
                        if len(parts) >= 4:
                            user_id, item_id, rating = parts[0], parts[1], parts[2]
                            timestamp = ','.join(parts[3:])  # Rejoin timestamp if it has commas
                            ratings_data.append({
                                'user_id': int(user_id),
                                'item_id': int(item_id),
                                'rating': float(rating),
                                'timestamp': pd.to_datetime(timestamp)
                            })
                        else:
                            print(f"Warning: Line {line_num} has unexpected format: {line}")
                    except ValueError as e:
                        print(f"Error parsing line {line_num}: {line}")
                        print(f"Error details: {e}")
                        # Skip this line and continue
            
            # Create DataFrame
            self.ratings_df = pd.DataFrame(ratings_data)
            print(f"Ratings data loaded: {len(self.ratings_df)} entries")
            
            # === ITEMS DATA ===
            # Read just the header first to check structure
            with open(os.path.join(data_dir, 'items.csv'), 'r', encoding='utf-8') as f:
                header = f.readline().strip()
                print(f"Items header: {header}")
            
            self.items_df = pd.read_csv(
                os.path.join(data_dir, 'items.csv'),
                encoding='utf-8'
            )
            print(f"Items data loaded: {len(self.items_df)} entries")
            
            # === USERS DATA ===
            # Read just the header first to check structure
            with open(os.path.join(data_dir, 'users.csv'), 'r', encoding='utf-8') as f:
                header = f.readline().strip()
                print(f"Users header: {header}")
                
            self.users_df = pd.read_csv(
                os.path.join(data_dir, 'users.csv'),
                encoding='utf-8'
            )
            print(f"Users data loaded: {len(self.users_df)} entries")
            
            print("All CSV files loaded successfully!")
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Convert string representation of lists to actual lists
        self.items_df['tags'] = self.items_df['tags'].apply(lambda x: eval(x) if pd.notna(x) else [])
        self.items_df['price'] = self.items_df['price'].astype(float)
        self.items_df['average_rating'] = self.items_df['average_rating'].astype(float)
        self.items_df['num_ratings'] = self.items_df['num_ratings'].astype(int)
        self.items_df['stock_level'] = self.items_df['stock_level'].astype(int)
        self.items_df['release_date'] = pd.to_datetime(self.items_df['release_date'])
        
        # Convert string representation of lists to actual lists
        self.users_df['interests'] = self.users_df['interests'].apply(lambda x: eval(x) if pd.notna(x) else [])
        self.users_df['registration_date'] = pd.to_datetime(self.users_df['registration_date'])
        self.users_df['last_active'] = pd.to_datetime(self.users_df['last_active'])
        
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
                'main_category': 'Unknown',
                'subcategory': 'Unknown',
                'brand': 'Unknown',
                'price': 0.0,
                'tags': [],
                'condition': 'Unknown',
                'stock_level': 0,
                'description': 'No description available'
            }
        
        details = {
            'name': item['name'].values[0],
            'main_category': item['main_category'].values[0],
            'subcategory': item['subcategory'].values[0],
            'brand': item['brand'].values[0],
            'tags': item['tags'].values[0],
            'price': float(item['price'].values[0]),
            'condition': item['condition'].values[0],
            'average_rating': float(item['average_rating'].values[0]),
            'num_ratings': int(item['num_ratings'].values[0]),
            'stock_level': int(item['stock_level'].values[0]),
            'release_date': pd.Timestamp(item['release_date'].values[0]).strftime('%Y-%m-%d'),  # Convert numpy.datetime64 to string
            'description': item['description'].values[0]
        }
        
        # Add color and size for fashion/sports items
        if item['main_category'].values[0] in ['Fashion', 'Sports']:
            details['color'] = item['color'].values[0]
            details['size'] = item['size'].values[0]
        
        return details
    
    def get_all_items(self):
        """Get all items with their details"""
        items = []
        for _, item in self.items_df.iterrows():
            item_details = {
                'item_id': int(item['item_id']),
                'name': item['name'],
                'main_category': item['main_category'],
                'subcategory': item['subcategory'],
                'brand': item['brand'],
                'tags': item['tags'],
                'price': float(item['price']),
                'condition': item['condition'],
                'average_rating': float(item['average_rating']),
                'num_ratings': int(item['num_ratings']),
                'stock_level': int(item['stock_level']),
                'release_date': pd.Timestamp(item['release_date']).strftime('%Y-%m-%d'),  # Convert numpy.datetime64 to string
                'description': item['description']
            }
            
            # Add color and size for fashion/sports items
            if item['main_category'] in ['Fashion', 'Sports']:
                item_details['color'] = item['color']
                item_details['size'] = item['size']
            
            items.append(item_details)
        return items
    
    def get_all_users(self):
        """Get all users with their details"""
        users = []
        for _, user in self.users_df.iterrows():
            users.append({
                'user_id': int(user['user_id']),
                'country': user['country'],
                'city': user['city'],
                'age_group': user['age_group'],
                'gender': user['gender'],
                'language': user['language'],
                'income_bracket': user['income_bracket'],
                'interests': user['interests'],
                'registration_date': pd.Timestamp(user['registration_date']).strftime('%Y-%m-%d'),  # Convert numpy.datetime64 to string
                'last_active': pd.Timestamp(user['last_active']).strftime('%Y-%m-%d')  # Convert numpy.datetime64 to string
            })
        return users 