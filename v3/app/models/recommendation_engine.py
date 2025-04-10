"""
Enhanced Recommendation Engine V2
Combines collaborative filtering, content-based filtering, and contextual information
with financial behavior analysis for responsible recommendations
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from collections import defaultdict
import random
from datetime import datetime

# Import new components
from app.models.hybrid_recommender import HybridRecommender
from app.models.financial_behavior import FinancialBehaviorAnalyzer
from app.models.ncf_model import NCFRecommender

class RecommendationEngine:
    """
    Enhanced recommendation engine that implements multiple recommendation techniques
    and responsible financial behavior analysis.
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
                # Skip header
                header = f.readline().strip()
                
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
                    except ValueError as e:
                        print(f"Error parsing line {line_num}: {line}")
                        print(f"Error details: {e}")
            
            # Create DataFrame
            self.ratings_df = pd.DataFrame(ratings_data)
            
            # === ITEMS DATA ===
            self.items_df = pd.read_csv(os.path.join(data_dir, 'items.csv'))
            
            # === USERS DATA ===
            self.users_df = pd.read_csv(os.path.join(data_dir, 'users.csv'))
            
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
        
        # Initialize hybrid recommender
        self.hybrid_recommender = HybridRecommender(self)
        
        # Initialize financial behavior analyzer
        self.financial_analyzer = FinancialBehaviorAnalyzer(self)
        
        # Initialize NCF recommender
        self.ncf_recommender = NCFRecommender(self)
        
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
    
    # V2 ENHANCED METHODS
    
    def get_hybrid_recommendations(self, user_id, num_recommendations=5, weights=None, context=None):
        """
        Get hybrid recommendations combining multiple recommendation approaches.
        
        Args:
            user_id: ID of the user to get recommendations for
            num_recommendations: Number of recommendations to return
            weights: Dictionary of weights for different recommendation approaches
            context: Contextual information for recommendations
            
        Returns:
            DataFrame containing recommended items
        """
        if weights is None:
            weights = {
                'collaborative': 0.3,
                'content': 0.2,
                'contextual': 0.2,
                'ncf': 0.3
            }
            
        # Get recommendations from different approaches
        collab_recs = self.recommend_for_user(user_id, num_recommendations)
        content_recs = self.get_content_based_recommendations(user_id, num_recommendations)
        context_recs = self.get_contextual_recommendations(user_id, num_recommendations, context)
        ncf_recs = self.get_ncf_recommendations(user_id, num_recommendations)
        
        # Combine recommendations using weights
        all_recs = pd.concat([
            collab_recs.assign(source='collaborative', weight=weights.get('collaborative', 0.25)),
            content_recs.assign(source='content', weight=weights.get('content', 0.25)),
            context_recs.assign(source='contextual', weight=weights.get('contextual', 0.25)),
            ncf_recs.assign(source='ncf', weight=weights.get('ncf', 0.25))
        ])
        
        # Aggregate scores
        final_recs = all_recs.groupby('item_id').agg({
            'score': lambda x: np.average(x, weights=all_recs.loc[x.index, 'weight'])
        }).reset_index()
        
        # Sort and get top recommendations
        final_recs = final_recs.sort_values('score', ascending=False).head(num_recommendations)
        
        # Add item details
        final_recs = final_recs.merge(self.items_df, on='item_id', how='left')
        
        return final_recs
    
    def get_content_based_recommendations(self, user_id, num_recommendations=5):
        """
        Generate content-based recommendations for a user based on their interests
        and previous ratings
        """
        if hasattr(self, 'hybrid_recommender'):
            return self.hybrid_recommender._get_content_based_recommendations(user_id, num_recommendations)
        else:
            # Fallback to standard recommendation if hybrid recommender not available
            return self.recommend_for_user(user_id, num_recommendations)
    
    def get_content_based_recommendations_for_item(self, item_id, num_recommendations=5, attribute_weights=None):
        """
        Generate content-based recommendations for a specific item based on its attributes
        """
        item_id = int(item_id)
        
        # Handle case where item doesn't exist
        item = self.items_df[self.items_df['item_id'] == item_id]
        if len(item) == 0:
            return self.recommend_popular_items(num_recommendations)
        
        # Extract item attributes
        item = item.iloc[0]
        main_category = item['main_category']
        subcategory = item['subcategory']
        brand = item['brand']
        tags = item['tags'] if isinstance(item['tags'], list) else []
        price = float(item['price'])
        
        # Set default attribute weights if not provided
        if attribute_weights is None:
            attribute_weights = {
                'category': 0.3,
                'brand': 0.2,
                'tags': 0.4,
                'price': 0.1
            }
        
        print(f"Content-based recommendations for item {item_id} with attribute weights: {attribute_weights}")
        
        # Get all items except the current one
        other_items = self.items_df[self.items_df['item_id'] != item_id]
        
        # Calculate content-based scores
        item_scores = []
        
        for _, other_item in other_items.iterrows():
            # Calculate category match score
            category_score = 0.0
            if other_item['main_category'] == main_category:
                category_score += 1.0
            if other_item['subcategory'] == subcategory:
                category_score += 0.5
            
            # Calculate brand match score
            brand_score = 1.0 if other_item['brand'] == brand else 0.0
            
            # Calculate tags match score
            other_tags = other_item['tags'] if isinstance(other_item['tags'], list) else []
            tag_matches = len(set(tags).intersection(set(other_tags)))
            tags_score = tag_matches / max(len(tags), 1) if tags else 0.0
            
            # Calculate price similarity score
            other_price = float(other_item['price'])
            # Calculate price difference as percentage
            max_price = max(price, other_price)
            min_price = min(price, other_price)
            if max_price == 0:
                price_score = 1.0  # Both prices are 0
            else:
                price_diff_pct = (max_price - min_price) / max_price
                price_score = 1.0 - price_diff_pct  # Higher score for similar prices
            
            # Debug log scores for this item
            if other_item['item_id'] % 10 == 0:  # Log only some items to avoid flooding
                print(f"Item {other_item['item_id']} scores - Category: {category_score}, Brand: {brand_score}, Tags: {tags_score}, Price: {price_score}")
            
            # Combine scores with weights - only use attributes with weights > 0
            total_score = 0.0
            weight_used = False
            
            if attribute_weights.get('category', 0) > 0:
                total_score += category_score * attribute_weights.get('category', 0.3)
                weight_used = True
            
            if attribute_weights.get('brand', 0) > 0:
                total_score += brand_score * attribute_weights.get('brand', 0.2)
                weight_used = True
            
            if attribute_weights.get('tags', 0) > 0:
                total_score += tags_score * attribute_weights.get('tags', 0.4)
                weight_used = True
            
            if attribute_weights.get('price', 0) > 0:
                total_score += price_score * attribute_weights.get('price', 0.1)
                weight_used = True
            
            # If no weights were used, return popular items
            if not weight_used:
                print("No attribute weights are active, using default weighting")
                total_score = (
                    category_score * 0.3 +
                    brand_score * 0.2 +
                    tags_score * 0.4 +
                    price_score * 0.1
                )
            
            # Only include items with a positive score
            if total_score > 0:
                item_scores.append({
                    'item_id': int(other_item['item_id']),
                    'score': float(total_score),
                    'name': other_item['name'],
                    'main_category': other_item['main_category'],
                    'subcategory': other_item['subcategory'],
                    'brand': other_item['brand'],
                    'tags': other_item['tags'],
                    'price': float(other_item['price']),
                    'condition': other_item['condition'],
                    'average_rating': float(other_item['average_rating']),
                    'num_ratings': int(other_item['num_ratings']),
                    'stock_level': int(other_item['stock_level']),
                    'description': other_item['description']
                })
        
        # Sort by score descending
        item_scores.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"Found {len(item_scores)} content-based recommendations for item {item_id}")
        if len(item_scores) > 0:
            print(f"Top recommendation score: {item_scores[0]['score']}, name: {item_scores[0]['name']}")
        
        # Return top N recommendations
        return item_scores[:num_recommendations]
    
    def get_contextual_recommendations(self, user_id, num_recommendations=5, context=None):
        """
        Generate recommendations based on contextual factors like time, season, etc.
        """
        if hasattr(self, 'hybrid_recommender'):
            return self.hybrid_recommender._get_contextual_recommendations(user_id, num_recommendations, context)
        else:
            # Fallback to standard recommendation if hybrid recommender not available
            return self.recommend_for_user(user_id, num_recommendations)
    
    def get_demographic_recommendations(self, user_id, num_recommendations=5, demographic_filters=None, context=None):
        """
        Generate recommendations based on demographic matching
        """
        user_id = int(user_id)
        
        # Get user details
        user_details = None
        for user in self.get_all_users():
            if user['user_id'] == user_id:
                user_details = user
                break
        
        if user_details is None:
            return self.recommend_popular_items(num_recommendations)
        
        # Default filters if not provided
        if demographic_filters is None:
            demographic_filters = {
                'include_age': True,
                'include_gender': True,
                'include_location': False,
                'include_income': False,
                'include_interests': True
            }
            
        print(f"Demographic recommendations for user {user_id} with filters: {demographic_filters}")
        
        # Get all users
        all_users = self.get_all_users()
        
        # Find similar users based on demographic attributes
        similar_users = []
        
        # Check if any filter is enabled
        any_filter_enabled = any(demographic_filters.values())
        if not any_filter_enabled:
            print("No demographic filters enabled, using popular items")
            return self.recommend_popular_items(num_recommendations)
        
        for other_user in all_users:
            if other_user['user_id'] == user_id:
                continue  # Skip the current user
            
            similarity_score = 0.0
            factors_considered = 0
            
            # Age group similarity
            if demographic_filters.get('include_age', True):
                factors_considered += 1
                if user_details.get('age_group') == other_user.get('age_group'):
                    similarity_score += 0.3
            
            # Gender similarity
            if demographic_filters.get('include_gender', True):
                factors_considered += 1
                if user_details.get('gender') == other_user.get('gender'):
                    similarity_score += 0.2
            
            # Location similarity
            if demographic_filters.get('include_location', False):
                factors_considered += 1
                if user_details.get('country') == other_user.get('country'):
                    similarity_score += 0.1
                if user_details.get('city') == other_user.get('city'):
                    similarity_score += 0.2
            
            # Income bracket similarity
            if demographic_filters.get('include_income', False):
                factors_considered += 1
                if user_details.get('income_bracket') == other_user.get('income_bracket'):
                    similarity_score += 0.2
            
            # Interest similarity
            if demographic_filters.get('include_interests', True):
                factors_considered += 1
                user_interests = set(user_details.get('interests', []))
                other_interests = set(other_user.get('interests', []))
                
                if user_interests and other_interests:
                    common_interests = user_interests.intersection(other_interests)
                    interest_similarity = len(common_interests) / max(len(user_interests), 1)
                    similarity_score += 0.3 * interest_similarity
            
            # Normalize similarity score by the number of factors considered
            if factors_considered > 0:
                similarity_score = similarity_score / factors_considered
            
            # Add to similar users if similarity is significant
            if similarity_score > 0.1:  # Lower threshold for demographic similarity
                similar_users.append({
                    'user_id': other_user['user_id'],
                    'similarity': similarity_score
                })
        
        print(f"Found {len(similar_users)} demographically similar users")
        
        # Sort by similarity
        similar_users.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get top similar users
        top_similar_users = [u['user_id'] for u in similar_users[:min(10, len(similar_users))]]  # Use top 10 similar users
        
        if not top_similar_users:
            print("No similar users found, using popular items")
            return self.recommend_popular_items(num_recommendations)
        
        # Get items rated highly by similar users
        similar_user_ratings = self.ratings_df[self.ratings_df['user_id'].isin(top_similar_users)]
        high_ratings = similar_user_ratings[similar_user_ratings['rating'] >= 4.0]
        
        # Count item frequencies
        item_counts = high_ratings['item_id'].value_counts().to_dict()
        
        # Get items already rated by the user
        user_rated_items = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['item_id'])
        
        # Filter out items already rated by the user
        item_counts = {item_id: count for item_id, count in item_counts.items() if item_id not in user_rated_items}
        
        # Sort items by frequency
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(sorted_items)} recommended items based on demographic similarity")
        
        # Get recommendations
        recommendations = []
        for item_id, count in sorted_items[:num_recommendations]:
            item_details = self.get_item_details(item_id)
            recommendations.append({
                'item_id': int(item_id),
                'score': float(count / len(top_similar_users)),  # Normalize score
                **item_details
            })
        
        # If not enough recommendations, add popular items
        if len(recommendations) < num_recommendations:
            print(f"Not enough recommendations ({len(recommendations)}), adding popular items to reach {num_recommendations}")
            popular_items = self.recommend_popular_items(num_recommendations - len(recommendations))
            # Filter out items already in recommendations
            existing_ids = {rec['item_id'] for rec in recommendations}
            popular_items = [item for item in popular_items if item['item_id'] not in existing_ids]
            recommendations.extend(popular_items)
        
        return recommendations[:num_recommendations]
    
    def record_purchase(self, user_id, item_id):
        """
        Record a user purchase for financial behavior analysis
        """
        item_details = self.get_item_details(item_id)
        price = item_details.get('price', 0)
        self.financial_analyzer.add_purchase(user_id, item_id, price)
    
    def get_user_spending_profile(self, user_id):
        """
        Get a user's spending profile for financial insights
        """
        return self.financial_analyzer.get_user_spending_profile(user_id)
    
    def get_financial_advice(self, user_id):
        """
        Get responsible spending advice for a user
        """
        return self.financial_analyzer.get_responsible_spending_advice(user_id)

    def get_ncf_recommendations(self, user_id, num_recommendations=5, exclude_rated=True):
        """
        Get recommendations using the Neural Collaborative Filtering model.
        
        Args:
            user_id: ID of the user to get recommendations for
            num_recommendations: Number of recommendations to return
            exclude_rated: Whether to exclude items the user has already rated
            
        Returns:
            DataFrame containing recommended items with scores
        """
        return self.ncf_recommender.get_recommendations(
            user_id,
            num_recommendations=num_recommendations,
            exclude_rated=exclude_rated
        )

    def train_ncf_model(self, validation_split=0.2, epochs=20, batch_size=256):
        """
        Train the Neural Collaborative Filtering model.
        
        Args:
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training history
        """
        return self.ncf_recommender.train(
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size
        ) 