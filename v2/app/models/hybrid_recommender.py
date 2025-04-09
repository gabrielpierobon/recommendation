"""
Hybrid Recommendation Module
Combines multiple recommendation techniques including:
1. Collaborative filtering
2. Content-based filtering 
3. Contextual information (time, location, seasonal factors)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import math

class HybridRecommender:
    """
    Hybrid recommendation system that combines multiple recommendation techniques
    """
    
    def __init__(self, recommendation_engine):
        """
        Initialize the hybrid recommender with the base recommendation engine
        """
        self.engine = recommendation_engine
        
        # Default weights for different components
        self.default_weights = {
            'collaborative': 0.6,
            'content_based': 0.25,
            'contextual': 0.15
        }
        
        # Seasonal items categories and their peak months
        self.seasonal_categories = {
            'winter_clothing': [11, 12, 1, 2],  # Nov-Feb
            'summer_clothing': [5, 6, 7, 8],    # May-Aug
            'back_to_school': [8, 9],           # Aug-Sep
            'holiday_items': [11, 12],          # Nov-Dec
            'fitness_equipment': [1, 2],        # Jan-Feb (New Year's resolutions)
            'outdoor_sports': [4, 5, 6, 7, 8],  # Apr-Aug
            'halloween': [10],                  # Oct
            'easter': [3, 4],                   # Mar-Apr (varies)
            'valentine': [2],                   # Feb
        }
        
        # Category mappings for seasonal weights
        self.category_season_mapping = {
            'Fashion': {
                'Winter Wear': 'winter_clothing',
                'Summer Collection': 'summer_clothing',
                'School Uniform': 'back_to_school',
                'Holiday Attire': 'holiday_items',
                'Sportswear': 'fitness_equipment'
            },
            'Sports': {
                'Winter Sports': 'winter_clothing',
                'Outdoor Activities': 'outdoor_sports',
                'Fitness': 'fitness_equipment'
            }
        }
    
    def recommend(self, user_id, num_recommendations=5, weights=None, context=None):
        """
        Generate hybrid recommendations by combining multiple techniques
        
        Parameters:
        - user_id: ID of the user to recommend for
        - num_recommendations: Number of recommendations to generate
        - weights: Custom weights for different recommendation components
        - context: Contextual information (time, location, etc.)
        
        Returns:
        - List of recommended items
        """
        # Use default weights if not specified
        if weights is None:
            weights = self.default_weights
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Initialize empty results for each component
        collaborative_recs = []
        content_based_recs = []
        contextual_recs = []
        
        # Get more recommendations than required for each component to ensure sufficient diversity
        factor = 3  # Get 3x more recommendations than needed
        
        # Get collaborative filtering recommendations (user-based)
        collaborative_recs = self.engine.recommend_for_user(user_id, num_recommendations * factor)
        
        # Get content-based recommendations based on user's interests and previously rated items
        content_based_recs = self._get_content_based_recommendations(user_id, num_recommendations * factor)
        
        # Get contextual recommendations based on current time, season, etc.
        contextual_recs = self._get_contextual_recommendations(user_id, num_recommendations * factor, context)
        
        # Combine all recommendations with their respective weights
        all_recommendations = {}
        
        # Process collaborative filtering recommendations
        for rank, item in enumerate(collaborative_recs):
            item_id = item['item_id']
            # Apply rank-based score adjustment: higher ranks get higher scores
            # Add small decay factor based on rank
            score = item['score'] * (1 - 0.01 * rank) * normalized_weights['collaborative']
            if item_id not in all_recommendations:
                all_recommendations[item_id] = {
                    'item_id': item_id,
                    'score': 0,
                    **{k: v for k, v in item.items() if k not in ['item_id', 'score']}
                }
            all_recommendations[item_id]['score'] += score
        
        # Process content-based recommendations
        for rank, item in enumerate(content_based_recs):
            item_id = item['item_id']
            # Apply rank-based score adjustment
            score = item['score'] * (1 - 0.01 * rank) * normalized_weights['content_based']
            if item_id not in all_recommendations:
                all_recommendations[item_id] = {
                    'item_id': item_id,
                    'score': 0,
                    **{k: v for k, v in item.items() if k not in ['item_id', 'score']}
                }
            all_recommendations[item_id]['score'] += score
        
        # Process contextual recommendations
        for rank, item in enumerate(contextual_recs):
            item_id = item['item_id']
            # Apply rank-based score adjustment
            score = item['score'] * (1 - 0.01 * rank) * normalized_weights['contextual']
            if item_id not in all_recommendations:
                all_recommendations[item_id] = {
                    'item_id': item_id,
                    'score': 0,
                    **{k: v for k, v in item.items() if k not in ['item_id', 'score']}
                }
            all_recommendations[item_id]['score'] += score
        
        # Convert dictionary to sorted list
        final_recommendations = list(all_recommendations.values())
        final_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Add explanation for each recommendation
        for item in final_recommendations:
            item['explanation'] = self._generate_explanation(item, user_id)
        
        # Return top N recommendations
        return final_recommendations[:num_recommendations]
    
    def _get_content_based_recommendations(self, user_id, num_recommendations):
        """
        Generate content-based recommendations based on user's interests and profile
        """
        user_id = int(user_id)
        
        # Get user details
        user_details = None
        for user in self.engine.get_all_users():
            if user['user_id'] == user_id:
                user_details = user
                break
        
        if user_details is None:
            # Fallback to popular items if user not found
            return self.engine.recommend_popular_items(num_recommendations)
        
        # Get user interests
        user_interests = user_details.get('interests', [])
        
        # Get all items
        all_items = self.engine.get_all_items()
        
        # Calculate content-based scores
        item_scores = []
        for item in all_items:
            # Skip items the user has already rated
            user_rated_items = set(self.engine.ratings_df[self.engine.ratings_df['user_id'] == user_id]['item_id'])
            if item['item_id'] in user_rated_items:
                continue
            
            # Calculate interest match score
            interest_score = 0
            # Check if main_category or subcategory matches user interests
            if item['main_category'] in user_interests:
                interest_score += 1.0
            if item['subcategory'] in user_interests:
                interest_score += 0.5
            
            # Check if tags match user interests
            tag_score = 0
            tags = item.get('tags', [])
            for tag in tags:
                if tag in user_interests:
                    tag_score += 0.3
            
            # Normalize tag score (cap at 1.0)
            tag_score = min(tag_score, 1.0)
            
            # Combine scores with weights
            content_score = (interest_score * 0.7) + (tag_score * 0.3)
            
            # Adjust by item's average rating
            content_score *= item['average_rating'] / 5.0
            
            # Add to list if score is positive
            if content_score > 0:
                item_scores.append({
                    'item_id': item['item_id'],
                    'score': content_score,
                    **item
                })
        
        # Sort by score descending
        item_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top N recommendations
        return item_scores[:num_recommendations]
    
    def _get_contextual_recommendations(self, user_id, num_recommendations, context=None):
        """
        Generate recommendations based on contextual factors like time and season
        """
        # If no context provided, use current time
        if context is None:
            current_time = datetime.now()
            context = {
                'month': current_time.month,
                'day_of_week': current_time.weekday(),
                'hour': current_time.hour
            }
        else:
            current_time = datetime.now()
            # Ensure required context keys exist
            context.setdefault('month', current_time.month)
            context.setdefault('day_of_week', current_time.weekday())
            context.setdefault('hour', current_time.hour)
        
        # Get all items
        all_items = self.engine.get_all_items()
        
        # Calculate seasonal weights for each item
        seasonal_scores = []
        for item in all_items:
            # Skip items the user has already rated
            user_rated_items = set(self.engine.ratings_df[self.engine.ratings_df['user_id'] == user_id]['item_id'])
            if item['item_id'] in user_rated_items:
                continue
            
            # Calculate seasonal relevance score
            seasonal_score = self._calculate_seasonal_score(item, context)
            
            # Calculate time-of-day relevance
            time_score = self._calculate_time_score(item, context)
            
            # Combine contextual scores
            contextual_score = (seasonal_score * 0.7) + (time_score * 0.3)
            
            # Only include items with some contextual relevance
            if contextual_score > 0:
                seasonal_scores.append({
                    'item_id': item['item_id'],
                    'score': contextual_score,
                    **item
                })
        
        # Sort by score descending
        seasonal_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top N recommendations
        return seasonal_scores[:num_recommendations]
    
    def _calculate_seasonal_score(self, item, context):
        """
        Calculate how relevant an item is based on the current season/month
        """
        current_month = context['month']
        score = 0.0
        
        # Check if item belongs to a seasonal category
        main_category = item.get('main_category', '')
        subcategory = item.get('subcategory', '')
        
        # Look up seasonal mapping
        season_key = None
        if main_category in self.category_season_mapping:
            if subcategory in self.category_season_mapping[main_category]:
                season_key = self.category_season_mapping[main_category][subcategory]
        
        # If no direct mapping, try to infer from tags
        if not season_key:
            tags = item.get('tags', [])
            for tag in tags:
                tag_lower = tag.lower()
                if 'winter' in tag_lower:
                    season_key = 'winter_clothing'
                    break
                elif 'summer' in tag_lower:
                    season_key = 'summer_clothing'
                    break
                elif 'holiday' in tag_lower or 'christmas' in tag_lower:
                    season_key = 'holiday_items'
                    break
                elif 'school' in tag_lower:
                    season_key = 'back_to_school'
                    break
        
        # If we have a seasonal key, check if current month is in peak season
        if season_key and season_key in self.seasonal_categories:
            peak_months = self.seasonal_categories[season_key]
            if current_month in peak_months:
                # Item is in peak season
                score = 1.0
            else:
                # Calculate distance to nearest peak month
                distances = [min((current_month - m) % 12, (m - current_month) % 12) for m in peak_months]
                min_distance = min(distances)
                # Score decreases with distance from peak months
                score = max(0, 1.0 - (min_distance * 0.25))
        else:
            # Non-seasonal items get a moderate score
            score = 0.5
        
        return score
    
    def _calculate_time_score(self, item, context):
        """
        Calculate relevance based on time of day
        """
        hour = context.get('hour', 12)  # Default to noon if not specified
        score = 0.5  # Default moderate score
        
        # Map categories to relevant time periods
        time_mappings = {
            'Electronics': {'morning': 0.3, 'afternoon': 0.7, 'evening': 0.9},
            'Fashion': {'morning': 0.5, 'afternoon': 0.8, 'evening': 0.6},
            'Home': {'morning': 0.8, 'afternoon': 0.6, 'evening': 0.4},
            'Kitchen': {'morning': 0.9, 'afternoon': 0.5, 'evening': 0.7},
            'Sports': {'morning': 0.8, 'afternoon': 0.6, 'evening': 0.3}
        }
        
        # Determine time period
        if 5 <= hour < 12:
            period = 'morning'
        elif 12 <= hour < 18:
            period = 'afternoon'
        else:
            period = 'evening'
        
        # Get category-based time score
        main_category = item.get('main_category', '')
        if main_category in time_mappings:
            score = time_mappings[main_category][period]
        
        return score
    
    def _generate_explanation(self, item, user_id):
        """
        Generate a human-readable explanation for why this item was recommended
        """
        explanations = []
        
        # Add explanation based on user similarity if applicable
        user_based_explanation = self._explain_user_similarity(item, user_id)
        if user_based_explanation:
            explanations.append(user_based_explanation)
        
        # Add explanation based on item features
        content_explanation = self._explain_content_features(item, user_id)
        if content_explanation:
            explanations.append(content_explanation)
        
        # Add seasonal/contextual explanation if applicable
        contextual_explanation = self._explain_contextual_relevance(item)
        if contextual_explanation:
            explanations.append(contextual_explanation)
        
        # If no explanations were generated, provide a generic one
        if not explanations:
            explanations.append("This is a popular item that many customers enjoy.")
        
        # Return the primary explanation (most specific one)
        return explanations[0]
    
    def _explain_user_similarity(self, item, user_id):
        """Generate explanation based on user similarity"""
        try:
            # Find if this item came from similar users' preferences
            if user_id in self.engine.user_similarity.index:
                similar_users = self.engine.user_similarity[user_id].sort_values(ascending=False).index[1:4]  # top 3 similar users
                
                for similar_user in similar_users:
                    # Check if similar user rated this item highly
                    similar_user_ratings = self.engine.ratings_df[
                        (self.engine.ratings_df['user_id'] == similar_user) & 
                        (self.engine.ratings_df['item_id'] == item['item_id'])
                    ]
                    
                    if not similar_user_ratings.empty and similar_user_ratings.iloc[0]['rating'] > 4:
                        return "Recommended because similar customers with your taste enjoyed this item."
            
            return ""
        except Exception as e:
            return ""
    
    def _explain_content_features(self, item, user_id):
        """Generate explanation based on content features"""
        try:
            # Get user interests
            user_details = None
            for user in self.engine.get_all_users():
                if user['user_id'] == user_id:
                    user_details = user
                    break
            
            if user_details is None:
                return ""
            
            user_interests = user_details.get('interests', [])
            
            # Check if category matches user interests
            if item['main_category'] in user_interests:
                return f"Matches your interest in {item['main_category']} products."
            
            # Check if subcategory matches
            if item['subcategory'] in user_interests:
                return f"Matches your interest in {item['subcategory']} products."
            
            # Check tags
            matching_tags = [tag for tag in item.get('tags', []) if tag in user_interests]
            if matching_tags:
                return f"Matches your interest in {matching_tags[0]}."
            
            # If brand is distinctive, mention it
            if item.get('brand') and item.get('brand') != 'Unknown':
                return f"From {item['brand']}, a brand you might enjoy."
            
            return ""
        except Exception as e:
            return ""
    
    def _explain_contextual_relevance(self, item):
        """Generate explanation based on contextual relevance"""
        current_month = datetime.now().month
        
        # Check if item is seasonal
        main_category = item.get('main_category', '')
        subcategory = item.get('subcategory', '')
        
        # Determine season
        if 3 <= current_month <= 5:
            current_season = "spring"
        elif 6 <= current_month <= 8:
            current_season = "summer"
        elif 9 <= current_month <= 11:
            current_season = "fall"
        else:
            current_season = "winter"
        
        # Look for seasonal terms in tags
        tags = item.get('tags', [])
        for tag in tags:
            tag_lower = tag.lower()
            if current_season in tag_lower:
                return f"Perfect for this {current_season} season."
            if 'seasonal' in tag_lower:
                return "Currently in season and highly relevant."
        
        # Check category seasonal mapping
        season_key = None
        if main_category in self.category_season_mapping:
            if subcategory in self.category_season_mapping[main_category]:
                season_key = self.category_season_mapping[main_category][subcategory]
                
                if season_key and season_key in self.seasonal_categories:
                    peak_months = self.seasonal_categories[season_key]
                    if current_month in peak_months:
                        if 'winter' in season_key:
                            return "Perfect for the winter season."
                        elif 'summer' in season_key:
                            return "Great for the summer season."
                        elif 'holiday' in season_key:
                            return "Popular during the holiday season."
                        elif 'back_to_school' in season_key:
                            return "Ideal for the back-to-school season."
        
        return "" 