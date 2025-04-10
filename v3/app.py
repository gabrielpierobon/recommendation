"""
Recommendation System - Version 2
Enhanced recommendation engine with hybrid models and financial responsibility features
"""

from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
from app.models.recommendation_engine import RecommendationEngine
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Initialize recommendation engine
rec_engine = None  # Will be initialized later

@app.route('/')
def index():
    """Render the main page"""
    logger.info("Rendering main page")
    return render_template('index.html')

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users"""
    logger.info("API request for all users")
    try:
        users = rec_engine.get_all_users()
        logger.info(f"Returning {len(users)} users")
        return jsonify(users)
    except Exception as e:
        logger.error(f"Error getting users: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/api/items', methods=['GET'])
def get_items():
    """Get all items"""
    logger.info("API request for all items")
    try:
        items = rec_engine.get_all_items()
        logger.info(f"Returning {len(items)} items")
        return jsonify(items)
    except Exception as e:
        logger.error(f"Error getting items: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/api/recommendations/user/<user_id>', methods=['GET'])
def user_recommendations(user_id):
    """Get recommendations for a user using traditional collaborative filtering"""
    logger.info(f"API request for user recommendations. User ID: {user_id}")
    try:
        num_recommendations = int(request.args.get('num', 5))
        user_id = int(user_id)
        logger.debug(f"Getting {num_recommendations} recommendations for user {user_id}")
        recommendations = rec_engine.recommend_for_user(user_id, num_recommendations)
        logger.info(f"Returning {len(recommendations)} recommendations for user {user_id}")
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/api/recommendations/item/<item_id>', methods=['GET'])
def item_recommendations(item_id):
    """Get similar items using traditional collaborative filtering"""
    logger.info(f"API request for item recommendations. Item ID: {item_id}")
    try:
        num_recommendations = int(request.args.get('num', 5))
        item_id = int(item_id)
        logger.debug(f"Getting {num_recommendations} similar items for item {item_id}")
        recommendations = rec_engine.recommend_similar_items(item_id, num_recommendations)
        logger.info(f"Returning {len(recommendations)} similar items for item {item_id}")
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f"Error getting similar items for item {item_id}: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/api/recommendations/popular', methods=['GET'])
def popular_recommendations():
    """Get popular items"""
    logger.info("API request for popular items")
    try:
        num_recommendations = int(request.args.get('num', 5))
        logger.debug(f"Getting {num_recommendations} popular items")
        recommendations = rec_engine.recommend_popular_items(num_recommendations)
        logger.info(f"Returning {len(recommendations)} popular items")
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f"Error getting popular items: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

# V2 ENHANCED ENDPOINTS

@app.route('/api/recommendations/hybrid/<user_id>', methods=['GET'])
def hybrid_recommendations(user_id):
    """Get hybrid recommendations combining collaborative, content-based, and contextual filtering"""
    logger.info(f"API request for hybrid recommendations. User ID: {user_id}")
    try:
        num_recommendations = int(request.args.get('num', 5))
        user_id = int(user_id)
        logger.debug(f"Request args: {request.args}")
        
        # Get custom weights if provided
        weights = {}
        if request.args.get('collaborative_weight'):
            weights['collaborative'] = float(request.args.get('collaborative_weight'))
        if request.args.get('content_weight'):
            weights['content_based'] = float(request.args.get('content_weight'))
        if request.args.get('contextual_weight'):
            weights['contextual'] = float(request.args.get('contextual_weight'))
        
        logger.debug(f"Weights: {weights}")
        
        # Use the current date/time as context if not provided
        context = {
            'month': datetime.now().month,
            'day_of_week': datetime.now().weekday(),
            'hour': datetime.now().hour
        }
        
        # Override context if provided in request
        if request.args.get('month'):
            context['month'] = int(request.args.get('month'))
        if request.args.get('day_of_week'):
            context['day_of_week'] = int(request.args.get('day_of_week'))
        if request.args.get('hour'):
            context['hour'] = int(request.args.get('hour'))
        
        logger.debug(f"Context: {context}")
        
        # Use empty weights dict if none were provided
        weights = weights if weights else None
        
        recommendations = rec_engine.get_hybrid_recommendations(
            user_id, 
            num_recommendations, 
            weights, 
            context
        )
        logger.info(f"Returning {len(recommendations)} hybrid recommendations for user {user_id}")
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f"Error getting hybrid recommendations for user {user_id}: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/api/recommendations/content/<item_id>', methods=['GET'])
def content_recommendations(item_id):
    """Get content-based recommendations for a similar item"""
    logger.info(f"API request for content-based recommendations. Item ID: {item_id}")
    try:
        num_recommendations = int(request.args.get('num', 5))
        item_id = int(item_id)
        logger.debug(f"Getting {num_recommendations} content-based recommendations for item {item_id}")
        
        # Check for attribute weights if provided
        attribute_weights = {}
        
        # Parse and convert weights to float, defaulting to 0 if missing or invalid
        try:
            if 'category_weight' in request.args:
                attribute_weights['category'] = float(request.args.get('category_weight', '0'))
            
            if 'brand_weight' in request.args:
                attribute_weights['brand'] = float(request.args.get('brand_weight', '0'))
            
            if 'tags_weight' in request.args:
                attribute_weights['tags'] = float(request.args.get('tags_weight', '0'))
            
            if 'price_weight' in request.args:
                attribute_weights['price'] = float(request.args.get('price_weight', '0'))
        except ValueError as e:
            logger.warning(f"Invalid weight parameter: {str(e)}")
            # Continue with any valid weights that were parsed
        
        logger.info(f"Content-based attribute weights: {attribute_weights}")
        
        recommendations = rec_engine.get_content_based_recommendations_for_item(
            item_id, 
            num_recommendations,
            attribute_weights if attribute_weights else None
        )
        logger.info(f"Returning {len(recommendations)} content-based recommendations for item {item_id}")
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f"Error getting content-based recommendations for item {item_id}: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/api/recommendations/contextual/<user_id>', methods=['GET'])
def contextual_recommendations(user_id):
    """Get contextual recommendations based on demographic matching"""
    logger.info(f"API request for contextual recommendations. User ID: {user_id}")
    try:
        num_recommendations = int(request.args.get('num', 5))
        user_id = int(user_id)
        logger.debug(f"Request args: {request.args}")
        
        # Get the demographic filters - default to false if not explicitly set to true
        demographic_filters = {
            'include_age': request.args.get('include_age', 'false').lower() == 'true',
            'include_gender': request.args.get('include_gender', 'false').lower() == 'true',
            'include_location': request.args.get('include_location', 'false').lower() == 'true',
            'include_income': request.args.get('include_income', 'false').lower() == 'true',
            'include_interests': request.args.get('include_interests', 'false').lower() == 'true',
        }
        
        logger.info(f"Demographic filters: {demographic_filters}")
        
        # Use the current date/time as context if not provided
        context = {
            'month': datetime.now().month,
            'day_of_week': datetime.now().weekday(),
            'hour': datetime.now().hour
        }
        
        # Override context if provided in request
        if request.args.get('month'):
            context['month'] = int(request.args.get('month'))
        if request.args.get('day_of_week'):
            context['day_of_week'] = int(request.args.get('day_of_week'))
        if request.args.get('hour'):
            context['hour'] = int(request.args.get('hour'))
        
        logger.debug(f"Context: {context}")
        
        # Check if any demographic filter is enabled
        if not any(demographic_filters.values()):
            logger.warning("No demographic filters enabled, will use popular items")
        
        recommendations = rec_engine.get_demographic_recommendations(
            user_id, 
            num_recommendations,
            demographic_filters,
            context
        )
        logger.info(f"Returning {len(recommendations)} contextual recommendations for user {user_id}")
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f"Error getting contextual recommendations for user {user_id}: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/api/financial/profile/<user_id>', methods=['GET'])
def financial_profile(user_id):
    """Get a user's financial profile for responsible recommendations"""
    try:
        user_id = int(user_id)
        profile = rec_engine.get_user_spending_profile(user_id)
        return jsonify(profile)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/financial/advice/<user_id>', methods=['GET'])
def financial_advice(user_id):
    """Get financial advice for a user"""
    try:
        user_id = int(user_id)
        advice = rec_engine.get_financial_advice(user_id)
        return jsonify(advice)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/purchases/record', methods=['POST'])
def record_purchase():
    """Record a purchase for a user"""
    logger.info("API request to record purchase")
    try:
        data = request.json
        logger.debug(f"Purchase data: {data}")
        user_id = int(data.get('user_id'))
        item_id = int(data.get('item_id'))
        rec_engine.record_purchase(user_id, item_id)
        logger.info(f"Purchase recorded for user {user_id}, item {item_id}")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error recording purchase: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/api/users/<user_id>', methods=['GET'])
def get_user(user_id):
    """Get a specific user by ID"""
    logger.info(f"API request for user {user_id}")
    try:
        user_id = int(user_id)
        users = rec_engine.get_all_users()
        user = next((u for u in users if u['user_id'] == user_id), None)
        
        if user:
            logger.info(f"Returning user {user_id}")
            return jsonify(user)
        else:
            logger.warning(f"User {user_id} not found")
            return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/api/items/<item_id>', methods=['GET'])
def get_item(item_id):
    """Get a specific item by ID"""
    logger.info(f"API request for item {item_id}")
    try:
        item_id = int(item_id)
        item_details = rec_engine.get_item_details(item_id)
        
        if item_details.get('name') != f'Unknown Item {item_id}':
            logger.info(f"Returning item {item_id}")
            return jsonify(item_details)
        else:
            logger.warning(f"Item {item_id} not found")
            return jsonify({'error': 'Item not found'}), 404
    except Exception as e:
        logger.error(f"Error getting item {item_id}: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/api/recommendations/ncf/<user_id>', methods=['GET'])
def ncf_recommendations(user_id):
    """Get recommendations using Neural Collaborative Filtering"""
    logger.info(f"API request for NCF recommendations. User ID: {user_id}")
    try:
        num_recommendations = int(request.args.get('num', 5))
        user_id = int(user_id)
        exclude_rated = request.args.get('exclude_rated', 'true').lower() == 'true'
        
        logger.debug(f"Getting {num_recommendations} NCF recommendations for user {user_id}")
        recommendations = rec_engine.get_ncf_recommendations(
            user_id,
            num_recommendations=num_recommendations,
            exclude_rated=exclude_rated
        )
        
        # Convert DataFrame to list of dictionaries and ensure all values are JSON serializable
        if isinstance(recommendations, pd.DataFrame):
            recommendations_list = []
            for _, row in recommendations.iterrows():
                item_dict = {}
                for column in row.index:
                    value = row[column]
                    # Handle different types of values
                    if isinstance(value, (np.ndarray, list)):
                        # Convert arrays/lists to lists with native Python types
                        value = [
                            float(x) if isinstance(x, (np.float64, np.float32)) else
                            int(x) if isinstance(x, (np.int64, np.int32)) else
                            bool(x) if isinstance(x, np.bool_) else x
                            for x in value
                        ]
                    elif pd.isna(value):
                        value = None
                    elif isinstance(value, (np.int64, np.int32)):
                        value = int(value)
                    elif isinstance(value, (np.float64, np.float32)):
                        value = float(value) if np.isfinite(value) else None
                    elif isinstance(value, np.bool_):
                        value = bool(value)
                    item_dict[column] = value
                recommendations_list.append(item_dict)
        else:
            recommendations_list = recommendations
            
        logger.info(f"Returning {len(recommendations_list)} NCF recommendations for user {user_id}")
        return jsonify(recommendations_list)
    except Exception as e:
        logger.error(f"Error getting NCF recommendations for user {user_id}: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/api/models/ncf/train', methods=['POST'])
def train_ncf_model():
    """Train the NCF model with specified parameters"""
    logger.info("API request to train NCF model")
    try:
        data = request.json or {}
        validation_split = float(data.get('validation_split', 0.2))
        epochs = int(data.get('epochs', 20))
        batch_size = int(data.get('batch_size', 256))
        
        logger.debug(f"Training NCF model with parameters: validation_split={validation_split}, epochs={epochs}, batch_size={batch_size}")
        history = rec_engine.train_ncf_model(
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Convert training history to serializable format
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'auc': [float(x) for x in history.history['auc']],
            'val_auc': [float(x) for x in history.history['val_auc']]
        }
        
        logger.info("NCF model training completed")
        return jsonify({
            'success': True,
            'history': history_dict
        })
    except Exception as e:
        logger.error(f"Error training NCF model: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Initialize recommendation engine
    print("Initializing recommendation engine...")
    rec_engine = RecommendationEngine()
    
    # Print stats
    print("Data loaded successfully!")
    print(f"Users: {len(rec_engine.get_all_users())}")
    print(f"Items: {len(rec_engine.get_all_items())}")
    print(f"Ratings: {len(rec_engine.ratings_df)}")
    
    # Run app
    logger.info("Starting Flask application")
    app.run(debug=True) 