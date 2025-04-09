from flask import Flask, render_template, request, jsonify
import os
import sys
import json

# Add the app directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from app.models.recommendation_engine import RecommendationEngine

app = Flask(__name__, 
            static_folder='app/static',
            template_folder='app/templates')

# Initialize recommendation engine
rec_engine = RecommendationEngine()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """Get recommendations based on user input"""
    data = request.get_json()
    user_id = data.get('user_id', None)
    item_id = data.get('item_id', None)
    num_recommendations = int(data.get('num_recommendations', 5))  # Convert to int to be safe
    
    if user_id:
        # Get user-based recommendations
        recommendations = rec_engine.recommend_for_user(user_id, num_recommendations)
    elif item_id:
        # Get item-based recommendations
        recommendations = rec_engine.recommend_similar_items(item_id, num_recommendations)
    else:
        # Get popular items
        recommendations = rec_engine.recommend_popular_items(num_recommendations)
    
    return jsonify(recommendations)

@app.route('/items')
def get_items():
    """Get all available items"""
    items = rec_engine.get_all_items()
    return jsonify(items)

@app.route('/users')
def get_users():
    """Get all available users"""
    users = rec_engine.get_all_users()
    return jsonify(users)

if __name__ == '__main__':
    app.run(debug=True) 