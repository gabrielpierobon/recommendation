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

@app.route('/users')
def get_users():
    """Get all users"""
    try:
        users = rec_engine.get_all_users()
        return jsonify(users)
    except Exception as e:
        print(f"Error getting users: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/items')
def get_items():
    """Get all items"""
    try:
        items = rec_engine.get_all_items()
        return jsonify(items)
    except Exception as e:
        print(f"Error getting items: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """Get recommendations based on user input"""
    try:
        data = request.get_json()
        print(f"Received recommendation request: {data}")
        
        user_id = data.get('user_id', None)
        item_id = data.get('item_id', None)
        num_recommendations = int(data.get('num_recommendations', 5))  # Convert to int to be safe
        
        if user_id:
            # Get user-based recommendations
            print(f"Getting recommendations for user {user_id}")
            recommendations = rec_engine.recommend_for_user(user_id, num_recommendations)
        elif item_id:
            # Get item-based recommendations
            print(f"Getting recommendations for item {item_id}")
            recommendations = rec_engine.recommend_similar_items(item_id, num_recommendations)
        else:
            # Get popular items
            print("Getting popular item recommendations")
            recommendations = rec_engine.recommend_popular_items(num_recommendations)
        
        print(f"Generated {len(recommendations)} recommendations")
        return jsonify(recommendations)
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 