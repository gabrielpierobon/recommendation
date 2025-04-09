import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_user_data(num_users=100):
    """Generate realistic user profiles."""
    countries = ['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Japan', 'Brazil', 'India', 'Spain']
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
    genders = ['M', 'F', 'Other']
    
    users = {
        'user_id': range(1, num_users + 1),
        'country': np.random.choice(countries, num_users),
        'age_group': np.random.choice(age_groups, num_users),
        'gender': np.random.choice(genders, num_users),
        'registration_date': [
            (datetime.now() - timedelta(days=np.random.randint(0, 365*2))).strftime('%Y-%m-%d')
            for _ in range(num_users)
        ]
    }
    
    return pd.DataFrame(users)

def generate_item_data(num_items=50):
    """Generate realistic item data."""
    brands = ['TechPro', 'SmartLife', 'HomeEssentials', 'Fashionista', 'SportsFit', 
              'GourmetKit', 'BookWorld', 'MusicHub', 'ArtSpace', 'TravelGear']
    categories = ['Electronics', 'Home', 'Fashion', 'Sports', 'Kitchen', 
                 'Books', 'Music', 'Art', 'Travel', 'Beauty']
    
    items = {
        'item_id': range(1, num_items + 1),
        'name': [f"{np.random.choice(brands)} {np.random.choice(categories)} {i}" 
                for i in range(1, num_items + 1)],
        'category': np.random.choice(categories, num_items),
        'brand': np.random.choice(brands, num_items),
        'price': np.random.uniform(10, 1000, num_items).round(2),
        'release_date': [
            (datetime.now() - timedelta(days=np.random.randint(0, 365*3))).strftime('%Y-%m-%d')
            for _ in range(num_items)
        ]
    }
    
    return pd.DataFrame(items)

def generate_ratings(num_users=100, num_items=50, sparsity=0.1):
    """Generate realistic rating data."""
    # Calculate number of ratings based on sparsity
    num_ratings = int(num_users * num_items * sparsity)
    
    # Generate random user-item pairs
    user_ids = np.random.randint(1, num_users + 1, num_ratings)
    item_ids = np.random.randint(1, num_items + 1, num_ratings)
    
    # Generate ratings (1-5 scale)
    ratings = np.random.randint(1, 6, num_ratings)
    
    # Generate timestamps
    timestamps = [
        (datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d %H:%M:%S')
        for _ in range(num_ratings)
    ]
    
    ratings_data = {
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': timestamps
    }
    
    return pd.DataFrame(ratings_data)

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    print("Generating user data...")
    users_df = generate_user_data(100)
    users_df.to_excel('data/users.xlsx', index=False)
    
    print("Generating item data...")
    items_df = generate_item_data(50)
    items_df.to_excel('data/items.xlsx', index=False)
    
    print("Generating ratings data...")
    ratings_df = generate_ratings(100, 50, 0.1)
    ratings_df.to_excel('data/ratings.xlsx', index=False)
    
    print("Data generation complete! Files saved in the 'data' directory.")

if __name__ == "__main__":
    main() 