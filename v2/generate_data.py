import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_user_data(num_users=100):
    """Generate realistic user profiles with enhanced demographics."""
    countries = ['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Japan', 'Brazil', 'India', 'Spain']
    cities = {
        'USA': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'],
        'UK': ['London', 'Manchester', 'Birmingham', 'Liverpool', 'Edinburgh'],
        'Canada': ['Toronto', 'Vancouver', 'Montreal', 'Calgary', 'Ottawa'],
        'Australia': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide'],
        'Germany': ['Berlin', 'Munich', 'Hamburg', 'Frankfurt', 'Cologne'],
        'France': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice'],
        'Japan': ['Tokyo', 'Osaka', 'Kyoto', 'Yokohama', 'Sapporo'],
        'Brazil': ['Sao Paulo', 'Rio de Janeiro', 'Brasilia', 'Salvador', 'Fortaleza'],
        'India': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'],
        'Spain': ['Madrid', 'Barcelona', 'Valencia', 'Seville', 'Bilbao']
    }
    
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    genders = ['M', 'F', 'Other', 'Prefer not to say']
    languages = ['English', 'Spanish', 'French', 'German', 'Japanese', 'Chinese', 'Hindi', 'Portuguese']
    income_brackets = ['0-25k', '25k-50k', '50k-75k', '75k-100k', '100k-150k', '150k+']
    interests = ['Technology', 'Fashion', 'Sports', 'Home & Garden', 'Books', 'Travel', 
                'Food & Cooking', 'Health & Fitness', 'Music', 'Movies', 'Art', 'Gaming']
    
    # Generate country first, then assign city based on country
    countries_list = np.random.choice(countries, num_users)
    cities_list = [np.random.choice(cities[country]) for country in countries_list]
    
    # Generate multiple interests for each user
    user_interests = [
        np.random.choice(interests, size=np.random.randint(2, 6), replace=False).tolist()
        for _ in range(num_users)
    ]
    
    users = {
        'user_id': range(1, num_users + 1),
        'country': countries_list,
        'city': cities_list,
        'age_group': np.random.choice(age_groups, num_users),
        'gender': np.random.choice(genders, num_users),
        'language': np.random.choice(languages, num_users),
        'income_bracket': np.random.choice(income_brackets, num_users),
        'interests': user_interests,
        'registration_date': [
            (datetime.now() - timedelta(days=np.random.randint(0, 365*3))).strftime('%Y-%m-%d')
            for _ in range(num_users)
        ],
        'last_active': [
            (datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime('%Y-%m-%d')
            for _ in range(num_users)
        ]
    }
    
    return pd.DataFrame(users)

def generate_item_data(num_items=50):
    """Generate realistic item data with enhanced attributes."""
    categories = {
        'Electronics': ['Smartphones', 'Laptops', 'Tablets', 'Headphones', 'Cameras'],
        'Home': ['Furniture', 'Decor', 'Bedding', 'Lighting', 'Storage'],
        'Fashion': ['Clothing', 'Shoes', 'Accessories', 'Bags', 'Jewelry'],
        'Sports': ['Equipment', 'Apparel', 'Footwear', 'Accessories', 'Nutrition'],
        'Kitchen': ['Appliances', 'Cookware', 'Utensils', 'Storage', 'Gadgets'],
        'Books': ['Fiction', 'Non-Fiction', 'Educational', 'Children', 'Comics'],
        'Beauty': ['Skincare', 'Makeup', 'Haircare', 'Fragrance', 'Tools']
    }
    
    brands = {
        'Electronics': ['TechPro', 'SmartLife', 'InnovateTech', 'DigiMax', 'EliteGear'],
        'Home': ['HomeEssentials', 'LuxLiving', 'ComfortZone', 'NestWell', 'CozySpace'],
        'Fashion': ['Fashionista', 'StyleHub', 'TrendSetters', 'ChicBoutique', 'VogueLine'],
        'Sports': ['SportsFit', 'ActiveLife', 'PeakPerform', 'AthletePro', 'FitZone'],
        'Kitchen': ['GourmetKit', 'ChefChoice', 'KitchenPro', 'CookMaster', 'CulinaryPlus'],
        'Books': ['BookWorld', 'ReadMore', 'PageTurner', 'StoryVerse', 'LitHub'],
        'Beauty': ['GlowUp', 'BeautyBasics', 'LuxeBeauty', 'PurePamper', 'BeautyCo']
    }
    
    colors = ['Black', 'White', 'Silver', 'Gold', 'Blue', 'Red', 'Green', 'Purple', 'Pink', 'Brown']
    sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'One Size']
    conditions = ['New', 'Like New', 'Good', 'Fair']
    
    items = []
    for i in range(1, num_items + 1):
        # Select random main category and subcategory
        main_category = np.random.choice(list(categories.keys()))
        subcategory = np.random.choice(categories[main_category])
        brand = np.random.choice(brands[main_category])
        
        # Generate tags based on category and features
        tags = [main_category, subcategory, brand]
        if main_category in ['Fashion', 'Sports']:
            tags.extend([np.random.choice(colors), np.random.choice(sizes)])
        
        # Remove duplicates and convert to list
        tags = list(set(tags))
        
        item = {
            'item_id': i,
            'name': f"{brand} {subcategory} {i}",
            'main_category': main_category,
            'subcategory': subcategory,
            'brand': brand,
            'tags': tags,
            'price': round(np.random.uniform(10, 1000), 2),
            'condition': np.random.choice(conditions),
            'average_rating': round(np.random.uniform(3.5, 5.0), 1),
            'num_ratings': np.random.randint(10, 1000),
            'stock_level': np.random.randint(0, 100),
            'release_date': (datetime.now() - timedelta(days=np.random.randint(0, 365*2))).strftime('%Y-%m-%d'),
            'description': f"High-quality {subcategory.lower()} from {brand}. Perfect for everyday use."
        }
        
        # Add color and size for fashion items
        if main_category in ['Fashion', 'Sports']:
            item['color'] = np.random.choice(colors)
            item['size'] = np.random.choice(sizes)
        
        items.append(item)
    
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