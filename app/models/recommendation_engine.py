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
        """Initialize recommendation engine with sample data"""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        
        # Load or generate sample data
        self.ratings_df = self._load_or_generate_ratings(os.path.join(data_dir, 'ratings.csv'))
        self.items_df = self._load_or_generate_items(os.path.join(data_dir, 'items.csv'))
        self.users_df = self._load_or_generate_users(os.path.join(data_dir, 'users.csv'))
        
        # Create user-item matrix for collaborative filtering
        self.user_item_matrix = self._create_user_item_matrix()
        
        # Pre-compute item similarity matrix
        self.item_similarity = self._compute_item_similarity()
        
        # Pre-compute user similarity matrix
        self.user_similarity = self._compute_user_similarity()
        
        # Pre-compute popular items
        self.popular_items = self._compute_popular_items()
    
    def _load_or_generate_ratings(self, file_path):
        """Load ratings data or generate if not exists"""
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        
        # Generate sample ratings
        np.random.seed(42)
        num_users = 100
        num_items = 120  # Increased number of items
        sparsity = 0.1  # 10% of possible ratings are generated
        
        # Generate random ratings
        user_ids = np.random.randint(1, num_users+1, size=int(num_users * num_items * sparsity))
        item_ids = np.random.randint(1, num_items+1, size=int(num_users * num_items * sparsity))
        
        # Make ratings slightly biased toward higher scores (more 4s and 5s)
        ratings = np.random.choice([1, 2, 3, 4, 5], size=int(num_users * num_items * sparsity), 
                                 p=[0.1, 0.15, 0.2, 0.3, 0.25])
        
        # Create dataframe
        ratings_df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings
        })
        
        # Remove duplicates (user-item pairs should be unique)
        ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'item_id'])
        
        # Save to csv
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        ratings_df.to_csv(file_path, index=False)
        
        return ratings_df
    
    def _load_or_generate_items(self, file_path):
        """Load items data or generate if not exists"""
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        
        # Categories based on Frasers Group retail brands
        categories = {
            'Sports Apparel': ['Nike', 'Adidas', 'Under Armour', 'Puma', 'Reebok', 'New Balance', 'Asics', 'Castore'],
            'Sports Equipment': ['Wilson', 'Slazenger', 'Dunlop', 'Everlast', 'Callaway', 'Titleist', 'Speedo'],
            'Fashion': ['Hugo Boss', 'Tommy Hilfiger', 'Calvin Klein', 'Lacoste', 'Ralph Lauren', 'Jack Wills', 'Michael Kors'],
            'Luxury Fashion': ['Gucci', 'Balenciaga', 'Saint Laurent', 'Burberry', 'Off-White', 'Stone Island', 'Versace', 'Prada'],
            'Footwear': ['Nike', 'Adidas', 'Converse', 'Vans', 'Dr. Martens', 'UGG', 'Timberland', 'Salomon'],
            'Outdoor': ['The North Face', 'Columbia', 'Berghaus', 'Patagonia', 'Regatta', 'Mammut', 'Osprey'],
            'Beauty': ['MAC', 'Bobbi Brown', 'Charlotte Tilbury', 'Clinique', 'Estée Lauder', 'Fenty Beauty', 'NARS'],
            'Home': ['Emma Bridgewater', 'Le Creuset', 'Smeg', 'Nespresso', 'Denby', 'Sophie Conran', 'KitchenAid'],
            'Gaming': ['PlayStation', 'Xbox', 'Nintendo', 'Razer', 'SteelSeries', 'Corsair', 'HyperX']
        }
        
        # Product templates by category
        product_templates = {
            'Sports Apparel': [
                '{brand} Running T-Shirt', '{brand} Training Shorts', '{brand} Leggings', 
                '{brand} Windbreaker Jacket', '{brand} Hooded Sweatshirt', '{brand} Tracksuit',
                '{brand} Performance Polo', '{brand} Compression Shirt'
            ],
            'Sports Equipment': [
                '{brand} Tennis Racket', '{brand} Football', '{brand} Basketball', 
                '{brand} Golf Clubs Set', '{brand} Fitness Mat', '{brand} Swimming Goggles',
                '{brand} Dumbbell Set', '{brand} Cycling Helmet'
            ],
            'Fashion': [
                '{brand} Polo Shirt', '{brand} Slim Fit Jeans', '{brand} Casual Shirt', 
                '{brand} Chino Trousers', '{brand} Knitted Sweater', '{brand} Puffer Jacket',
                '{brand} Logo T-Shirt', '{brand} Denim Jacket'
            ],
            'Luxury Fashion': [
                '{brand} Designer Handbag', '{brand} Silk Scarf', '{brand} Logo Belt', 
                '{brand} Designer Sunglasses', '{brand} Leather Wallet', '{brand} Statement Sneakers',
                '{brand} Cashmere Sweater', '{brand} Signature Fragrance'
            ],
            'Footwear': [
                '{brand} Running Shoes', '{brand} Casual Trainers', '{brand} Hiking Boots', 
                '{brand} Leather Loafers', '{brand} Canvas Shoes', '{brand} Slip-On Sneakers',
                '{brand} Winter Boots', '{brand} Sports Sandals'
            ],
            'Outdoor': [
                '{brand} Waterproof Jacket', '{brand} Hiking Backpack', '{brand} Walking Boots', 
                '{brand} Insulated Coat', '{brand} Camping Tent', '{brand} Thermal Fleece',
                '{brand} Trekking Poles', '{brand} Outdoor Water Bottle'
            ],
            'Beauty': [
                '{brand} Foundation', '{brand} Lipstick', '{brand} Eyeshadow Palette', 
                '{brand} Mascara', '{brand} Facial Serum', '{brand} Highlighter',
                '{brand} Skincare Set', '{brand} Makeup Brushes'
            ],
            'Home': [
                '{brand} Ceramic Mug Set', '{brand} Cast Iron Cookware', '{brand} Toaster', 
                '{brand} Coffee Machine', '{brand} Dinner Set', '{brand} Kitchen Knife Set',
                '{brand} Glassware Collection', '{brand} Bedding Set'
            ],
            'Gaming': [
                '{brand} Gaming Console', '{brand} Controller', '{brand} Gaming Headset', 
                '{brand} Gaming Mouse', '{brand} Mechanical Keyboard', '{brand} Gaming Monitor',
                '{brand} VR Headset', '{brand} Gaming Chair'
            ]
        }
        
        # Price ranges by category (min, max)
        price_ranges = {
            'Sports Apparel': (20, 80),
            'Sports Equipment': (30, 500),
            'Fashion': (40, 150),
            'Luxury Fashion': (200, 3000),
            'Footwear': (50, 250),
            'Outdoor': (50, 400),
            'Beauty': (20, 180),
            'Home': (30, 350),
            'Gaming': (40, 800)
        }
        
        # Generate sample items
        item_data = []
        item_id = 1
        
        # Generate a diverse mix of products across all categories
        for category, brands in categories.items():
            for brand in brands:
                # Select 2-3 products per brand
                num_products = random.randint(2, 3)
                templates = random.sample(product_templates[category], num_products)
                
                for template in templates:
                    name = template.format(brand=brand)
                    price_range = price_ranges[category]
                    price = round(random.uniform(price_range[0], price_range[1]), 2)
                    
                    # Create product descriptions based on category
                    if 'Sports' in category:
                        description = f"High-performance {name.lower()} designed for comfort and durability. Features moisture-wicking technology and ergonomic design."
                    elif 'Fashion' in category:
                        description = f"Stylish {name.lower()} crafted with premium materials. Perfect for casual or smart-casual occasions with signature {brand} aesthetics."
                    elif 'Luxury' in category:
                        description = f"Exclusive {name.lower()} showcasing {brand}'s iconic design. Made with the finest materials and exceptional craftsmanship."
                    elif 'Footwear' in category:
                        description = f"Comfortable and stylish {name.lower()} with cushioned insole and durable outsole. Perfect for everyday wear."
                    elif 'Outdoor' in category:
                        description = f"Durable {name.lower()} built to withstand the elements. Features technical fabrics and practical design for outdoor adventures."
                    elif 'Beauty' in category:
                        description = f"Premium quality {name.lower()} for a flawless finish. Long-lasting formula with rich pigmentation and skin-friendly ingredients."
                    elif 'Home' in category:
                        description = f"Elegant {name.lower()} combining style and functionality. Made with high-quality materials for everyday use and special occasions."
                    elif 'Gaming' in category:
                        description = f"High-performance {name.lower()} designed for serious gamers. Features cutting-edge technology for an immersive gaming experience."
                    else:
                        description = f"Quality {name.lower()} from {brand}. Combines style, functionality, and durability."
                    
                    item_data.append({
                        'item_id': item_id,
                        'name': name,
                        'category': category,
                        'brand': brand,
                        'price': price,
                        'description': description
                    })
                    
                    item_id += 1
        
        items_df = pd.DataFrame(item_data)
        
        # Save to csv
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        items_df.to_csv(file_path, index=False)
        
        return items_df
    
    def _load_or_generate_users(self, file_path):
        """Load users data or generate if not exists"""
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        
        # UK city names for realistic user profiles
        uk_cities = ['London', 'Manchester', 'Birmingham', 'Glasgow', 'Liverpool', 
                  'Edinburgh', 'Bristol', 'Leeds', 'Newcastle', 'Cardiff', 
                  'Sheffield', 'Belfast', 'Nottingham', 'Oxford', 'Cambridge',
                  'York', 'Brighton', 'Southampton', 'Portsmouth', 'Leicester']
        
        # UK first names
        first_names = [
            'James', 'Oliver', 'Harry', 'Jack', 'George', 'Noah', 'William', 'Thomas', 'Ethan', 'Muhammad',
            'Olivia', 'Amelia', 'Isla', 'Ava', 'Emily', 'Isabella', 'Mia', 'Poppy', 'Ella', 'Charlotte',
            'Sophia', 'Grace', 'Emma', 'Jacob', 'Charlie', 'Alfie', 'Freddie', 'Theodore', 'Arthur', 'Ryan',
            'Lily', 'Sophie', 'Evie', 'Ruby', 'Ivy', 'Willow', 'Rosie', 'Freya', 'Phoebe', 'Florence'
        ]
        
        # UK last names
        last_names = [
            'Smith', 'Jones', 'Williams', 'Brown', 'Taylor', 'Davies', 'Wilson', 'Evans', 'Thomas', 'Johnson',
            'Roberts', 'Walker', 'Wright', 'Thompson', 'Robinson', 'White', 'Hughes', 'Edwards', 'Green', 'Hall',
            'Lewis', 'Harris', 'Clarke', 'Patel', 'Jackson', 'Wood', 'Turner', 'Martin', 'Cooper', 'Hill',
            'Moore', 'Clark', 'Lee', 'King', 'Baker', 'Harrison', 'Morgan', 'Allen', 'James', 'Scott'
        ]
        
        # User interests (will be used for targeted preferences)
        interests = [
            'Sports', 'Fashion', 'Outdoor Activities', 'Luxury Shopping', 'Gaming',
            'Beauty', 'Home Decor', 'Fitness', 'Casual Wear', 'Premium Brands'
        ]
        
        # Generate sample users
        user_data = []
        np.random.seed(42)  # For reproducibility
        
        for i in range(1, 101):
            # Generate demographic information
            first_name = np.random.choice(first_names)
            last_name = np.random.choice(last_names)
            name = f"{first_name} {last_name}"
            age = np.random.randint(18, 70)
            gender = 'M' if first_name in ['James', 'Oliver', 'Harry', 'Jack', 'George', 'Noah', 'William', 'Thomas', 'Ethan', 'Muhammad',
                                          'Jacob', 'Charlie', 'Alfie', 'Freddie', 'Theodore', 'Arthur', 'Ryan'] else 'F'
            city = np.random.choice(uk_cities)
            
            # Assign 1-3 primary interests
            num_interests = np.random.randint(1, 4)
            primary_interests = ', '.join(np.random.choice(interests, size=num_interests, replace=False))
            
            # Generate spending profile (1-5, with 5 being highest)
            spending_profile = np.random.randint(1, 6)
            
            user_data.append({
                'user_id': i,
                'name': name,
                'age': age,
                'gender': gender,
                'city': city,
                'interests': primary_interests,
                'spending_profile': spending_profile
            })
        
        users_df = pd.DataFrame(user_data)
        
        # Save to csv
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        users_df.to_csv(file_path, index=False)
        
        return users_df
    
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
                'description': 'No description available'
            }
        
        return {
            'name': item['name'].values[0],
            'category': item['category'].values[0],
            'brand': item['brand'].values[0],
            'price': float(item['price'].values[0]),  # Convert numpy.float64 to Python float
            'description': item['description'].values[0]
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
                'description': item['description']
            })
        return items
    
    def get_all_users(self):
        """Get all users"""
        users = []
        for _, user in self.users_df.iterrows():
            user_data = {
                'user_id': int(user['user_id']),  # Convert numpy.int64 to Python int
                'name': user['name'],
                'age': int(user['age']),  # Convert numpy.int64 to Python int
                'gender': user['gender']
            }
            
            # Add additional fields if they exist in the dataframe
            if 'city' in self.users_df.columns:
                user_data['city'] = user['city']
            else:
                user_data['city'] = 'Unknown'
                
            if 'interests' in self.users_df.columns:
                user_data['interests'] = user['interests']
            else:
                user_data['interests'] = 'General Shopping'
                
            if 'spending_profile' in self.users_df.columns:
                user_data['spending_profile'] = int(user['spending_profile'])
            else:
                user_data['spending_profile'] = 3  # Default medium spending profile
                
            users.append(user_data)
        return users 