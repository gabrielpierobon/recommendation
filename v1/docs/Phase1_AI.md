# Recommendation System - Phase 1 AI Documentation

This document provides a detailed explanation of the artificial intelligence and machine learning techniques used in the Phase 1 implementation of our recommendation system.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Structure](#data-structure)
3. [Recommendation Techniques](#recommendation-techniques)
   - [User-Based Collaborative Filtering](#user-based-collaborative-filtering)
   - [Item-Based Collaborative Filtering](#item-based-collaborative-filtering)
   - [Popularity-Based Recommendations](#popularity-based-recommendations)
4. [Similarity Metrics](#similarity-metrics)
   - [Cosine Similarity](#cosine-similarity)
5. [Cold-Start Problem Handling](#cold-start-problem-handling)
6. [Scoring and Ranking](#scoring-and-ranking)
7. [Implementation Details](#implementation-details)
8. [Performance Considerations](#performance-considerations)
9. [Future Enhancements](#future-enhancements)

## Introduction

The recommendation system implemented in Phase 1 aims to provide personalized product recommendations to users based on their preferences and behavior, as well as the characteristics of products. The system utilizes several complementary techniques to ensure robust recommendations across various scenarios.

The core of our recommendation engine is built upon collaborative filtering techniques, which are enhanced with popularity-based recommendations to handle cold-start problems. This hybrid approach allows our system to deliver relevant recommendations for both new and existing users.

## Data Structure

Our recommendation system operates on three primary datasets:

### 1. Users Data (`users.csv`)

Contains user demographic and preference information:
- `user_id`: Unique identifier for each user
- `country`: User's country of residence
- `city`: User's city of residence
- `age_group`: Age bracket of the user (e.g., "18-24", "25-34")
- `gender`: User's gender
- `language`: Preferred language
- `income_bracket`: Income level category
- `interests`: List of interests/categories the user has expressed preference for
- `registration_date`: When the user registered
- `last_active`: Most recent activity date

### 2. Items Data (`items.csv`)

Contains detailed product information:
- `item_id`: Unique identifier for each product
- `name`: Product name
- `main_category`: Primary category (e.g., "Electronics", "Fashion")
- `subcategory`: More specific category (e.g., "Smartphones", "Casual Wear")
- `brand`: Manufacturer or brand name
- `tags`: List of keywords associated with the product
- `price`: Product price
- `condition`: State of the product (e.g., "New", "Used")
- `average_rating`: Mean rating from users (1-5 scale)
- `num_ratings`: Count of ratings received
- `stock_level`: Available inventory
- `release_date`: When the product was introduced
- `description`: Textual description of the product
- `color`: Product color (for applicable products)
- `size`: Product size (for applicable products)

### 3. Ratings Data (`ratings.csv`)

Contains user-item interactions and ratings:
- `user_id`: User who provided the rating
- `item_id`: Product being rated
- `rating`: Numerical rating (1-5 scale)
- `timestamp`: When the rating was given

## Recommendation Techniques

Our recommendation system employs multiple techniques to generate recommendations for different scenarios:

### User-Based Collaborative Filtering

User-based collaborative filtering works on the principle that users who have agreed in the past tend to agree again in the future. 

#### Intuition

The core intuition is simple: if Alice and Bob have similar taste in products (have rated products similarly), and Alice likes a product that Bob hasn't seen, that product will likely be a good recommendation for Bob.

#### Implementation Details

1. **User Similarity Matrix Construction**:
   - We create a user-item matrix where rows represent users and columns represent items.
   - Each cell represents a user's rating for a particular item.
   - We compute the cosine similarity between users based on their rating patterns.

```python
def compute_user_similarity(self):
    """Compute user similarity matrix using cosine similarity"""
    # Initialize user-item matrix
    user_item_matrix = pd.pivot_table(
        self.ratings_df,
        values='rating',
        index='user_id',
        columns='item_id',
        fill_value=0
    )
    
    # Compute cosine similarity between users
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    
    return user_item_matrix, user_similarity
```

2. **Recommendation Generation**:
   - For a target user, we identify similar users.
   - We collect items that similar users have rated highly but the target user hasn't rated.
   - We weight each item's rating by the similarity between the target user and the user who rated the item.
   - Items are ranked by their weighted scores.

```python
def recommend_for_user(self, user_id, num_recommendations=5):
    """Generate recommendations for a user using user-based collaborative filtering"""
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
```

#### Example

Consider two users, User A and User B, with the following ratings:

| Item     | User A | User B |
|----------|--------|--------|
| Item 1   | 5      | 4      |
| Item 2   | 3      | 3      |
| Item 3   | 4      | 5      |
| Item 4   | ?      | 5      |
| Item 5   | 2      | ?      |

User A and User B have similar rating patterns for items 1, 2, and 3. Since User B rated Item 4 highly, it would be recommended to User A. Similarly, User A's rating for Item 5 would influence recommendations for User B.

### Item-Based Collaborative Filtering

Item-based collaborative filtering works on the principle that items with similar rating patterns tend to be perceived similarly by users.

#### Intuition

If a user likes Item A, and Items A and B are often rated similarly by users, then the user might also like Item B. This approach focuses on item-item relationships rather than user-user relationships.

#### Implementation Details

1. **Item Similarity Matrix Construction**:
   - We create a user-item matrix where rows represent users and columns represent items.
   - Each cell represents a user's rating for a particular item.
   - We compute the cosine similarity between items based on their rating patterns.

```python
def compute_item_similarity(self):
    """Compute item similarity matrix using cosine similarity"""
    # Transpose user-item matrix for item-based collaborative filtering
    item_user_matrix = self.user_item_matrix.T
    
    # Compute cosine similarity between items
    item_similarity = cosine_similarity(item_user_matrix)
    item_similarity = pd.DataFrame(
        item_similarity,
        index=item_user_matrix.index,
        columns=item_user_matrix.index
    )
    
    return item_similarity
```

2. **Recommendation Generation**:
   - For a target item, we identify similar items.
   - We recommend these similar items to users who liked the target item.
   - Items are ranked by their similarity to the target item.

```python
def recommend_similar_items(self, item_id, num_recommendations=5):
    """Generate recommendations for similar items using item-based collaborative filtering"""
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
            'item_id': int(similar_item),
            'score': float(item_similarities[similar_item]),
            **self.get_item_details(similar_item)
        }
        for similar_item in similar_items
    ]
```

#### Example

Consider the following item similarity matrix:

| Item     | Item 1 | Item 2 | Item 3 | Item 4 | Item 5 |
|----------|--------|--------|--------|--------|--------|
| Item 1   | 1.0    | 0.2    | 0.8    | 0.3    | 0.1    |
| Item 2   | 0.2    | 1.0    | 0.3    | 0.7    | 0.5    |
| Item 3   | 0.8    | 0.3    | 1.0    | 0.2    | 0.3    |
| Item 4   | 0.3    | 0.7    | 0.2    | 1.0    | 0.6    |
| Item 5   | 0.1    | 0.5    | 0.3    | 0.6    | 1.0    |

If a user likes Item 1, we would recommend Item 3 (similarity 0.8), then Item 4 (similarity 0.3), and so on.

### Popularity-Based Recommendations

Popularity-based recommendations are simple but effective for new users or items without significant interaction data (the cold-start problem).

#### Intuition

Items that are generally popular with a wide user base are safe recommendations for new users with unknown preferences.

#### Implementation Details

1. **Popularity Metric Calculation**:
   - We count the number of ratings for each item.
   - We also consider the average rating to prioritize highly-rated items.
   - Items are ranked by a combination of rating count and average rating.

```python
def compute_popular_items(self):
    """Compute popularity metrics for items"""
    # Group by item_id and calculate statistics
    item_stats = self.ratings_df.groupby('item_id').agg({
        'rating': ['count', 'mean']
    })
    
    # Flatten multi-level column index
    item_stats.columns = ['rating_count', 'mean_rating']
    
    # Sort by rating count (descending) and then by mean rating (descending)
    return item_stats.sort_values(['rating_count', 'mean_rating'], ascending=False)
```

2. **Recommendation Generation**:
   - We select the top N items based on popularity metrics.
   - These items are recommended to new users or in cold-start scenarios.

```python
def recommend_popular_items(self, num_recommendations=5):
    """Recommend popular items (for cold start problem)"""
    popular_items = self.popular_items.head(num_recommendations)
    
    # Create recommendations list
    recommendations = [
        {
            'item_id': int(item_id),
            'score': float(row['mean_rating']),
            'popularity': int(row['rating_count']),
            **self.get_item_details(item_id)
        }
        for item_id, row in popular_items.iterrows()
    ]
```

#### Example

Consider the following popularity metrics:

| Item     | Rating Count | Average Rating |
|----------|--------------|----------------|
| Item 7   | 245          | 4.7            |
| Item 12  | 189          | 4.5            |
| Item 3   | 176          | 4.8            |
| Item 9   | 154          | 4.2            |
| Item 5   | 132          | 4.4            |

For a new user, we would recommend items 7, 12, 3, 9, and 5 in that order, based on a combination of their popularity and quality.

## Similarity Metrics

### Cosine Similarity

We use cosine similarity as our primary metric for comparing users and items. This metric measures the cosine of the angle between two vectors, effectively capturing the similarity in direction regardless of magnitude.

#### Mathematical Formula

For two vectors A and B, the cosine similarity is calculated as:

```
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```

where A · B is the dot product, and ||A|| and ||B|| are the Euclidean norms of vectors A and B.

#### Advantages

1. **Scale Invariance**: Cosine similarity is not affected by the magnitude of the vectors, only their direction. This is useful for rating data where some users might consistently rate higher or lower than others.

2. **Handling Sparsity**: In recommendation systems, the user-item matrix is typically very sparse (most users rate only a small fraction of items). Cosine similarity handles this sparsity well.

3. **Computational Efficiency**: Cosine similarity is relatively efficient to compute, making it suitable for large-scale recommendation systems.

#### Example

Let's say User A has rated items [5, 0, 3, 4, 0] and User B has rated items [4, 0, 0, 5, 0].

The cosine similarity would be:
```
(5×4 + 0×0 + 3×0 + 4×5 + 0×0) / (sqrt(5² + 0² + 3² + 4² + 0²) × sqrt(4² + 0² + 0² + 5² + 0²))
= (20 + 0 + 0 + 20 + 0) / (sqrt(50) × sqrt(41))
= 40 / (7.07 × 6.40)
= 40 / 45.25
≈ 0.88
```

A cosine similarity of 0.88 indicates that these users have very similar taste, despite not having rated all the same items.

## Cold-Start Problem Handling

The cold-start problem refers to the challenge of making recommendations for new users or items with little or no interaction data.

### Strategies Implemented

1. **For New Users**:
   - When a user with no rating history requests recommendations, we fall back to popularity-based recommendations.
   - This provides a reasonable starting point until more user-specific data is collected.

2. **For New Items**:
   - New items with few or no ratings will naturally appear less frequently in recommendations.
   - Content-based filtering (for future enhancements) can help recommend new items based on their attributes.

### Implementation

The system automatically detects cold-start scenarios and adjusts the recommendation strategy accordingly:

```python
def recommend_for_user(self, user_id, num_recommendations=5):
    # Handle cold start problem
    if user_id not in self.user_item_matrix.index:
        return self.recommend_popular_items(num_recommendations)
    
    # Proceed with user-based collaborative filtering...
```

## Scoring and Ranking

The recommendation system uses different scoring mechanisms depending on the type of recommendation:

### User-Based Collaborative Filtering

Scores are calculated as the weighted sum of ratings from similar users:

```python
recommendations[item_id] += similarity * row['rating']
```

Where:
- `similarity` is the cosine similarity between the target user and a similar user
- `row['rating']` is the rating given by the similar user to the item

### Item-Based Collaborative Filtering

Scores are directly based on the item similarity values:

```python
'score': float(item_similarities[similar_item])
```

### Popularity-Based Recommendations

Scores are based on the average rating of popular items:

```python
'score': float(row['mean_rating'])
```

### Score Normalization and Display

In the frontend, scores are normalized to percentages for better user understanding:

```javascript
// Format the score as percentage for better readability
const scorePercent = Math.min(100, Math.round((item.score / 5) * 100));
```

This converts scores to a 0-100% scale, with a cap at 100% for scores that might exceed the 1-5 rating scale due to weighted calculations.

## Implementation Details

### Data Loading and Preprocessing

The recommendation engine implements robust data loading with error handling:

1. **Handling File Encoding**: Explicitly using UTF-8 encoding to avoid character encoding issues.
2. **Duplicate Header Detection**: Detecting and skipping duplicate header rows in CSV files.
3. **Type Conversion**: Converting string data to appropriate types (int, float, datetime).
4. **List Handling**: Converting string representations of lists to actual Python lists.

### Matrix Construction and Similarity Computation

1. **User-Item Matrix Creation**: Using pandas pivot_table to create a sparse matrix.
2. **Similarity Calculation**: Utilizing scikit-learn's cosine_similarity function for efficient computation.
3. **Indexing**: Maintaining original user/item IDs as indices for easy lookup.

### Recommendation Generation and Caching

1. **Similarity Matrix Caching**: Pre-computing similarity matrices at initialization to avoid repeated calculations.
2. **On-Demand Recommendations**: Generating recommendations when requested rather than pre-computing for all users.
3. **Detail Enrichment**: Augmenting recommendations with item details for immediate display without additional queries.

## Performance Considerations

Several optimization techniques are employed to ensure good performance:

1. **Matrix-Based Operations**: Utilizing efficient numpy/pandas operations for matrix calculations.
2. **Similarity Caching**: Computing similarity matrices once at initialization.
3. **Limiting Similar Users**: Considering only the top 10 most similar users for recommendations.
4. **Data Type Optimization**: Using appropriate data types to minimize memory usage.
5. **Sparse Matrix Representation**: Using sparse data structures when appropriate.

## Future Enhancements

Future versions of the recommendation system could incorporate:

1. **Content-Based Filtering**: Recommending items based on their attributes and user preferences.
2. **Matrix Factorization**: Implementing SVD or ALS for more efficient and accurate recommendations.
3. **Deep Learning Approaches**: Using neural networks for recommendation generation.
4. **Contextual Recommendations**: Considering time, location, and other contextual factors.
5. **A/B Testing Framework**: Systematically evaluating different recommendation algorithms.
6. **Real-Time Updates**: Updating recommendations incrementally as new data arrives.
7. **Personalization Improvements**: Adding more factors to personalize recommendations.
8. **Explanation Generation**: Providing explanations for why items are recommended.

By implementing these enhancements, the recommendation system will continue to evolve in sophistication and accuracy, providing increasingly relevant and personalized recommendations to users. 