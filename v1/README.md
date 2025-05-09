# Recommendation System

A demonstration of a recommendation engine system with three versions, following the phases outlined in the interview document.

## Version 1 (Phase 2)

This version implements the following features:
- Item-based collaborative filtering
- User-based collaborative filtering
- Content-based filtering using product metadata
- Handling the cold-start problem with popular items

### Setup and Installation

1. Clone this repository:
```
git clone https://github.com/your_username/recommendations.git
cd recommendations
```

2. Create a virtual environment and activate it:
```
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

### Running the Application

1. Start the Flask development server:
```
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

## Features

- **User-Based Recommendations**: Get recommendations based on similar users' preferences
- **Item-Based Recommendations**: Find similar items to those a user has liked
- **Popular Items Recommendations**: Discover trending items based on popularity

## Implementation Details

The recommendation engine is built with several key components:

1. **Data Processing**: 
   - Loading and preprocessing user, item, and rating data
   - Handling data format inconsistencies and duplicate headers
   - Converting data types and normalizing values

2. **Recommendation Algorithms**:
   - User-based collaborative filtering using cosine similarity
   - Item-based collaborative filtering using cosine similarity
   - Popularity-based recommendations for cold-start problems

3. **UI Components**:
   - Interactive item cards showing product details
   - Match percentages indicating recommendation relevance
   - Category and subcategory badges for easy identification
   - Stock levels and user ratings visualization

### Recent Updates

- Fixed issue with match percentages exceeding 100% by capping at maximum value
- Resolved UI layout issues where category tags were overlapping with product names
- Improved header spacing and badge positioning in item cards
- Enhanced CSV file parsing to handle duplicate header rows

## Documentation

For detailed information about the recommendation algorithms and implementation details, see the [AI Documentation](docs/Phase1_AI.md).

## Future Versions

### Version 2 (Phase 3)
- Hybrid recommendation models
- Integration of financial behavior patterns
- Reinforcement learning components
- Enhanced product descriptions

### Version 3 (Phase 4)
- Real-time recommendation API
- Advanced monitoring systems
- Continuous feedback loops
- Compliance with financial regulations
