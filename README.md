# Advanced Recommendation System

A comprehensive recommendation engine system that evolves through multiple versions, each adding sophisticated features and intelligence layers.

## Abstract

Imagine walking into a store where the shopkeeper knows exactly what you like, remembers everything you've bought before, understands your budget, and can suggest items you'll love but haven't discovered yet. This is what a recommendation system does in the digital world.

### What is a Recommendation System?

At its core, a recommendation system is like a smart personal assistant that helps users discover items they might like. It works by:

1. **Learning from Past Behavior**
   - What you've liked before (ratings, purchases)
   - What similar users have liked ("people who bought X also bought Y")
   - Your browsing and interaction patterns

2. **Understanding Items**
   - Item characteristics (categories, features, price)
   - How items relate to each other
   - Which items are frequently bought together

3. **Making Smart Predictions**
   - Combining user preferences with item features
   - Considering context (time, location, device)
   - Adapting to changing preferences

### How Does It Work?

This project implements four main approaches to recommendations:

1. **Memory-Based (Version 1)**
   ```
   User A likes items [1,2,3]
   User B likes items [2,3,4]
   → User A might like item 4
   ```
   Like asking friends with similar tastes for recommendations.

2. **Model-Based with Context (Version 2)**
   ```
   User Profile: [age: 25-34, budget: medium, interests: tech]
   Item Features: [category: electronics, price: $$$, rating: 4.5]
   → Recommend if good match + within budget
   ```
   Like a financial advisor who considers your budget and preferences.

3. **Deep Learning (Version 3)**
   ```
   Complex Patterns:
   - Likes action movies but not on weekdays
   - Prefers expensive items only during sales
   - Combines multiple factors non-linearly
   ```
   Like having an AI assistant that understands subtle patterns in your behavior.

4. **Knowledge Graph (KG)**
   ```
   Relationship Patterns:
   - Users who rated X highly also rated Y highly
   - Items in category A are frequently bought with items in category B
   - Paths connecting users to potentially interesting items
   ```
   Like a smart network that understands how everything is connected.

### Real-World Impact

Our recommendation system helps:
- **Users**: Discover relevant items without endless searching
- **Businesses**: Increase sales through personalized suggestions
- **Platforms**: Improve user engagement and satisfaction

Think of it as building a bridge between what users want (even if they don't know it yet) and what's available, while being mindful of practical constraints like budget and context.

## Project Structure

```
recommendations/
├── v1/                 # Phase 1 - Base Recommendation System
├── v2/                 # Phase 2 - Advanced System with Financial Intelligence
├── v3/                 # Phase 3 - Deep Learning Integration with NCF
├── KG/                 # Knowledge Graph Recommendation System
│   ├── data/           # Data files for the knowledge graph
│   ├── docs/           # KG documentation files
│   └── venv_kg/        # KG virtual environment
├── docs/               # Documentation for each phase
│   ├── Phase1_AI.md
│   ├── Phase2_AI.md
│   └── Phase3_AI.md
└── README.md           # This file
```

## Version Overview

### Version 1 (Implemented)
Base recommendation system implementing core functionalities:
- User-based collaborative filtering
- Item-based collaborative filtering
- Content-based filtering using product metadata
- Cold-start problem handling with popular items
- Basic user interface with item cards

Key Features:
- Cosine similarity for user and item matching
- Interactive product recommendations
- Category and subcategory organization
- Stock level indicators
- Rating visualization

### Version 2 (Implemented)
Advanced system with financial intelligence and hybrid approaches:

1. **Hybrid Recommendation System**
   - Dynamic weight adjustment
   - Multiple approach combination:
     - Collaborative filtering
     - Content-based filtering
     - Demographic filtering
     - Financial profile-based filtering

2. **Financial Intelligence Layer**
   - User financial profile analysis
   - Price sensitivity filtering
   - Personalized financial advice
   - Budget-aware recommendations
   - Risk tolerance consideration

3. **Enhanced Features**
   - Advanced content analysis with TF-IDF
   - Multi-factor demographic scoring
   - Sophisticated cold-start handling
   - Dynamic scoring system
   - Performance optimizations

### Version 3 (Implemented)
Deep learning integration with Neural Collaborative Filtering:

1. **Neural Collaborative Filtering (NCF)**
   - Deep learning architecture combining GMF and MLP
   - Non-linear feature interaction modeling
   - Enhanced embedding-based representations
   - Improved cold-start handling through neural embeddings
   - Batch processing for efficient predictions

2. **Enhanced Hybrid System**
   - Integration of multiple approaches with updated weights:
     - Neural collaborative filtering (0.4)
     - Traditional collaborative filtering (0.3)
     - Content-based filtering (0.3)
   - Automatic model selection based on user context

3. **Production-Ready Features**
   - Efficient batch processing
   - LRU caching for frequent users
   - Real-time model updates
   - Comprehensive monitoring
   - Fallback strategies
   - A/B testing support

### Knowledge Graph (KG) System (Implemented)
Graph-based recommendation system leveraging relationships between entities:

1. **Knowledge Graph Modeling**
   - Entity representation: Users, Items, Categories, Subcategories, Brands, Tags
   - Explicit relationships: RATED, IN_CATEGORY, MADE_BY, HAS_TAG
   - Derived relationships: SIMILAR_TASTE, FREQUENTLY_LIKED_TOGETHER
   - Property-rich connections capturing ratings, weights, and timestamps

2. **Advanced Graph-Based Recommendations**
   - Path-based recommendation approaches
   - Graph traversal algorithms for discovering complex patterns
   - Hybrid recommendation combining multiple graph signals:
     - Collaborative filtering through similar users (0.5)
     - Content connections through categories and attributes (0.3)
     - Co-occurrence patterns between items (0.2)

3. **Knowledge Graph Analytics**
   - Community detection for user segmentation
   - Influential node identification
   - Path analysis for recommendation explanation
   - Visual graph exploration
   - Custom Cypher queries for advanced insights

4. **Implementation Features**
   - Neo4j graph database integration
   - Python-based data import and transformation
   - Relationship inference and enhancement
   - Command-line exploration tools
   - Comprehensive Cypher query examples
   - Docker containerization

### Version 4 (Planned)
Enterprise-ready system with advanced features:

1. **Advanced Deep Learning**
   - Attention mechanisms
   - Temporal dynamics modeling
   - Multi-task learning
   - Graph neural networks

2. **Enhanced Production Features**
   - Distributed training
   - Automated hyperparameter optimization
   - Enhanced A/B testing
   - Advanced monitoring and alerting

3. **Additional Features**
   - Real-time model updates
   - Enhanced security measures
   - Privacy-preserving learning
   - Advanced analytics dashboard
   - Natural language interaction
   - Cross-platform support

## Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/your_username/recommendations.git
cd recommendations
```

2. Choose the version you want to run:
```bash
# For Version 1 (Base System)
cd v1

# For Version 2 (Financial Intelligence)
cd v2

# For Version 3 (Neural CF)
cd v3

# For Knowledge Graph System
cd KG
```

3. Create a virtual environment and activate it:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Knowledge Graph (already has venv_kg)
# On Windows
venv_kg\Scripts\activate

# On macOS/Linux
source venv_kg/bin/activate
```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

5. For the Knowledge Graph system, additional setup is required:
```bash
# Start the Neo4j database using Docker
docker-compose up -d

# Import data into Neo4j
python neo4j_importer.py
```

## Running the Application

1. Start the Flask development server (for v1, v2, v3):
```bash
python app.py
```

2. For the Knowledge Graph system:
```bash
# Explore the graph
python graph_explorer.py --action summary

# Generate recommendations
python graph_explorer.py --action recommend --user_id 1 --method hybrid
```

3. Open your web browser:
   - For Flask applications (v1, v2, v3): `http://127.0.0.1:5000/`
   - For Neo4j Browser (KG): `http://localhost:7474/browser/` (username: neo4j, password: password)

## Documentation

Each version has its own detailed documentation:

- **Version 1**: See [Phase 1 Documentation](docs/Phase1_AI.md)
  - Core recommendation algorithms
  - Basic implementation details
  - Data structure and processing

- **Version 2**: See [Phase 2 Documentation](docs/Phase2_AI.md)
  - Financial intelligence layer
  - Hybrid recommendation approach
  - Enhanced features and optimizations

- **Version 3**: See [Phase 3 Documentation](docs/Phase3_AI.md)
  - Neural Collaborative Filtering implementation
  - Deep learning architecture details
  - Production considerations and optimizations

- **Knowledge Graph**: See [KG Documentation](KG/docs/kg_theory.md) and [Cypher Commands](KG/docs/neo4j_cypher_commands.md)
  - Knowledge graph theory and modeling
  - Neo4j implementation details
  - Graph-based recommendation algorithms
  - Cypher query examples

## Recent Updates

### Knowledge Graph System Release
- Implemented Neo4j-based knowledge graph for recommendations
- Created graph modeling of users, items, and their relationships
- Developed derived relationship inference (similarity, co-occurrence)
- Added path-based and hybrid graph recommendation algorithms
- Provided comprehensive Cypher query examples
- Integrated Docker containerization for Neo4j
- Added visual graph exploration capabilities

### Version 3.0 Release
- Implemented Neural Collaborative Filtering
- Added batch processing capabilities
- Enhanced cold-start handling with neural embeddings
- Improved scalability and performance
- Added comprehensive monitoring
- Enhanced error handling and fallbacks
- Updated UI with real-time features

### Version 2.0 Updates
- Implemented financial intelligence layer
- Added hybrid recommendation approach
- Enhanced demographic filtering
- Improved cold-start handling
- Optimized performance with caching
- Enhanced UI with financial insights

### Version 1.1 Updates
- Fixed match percentage calculations
- Resolved category tag overlap issues
- Improved header spacing
- Enhanced CSV parsing

## License

This project is licensed under the MIT License - see the LICENSE file for details.
