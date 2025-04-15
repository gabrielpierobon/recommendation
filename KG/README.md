# Knowledge Graph for Recommendation System

This component implements a knowledge graph representation of the recommendation system data using Neo4j. The knowledge graph enables more complex queries and visualizations of the relationships between users, items, categories, and other entities.

## Overview

The knowledge graph builds upon the existing recommendation system data, but represents it as a graph with nodes and relationships, enabling:

1. **Complex relationship discovery** - Find patterns that aren't obvious in tabular data
2. **Multi-hop recommendations** - Generate recommendations based on paths through the graph
3. **Visual exploration** - See how users, items, and other entities relate to each other
4. **Graph-based analytics** - Apply graph algorithms to find clusters, influential nodes, etc.

## Architecture & Implementation

The system is structured around several key components:

1. **Neo4j Database** - A graph database that stores entities as nodes and their connections as relationships
2. **Data Importers** - Python scripts that transform user-item interaction data into graph format
3. **Graph Enhancement Modules** - Functions that derive additional relationships from existing data
4. **Inference Engines** - Algorithms that traverse the graph to generate recommendations
5. **Exploration API** - Tools for querying and visualizing the graph

### Core Files and Their Functions

- **neo4j_importer.py** - Handles importing data from CSV/Excel files into Neo4j, creating the graph structure
- **graph_explorer.py** - Provides command-line interface for exploring the graph and generating recommendations
- **docker-compose.yml** - Configures the Neo4j database instance in Docker
- **requirements.txt** - Lists required Python dependencies
- **docs/** - Documentation on knowledge graph theory and Cypher query examples

## Data Handling

### Data Flow

1. **Input Data Sources**
   - User demographics: stored in `data/users.xlsx`
   - Item metadata: stored in `data/items.xlsx`
   - User-item ratings: stored in `data/ratings.xlsx`

2. **Import Process**
   - `neo4j_importer.py` reads these files and transforms them into graph structures
   - Data cleaning and validation happens during import
   - Primary keys are preserved and used as node identifiers

3. **Graph Construction Steps**
   ```python
   # Example from neo4j_importer.py
   def import_users(self, users_df):
       """Import user data from DataFrame to Neo4j."""
       for _, row in users_df.iterrows():
           # Create user node with properties
           self.graph.run("""
           CREATE (u:User {
               id: $id,
               name: $name,
               country: $country,
               city: $city,
               age_group: $age_group,
               gender: $gender,
               language: $language,
               income_bracket: $income_bracket,
               interests: $interests,
               registration_date: $registration_date,
               last_active: $last_active
           })
           """, parameters=row.to_dict())
   ```

4. **Data Update Mechanism**
   - The system supports incremental updates to existing data
   - New relationships can be derived when data changes
   - Existing nodes can be updated with new properties

## Graph Modeling in Neo4j

### Node Types

The knowledge graph models the following entity types as nodes:

1. **User Nodes** (`User` label)
   - Properties: `id`, `name`, `country`, `city`, `age_group`, `gender`, `language`, `income_bracket`, `interests`, `registration_date`, `last_active`
   - Primary key: `id`

2. **Item Nodes** (`Item` label)
   - Properties: `id`, `name`, `main_category`, `subcategory`, `brand`, `tags`, `price`, `condition`, `avg_rating`, `num_ratings`, `stock_level`, `release_date`, `description`
   - Primary key: `id`

3. **Category Nodes** (`Category` label)
   - Properties: `name`, `description`
   - Primary key: `name`

4. **Subcategory Nodes** (`Subcategory` label)
   - Properties: `name`, `description`
   - Primary key: `name`

5. **Brand Nodes** (`Brand` label)
   - Properties: `name`, `country_of_origin`, `founded_year`
   - Primary key: `name`

6. **Tag Nodes** (`Tag` label)
   - Properties: `name`
   - Primary key: `name`

### Relationship Types

The graph uses the following relationship types to connect nodes:

1. **Primary/Observed Relationships**
   - `RATED`: Connects `User` to `Item` with properties `rating` (1-5 scale) and `timestamp`
   - `IN_CATEGORY`: Connects `Item` to `Category`
   - `IN_SUBCATEGORY`: Connects `Item` to `Subcategory`
   - `MADE_BY`: Connects `Item` to `Brand`
   - `HAS_TAG`: Connects `Item` to `Tag`
   - `BELONGS_TO`: Connects `Subcategory` to `Category`

2. **Derived Relationships**
   - `SIMILAR_TASTE`: Connects `User` to `User` with property `weight` indicating similarity strength
   - `FREQUENTLY_LIKED_TOGETHER`: Connects `Item` to `Item` with property `weight` indicating co-occurrence strength
   - `INTERESTED_IN`: Connects `User` to `Category` with property `strength` based on rating patterns

### Graph Schema Implementation

The schema is implemented in Neo4j through constraints and indices:

```cypher
// Create constraints for unique node identifiers
CREATE CONSTRAINT ON (u:User) ASSERT u.id IS UNIQUE;
CREATE CONSTRAINT ON (i:Item) ASSERT i.id IS UNIQUE;
CREATE CONSTRAINT ON (c:Category) ASSERT c.name IS UNIQUE;
CREATE CONSTRAINT ON (s:Subcategory) ASSERT s.name IS UNIQUE;
CREATE CONSTRAINT ON (b:Brand) ASSERT b.name IS UNIQUE;
CREATE CONSTRAINT ON (t:Tag) ASSERT t.name IS UNIQUE;

// Create indices for frequently queried properties
CREATE INDEX ON :User(country);
CREATE INDEX ON :User(age_group);
CREATE INDEX ON :Item(price);
CREATE INDEX ON :Item(avg_rating);
```

## Derived Relationships and Graph Enhancement

The system adds value by deriving new relationships that aren't explicitly present in the raw data:

### Similarity Relationships

User-User similarity is calculated based on common ratings:

```python
# Example from neo4j_importer.py
def create_user_similarity_relationships(self):
    """Create SIMILAR_TASTE relationships between users based on ratings."""
    self.graph.run("""
    MATCH (u1:User)-[r1:RATED]->(i:Item)<-[r2:RATED]-(u2:User)
    WHERE u1 <> u2
    WITH u1, u2, COUNT(i) AS commonItems,
         SUM(ABS(r1.rating - r2.rating)) AS ratingDiff
    WHERE commonItems > 3
    MERGE (u1)-[s:SIMILAR_TASTE]-(u2)
    SET s.weight = commonItems / (1 + ratingDiff)
    """)
```

### Co-occurrence Relationships

Item-Item relationships are derived from co-ratings:

```python
# Example from neo4j_importer.py
def create_item_cooccurrence_relationships(self):
    """Create FREQUENTLY_LIKED_TOGETHER relationships between items."""
    self.graph.run("""
    MATCH (u:User)-[r1:RATED]->(i1:Item)
    MATCH (u)-[r2:RATED]->(i2:Item)
    WHERE i1 <> i2 AND r1.rating > 3 AND r2.rating > 3
    WITH i1, i2, COUNT(u) AS userCount
    WHERE userCount > 2
    MERGE (i1)-[r:FREQUENTLY_LIKED_TOGETHER]-(i2)
    SET r.weight = userCount
    """)
```

### Hierarchical Relationships

Category relationships are structured hierarchically:

```python
# Example from neo4j_importer.py
def create_category_hierarchy(self):
    """Create hierarchical relationships between categories and subcategories."""
    self.graph.run("""
    MATCH (s:Subcategory), (c:Category)
    WHERE s.parent_category = c.name
    MERGE (s)-[:BELONGS_TO]->(c)
    """)
```

## Inference for Recommendations

The system provides multiple recommendation algorithms implemented as graph traversals:

### Collaborative Filtering

Recommendations based on similar users:

```python
# Example from graph_explorer.py
def get_collaborative_recommendations(self, user_id, limit=10):
    """Generate recommendations based on similar users' preferences."""
    query = """
    MATCH (u:User {id: $user_id})-[:SIMILAR_TASTE]-(similar:User)-[r:RATED]->(i:Item)
    WHERE r.rating > 4 AND NOT EXISTS((u)-[:RATED]->(i))
    RETURN i.id AS ItemID, i.name AS ItemName, 
           COUNT(similar) AS RecommendedByUsers, AVG(r.rating) AS AvgRating
    ORDER BY RecommendedByUsers DESC, AvgRating DESC
    LIMIT $limit
    """
    return self.graph.run(query, user_id=user_id, limit=limit).data()
```

### Content-Based Filtering

Recommendations based on item attributes:

```python
# Example from graph_explorer.py
def get_content_based_recommendations(self, user_id, limit=10):
    """Generate recommendations based on item attributes user has liked."""
    query = """
    MATCH (u:User {id: $user_id})-[r:RATED]->(i:Item)-[:IN_CATEGORY]->(c:Category)
    WITH u, c, AVG(r.rating) AS categoryInterest
    WHERE categoryInterest > 3.5
    MATCH (c)<-[:IN_CATEGORY]-(rec:Item)
    WHERE NOT EXISTS((u)-[:RATED]->(rec))
    RETURN rec.id AS ItemID, rec.name AS ItemName, 
           c.name AS Category, categoryInterest AS InterestLevel
    ORDER BY InterestLevel DESC, rec.avg_rating DESC
    LIMIT $limit
    """
    return self.graph.run(query, user_id=user_id, limit=limit).data()
```

### Hybrid Recommendations

Combining multiple recommendation approaches:

```python
# Example from graph_explorer.py
def get_hybrid_recommendations(self, user_id, limit=10):
    """Generate hybrid recommendations combining different signals."""
    query = """
    MATCH (u:User {id: $user_id})
    
    // Collaborative filtering component - similar users
    OPTIONAL MATCH (u)-[:SIMILAR_TASTE]-(similar:User)-[r1:RATED]->(i1:Item)
    WHERE r1.rating > 4 AND NOT EXISTS((u)-[:RATED]->(i1))
    WITH u, i1, COUNT(similar) * 0.5 AS collaborativeScore
    
    // Content-based component - categories
    OPTIONAL MATCH (u)-[r2:RATED]->(rated:Item)-[:IN_CATEGORY]->(c:Category)<-[:IN_CATEGORY]-(i2:Item)
    WHERE r2.rating > 3 AND NOT EXISTS((u)-[:RATED]->(i2))
    AND (i1 IS NULL OR i2.id <> i1.id)
    WITH u, i1, collaborativeScore, i2, COUNT(DISTINCT c) * 0.3 AS contentScore
    
    // Co-occurrence component - frequently liked together
    OPTIONAL MATCH (u)-[r3:RATED]->(liked:Item)-[:FREQUENTLY_LIKED_TOGETHER]-(i3:Item)
    WHERE r3.rating > 3 AND NOT EXISTS((u)-[:RATED]->(i3))
    AND (i1 IS NULL OR i3.id <> i1.id)
    AND (i2 IS NULL OR i3.id <> i2.id)
    
    // Combine scores
    WITH i1, collaborativeScore, i2, contentScore, i3, COUNT(DISTINCT liked) * 0.2 AS cooccurrenceScore
    
    // Consolidate results
    WITH COLLECT({item: i1, score: collaborativeScore, type: 'collaborative'}) +
         COLLECT({item: i2, score: contentScore, type: 'content'}) +
         COLLECT({item: i3, score: cooccurrenceScore, type: 'cooccurrence'}) AS recommendations
    UNWIND recommendations AS recommendation
    WHERE recommendation.item IS NOT NULL
    
    RETURN recommendation.item.id AS ItemID, 
           recommendation.item.name AS ItemName,
           recommendation.type AS RecommendationType,
           recommendation.score AS Score
    ORDER BY Score DESC
    LIMIT $limit
    """
    return self.graph.run(query, user_id=user_id, limit=limit).data()
```

### Path-Based Recommendations

Recommendations using graph path analysis:

```python
# Example from graph_explorer.py
def get_path_based_recommendations(self, user_id, limit=10):
    """Generate recommendations using path analysis."""
    query = """
    MATCH path = (u:User {id: $user_id})-[:RATED]->(i1:Item)-[:FREQUENTLY_LIKED_TOGETHER]-(i2:Item)
    WHERE NOT EXISTS((u)-[:RATED]->(i2))
    WITH i2, COUNT(path) AS pathCount
    RETURN i2.id AS ItemID, i2.name AS ItemName, pathCount AS RecommendationStrength
    ORDER BY pathCount DESC
    LIMIT $limit
    """
    return self.graph.run(query, user_id=user_id, limit=limit).data()
```

## Graph-Based Analytics

Beyond recommendations, the system provides analytical capabilities:

### User Analysis

```python
# Example from graph_explorer.py
def get_user_statistics(self, user_id):
    """Get detailed statistics about a specific user."""
    query = """
    MATCH (u:User {id: $user_id})
    OPTIONAL MATCH (u)-[r:RATED]->(i:Item)
    WITH u, COUNT(r) AS totalRatings, AVG(r.rating) AS avgRating
    
    OPTIONAL MATCH (u)-[:RATED]->(i:Item)-[:IN_CATEGORY]->(c:Category)
    WITH u, totalRatings, avgRating, c.name AS category, COUNT(i) AS count
    ORDER BY count DESC
    LIMIT 3
    
    RETURN u.id AS UserID, u.name AS Name, u.country AS Country,
           totalRatings, avgRating, COLLECT(category) AS TopCategories
    """
    return self.graph.run(query, user_id=user_id).data()
```

### Item Analysis

```python
# Example from graph_explorer.py
def get_item_statistics(self, item_id):
    """Get detailed statistics about a specific item."""
    query = """
    MATCH (i:Item {id: $item_id})
    OPTIONAL MATCH (u:User)-[r:RATED]->(i)
    WITH i, COUNT(r) AS totalRatings, AVG(r.rating) AS avgRating
    
    OPTIONAL MATCH (i)-[:FREQUENTLY_LIKED_TOGETHER]-(other:Item)
    WITH i, totalRatings, avgRating, other
    ORDER BY other.avg_rating DESC
    LIMIT 5
    
    RETURN i.id AS ItemID, i.name AS Name, i.main_category AS Category,
           totalRatings, avgRating, COLLECT(other.name) AS FrequentlyLikedWith
    """
    return self.graph.run(query, item_id=item_id).data()
```

### Community Detection

The system can identify communities of users with similar interests:

```python
# Example from graph_explorer.py
def detect_user_communities(self, min_community_size=3):
    """Detect communities of users with similar tastes."""
    query = """
    CALL gds.graph.project(
      'userSimilarity',
      ['User'],
      {
        SIMILAR_TASTE: {
          type: 'SIMILAR_TASTE',
          orientation: 'UNDIRECTED',
          properties: ['weight']
        }
      }
    )
    
    CALL gds.louvain.stream('userSimilarity', {
      relationshipWeightProperty: 'weight'
    })
    YIELD nodeId, communityId
    WITH gds.util.asNode(nodeId) AS user, communityId
    WITH communityId, COLLECT(user) AS users
    WHERE SIZE(users) >= $minSize
    
    RETURN communityId AS CommunityID,
           SIZE(users) AS CommunitySize,
           [user IN users | user.name] AS Members
    ORDER BY CommunitySize DESC
    """
    return self.graph.run(query, minSize=min_community_size).data()
```

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Neo4j Desktop (optional, for visualization)

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the Neo4j server using Docker:
```bash
docker-compose up -d
```

3. Import the data into Neo4j:
```bash
python neo4j_importer.py
```

## Usage

### Exploring the Knowledge Graph

You can use the included `graph_explorer.py` script to explore the knowledge graph:

```bash
# Get a summary of the graph
python graph_explorer.py --action summary

# Visualize a subgraph
python graph_explorer.py --action visualize --output graph.png

# Get recommendations for a user
python graph_explorer.py --action recommend --user_id 1 --method hybrid

# Export recommendations to CSV
python graph_explorer.py --action export
```

### Web Interface

You can also access the Neo4j Browser UI at http://localhost:7474/browser/ with the following credentials:
- Username: neo4j
- Password: password

### Example Cypher Queries

Here are some example queries you can run in the Neo4j Browser:

1. Find users with similar tastes:
```cypher
MATCH (u1:User)-[s:SIMILAR_TASTE]-(u2:User)
WHERE s.weight > 2
RETURN u1.id, u1.name, u2.id, u2.name, s.weight
ORDER BY s.weight DESC
LIMIT 10
```

2. Find items that are frequently liked together:
```cypher
MATCH (i1:Item)-[r:FREQUENTLY_LIKED_TOGETHER]-(i2:Item)
RETURN i1.name, i2.name, r.weight
ORDER BY r.weight DESC
LIMIT 10
```

3. Get recommendations for a user:
```cypher
MATCH (u:User {id: 1})-[r1:RATED]->(i1:Item)-[r:FREQUENTLY_LIKED_TOGETHER]-(i2:Item)
WHERE NOT EXISTS((u)-[:RATED]->(i2))
RETURN i2.name, i2.main_category, i2.price, r.weight
ORDER BY r.weight DESC
LIMIT 10
```

4. Find the most influential users:
```cypher
MATCH (u:User)-[r:RATED]->()
WITH u, COUNT(r) AS num_ratings
RETURN u.id, u.name, num_ratings
ORDER BY num_ratings DESC
LIMIT 10
```

5. Get popular items by category:
```cypher
MATCH (i:Item)-[:IN_CATEGORY]->(c:Category)
MATCH (u:User)-[r:RATED]->(i)
WITH c, i, COUNT(r) AS num_ratings, AVG(r.rating) AS avg_rating
WHERE num_ratings > 2
RETURN c.name AS category, i.name AS item, num_ratings, avg_rating
ORDER BY c.name, avg_rating DESC
```

## Performance Optimization

The knowledge graph implementation includes several optimizations:

1. **Index Usage** - Critical properties are indexed for faster lookups
2. **Query Optimization** - Cypher queries are structured to minimize database hits
3. **Relationship Precomputation** - Derived relationships are precomputed during import
4. **Batch Processing** - Large data operations use batched transactions
5. **Memory Management** - Query results are paginated for large result sets

## Advantages of the Knowledge Graph Approach

1. **Relationship-First Modeling** - Focus on relationships between entities rather than tables and joins
2. **Flexible Schema** - Easily add new node and relationship types without migrations
3. **Path-Based Queries** - Find complex paths between entities that would be difficult with SQL
4. **Performance on Connected Data** - Efficient traversal of connections for recommendation
5. **Built-in Graph Algorithms** - Utilize Neo4j's graph algorithms for advanced analytics

## Integrating with the Recommendation System

The knowledge graph can complement the existing recommendation system by:

1. Providing more complex recommendation paths
2. Discovering relationship patterns for better recommendations
3. Enabling visual exploration of the recommendation space
4. Supporting graph-based features for the machine learning models 