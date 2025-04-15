# Neo4j Cypher Commands Cheatsheet

This document provides a collection of useful Cypher queries to explore and interact with the recommendation system knowledge graph, organized from simplest to most advanced.

## Basic Queries

### View All Nodes

```cypher
MATCH (n) 
RETURN n 
LIMIT 100;
```

### Count Nodes by Type

```cypher
MATCH (n)
RETURN labels(n) AS NodeType, count(n) AS Count
ORDER BY Count DESC;
```

### View All Users

```cypher
MATCH (u:User)
RETURN u
LIMIT 20;
```

### View All Items

```cypher
MATCH (i:Item)
RETURN i
LIMIT 20;
```

### View Specific Node by ID

```cypher
// Find user with ID 1
MATCH (u:User {id: 1})
RETURN u;

// Find item with ID 30
MATCH (i:Item {id: 30})
RETURN i;
```

### View All Relationships

```cypher
MATCH ()-[r]->()
RETURN type(r) AS RelationshipType, count(r) AS Count
ORDER BY Count DESC;
```

## Intermediate Queries

### Find User Ratings

```cypher
// Find all ratings by a specific user
MATCH (u:User {id: 1})-[r:RATED]->(i:Item)
RETURN u.id AS User, i.id AS Item, i.name AS ItemName, r.rating AS Rating
ORDER BY r.rating DESC;
```

### Find Item Ratings

```cypher
// Find all ratings for a specific item
MATCH (u:User)-[r:RATED]->(i:Item {id: 30})
RETURN u.id AS User, u.country AS Country, u.age_group AS AgeGroup, r.rating AS Rating
ORDER BY r.rating DESC;
```

### Find Users with Similar Taste

```cypher
MATCH (u1:User {id: 1})-[r:SIMILAR_TASTE]-(u2:User)
RETURN u1.id AS User1, u2.id AS User2, u2.country AS Country, r.weight AS SimilarityScore
ORDER BY r.weight DESC;
```

### Find Items Frequently Liked Together

```cypher
MATCH (i1:Item {id: 30})-[r:FREQUENTLY_LIKED_TOGETHER]-(i2:Item)
RETURN i1.name AS Item1, i2.id AS Item2ID, i2.name AS Item2, r.weight AS CooccurrenceScore
ORDER BY r.weight DESC;
```

### Find Items by Category

```cypher
MATCH (i:Item)-[:IN_CATEGORY]->(c:Category {name: 'Electronics'})
RETURN i.id AS ItemID, i.name AS ItemName, i.price AS Price
ORDER BY i.price DESC
LIMIT 20;
```

### Find Items by Multiple Properties

```cypher
MATCH (i:Item)
WHERE i.price > 500 AND i.avg_rating > 4
RETURN i.id AS ItemID, i.name AS ItemName, i.price AS Price, i.avg_rating AS Rating
ORDER BY i.price DESC;
```

## Advanced Queries

### Generate Recommendations Based on Similar Users

```cypher
MATCH (u:User {id: 1})-[:SIMILAR_TASTE]-(similar:User)-[r:RATED]->(i:Item)
WHERE r.rating > 4 AND NOT EXISTS((u)-[:RATED]->(i))
RETURN i.id AS ItemID, i.name AS ItemName, count(similar) AS RecommendedByUsers, avg(r.rating) AS AvgRating
ORDER BY RecommendedByUsers DESC, AvgRating DESC
LIMIT 10;
```

### Path-Based Recommendations

```cypher
MATCH path = (u:User {id: 1})-[:RATED]->(i1:Item)-[:FREQUENTLY_LIKED_TOGETHER]-(i2:Item)
WHERE NOT EXISTS((u)-[:RATED]->(i2))
RETURN DISTINCT i2.id AS RecommendedItemID, i2.name AS RecommendedItem, 
       count(path) AS RecommendationStrength
ORDER BY RecommendationStrength DESC
LIMIT 10;
```

### Category-Based Recommendations

```cypher
MATCH (u:User {id: 1})-[:RATED]->(i:Item)-[:IN_CATEGORY]->(c:Category)
MATCH (i2:Item)-[:IN_CATEGORY]->(c)
WHERE NOT EXISTS((u)-[:RATED]->(i2))
      AND i2.id <> i.id
RETURN i2.id AS ItemID, i2.name AS ItemName, c.name AS Category, count(i) AS CategoryRelevance
ORDER BY CategoryRelevance DESC
LIMIT 15;
```

### Tag-Based Recommendations

```cypher
MATCH (u:User {id: 1})-[:RATED]->(i:Item)-[:HAS_TAG]->(t:Tag)
MATCH (i2:Item)-[:HAS_TAG]->(t)
WHERE NOT EXISTS((u)-[:RATED]->(i2))
      AND i2.id <> i.id
RETURN i2.id AS ItemID, i2.name AS ItemName, 
       collect(DISTINCT t.name) AS SharedTags, 
       count(DISTINCT t) AS TagMatch
ORDER BY TagMatch DESC
LIMIT 15;
```

### Hybrid Recommendations 

```cypher
MATCH (u:User {id: 1})
OPTIONAL MATCH (u)-[:RATED]->(i1:Item)-[:FREQUENTLY_LIKED_TOGETHER]-(i2:Item)
WHERE NOT EXISTS((u)-[:RATED]->(i2))
WITH u, i2, sum(coalesce(i1.rating, 0)) * 0.6 AS collab_score

OPTIONAL MATCH (u)-[:RATED]->(i3:Item)-[:IN_CATEGORY]->(:Category)<-[:IN_CATEGORY]-(i4:Item)
WHERE i4 <> i3 AND i4 <> i2 AND NOT EXISTS((u)-[:RATED]->(i4))
WITH u, i2, collab_score, sum(coalesce(i3.rating, 0)) * 0.3 AS content_score, i4

RETURN DISTINCT 
    CASE WHEN i2 IS NOT NULL THEN i2.id ELSE i4.id END AS item_id,
    CASE WHEN i2 IS NOT NULL THEN i2.name ELSE i4.name END AS item_name,
    coalesce(collab_score, 0) + coalesce(content_score, 0) AS score
ORDER BY score DESC
LIMIT 10;
```

## Graph Algorithms

### Find Similar Users with Node Similarity

```cypher
CALL gds.nodeSimilarity.stream({
  nodeProjection: 'User',
  relationshipProjection: {
    RATED: {
      type: 'RATED',
      orientation: 'NATURAL'
    }
  }
})
YIELD node1, node2, similarity
WITH gds.util.asNode(node1) AS user1, gds.util.asNode(node2) AS user2, similarity
WHERE user1.id = 1 AND similarity > 0.1
RETURN user1.id AS User1, user2.id AS User2, similarity AS SimilarityScore
ORDER BY SimilarityScore DESC
LIMIT 10;
```

### Find Communities of Users

```cypher
CALL gds.louvain.stream({
  nodeProjection: 'User',
  relationshipProjection: {
    SIMILAR_TASTE: {
      type: 'SIMILAR_TASTE',
      orientation: 'UNDIRECTED',
      properties: {
        weight: {
          property: 'weight',
          defaultValue: 1.0
        }
      }
    }
  },
  relationshipWeightProperty: 'weight'
})
YIELD nodeId, communityId
WITH gds.util.asNode(nodeId) AS user, communityId
RETURN communityId AS Community, count(*) AS UserCount, 
       collect(user.id) AS UserIds
ORDER BY UserCount DESC
LIMIT 10;
```

### Find Influential Items with PageRank

```cypher
CALL gds.pageRank.stream({
  nodeProjection: ['User', 'Item'],
  relationshipProjection: {
    RATED: {
      type: 'RATED',
      orientation: 'NATURAL',
      properties: {
        weight: {
          property: 'rating',
          defaultValue: 1.0
        }
      }
    }
  },
  relationshipWeightProperty: 'weight'
})
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS node, score
WHERE node:Item
RETURN node.id AS ItemId, node.name AS ItemName, score AS InfluenceScore
ORDER BY score DESC
LIMIT 20;
```

## Database Maintenance

### View Database Schema

```cypher
CALL db.schema.visualization();
```

### View Constraints

```cypher
CALL db.constraints();
```

### View Indexes

```cypher
CALL db.indexes();
```

### Memory Usage

```cypher
CALL db.stats.retrieve("MEMORY_POOL");
```

## Visualization Queries

### User and Their Rated Items

```cypher
// Visualize a user and all items they've rated
MATCH (u:User {id: 1})-[r:RATED]->(i:Item)
RETURN u, r, i;
```

### User, Their Likes, and Similar Users

```cypher
// Visualize a user, their highly-rated items, and users with similar taste
MATCH (u:User {id: 1})-[r:RATED]->(i:Item)
WHERE r.rating >= 4
WITH u, i
MATCH (u)-[s:SIMILAR_TASTE]-(similar:User)
RETURN u, r, i, s, similar
LIMIT 20;
```

### Exploration of Items Liked by Similar Users

```cypher
// Visualize items liked by users with similar taste
MATCH (u:User {id: 1})-[s:SIMILAR_TASTE]-(similar:User)-[r:RATED]->(i:Item)
WHERE r.rating > 4 AND NOT EXISTS((u)-[:RATED]->(i))
RETURN u, s, similar, r, i
LIMIT 25;
```

### Recommendation Path Visualization

```cypher
// Visualize the path from a user to recommended items through similar users
MATCH path = (u:User {id: 1})-[:SIMILAR_TASTE]-(similar:User)-[r:RATED]->(i:Item)
WHERE r.rating > 4 AND NOT EXISTS((u)-[:RATED]->(i))
RETURN path
LIMIT 15;
```

### Co-purchased Items Network

```cypher
// Visualize the network of frequently liked together items
MATCH (i1:Item)<-[:RATED]-(u:User {id: 1})
MATCH (i1)-[r:FREQUENTLY_LIKED_TOGETHER]-(i2:Item)
RETURN i1, r, i2
LIMIT 20;
```

### Category-based User Interest Map

```cypher
// Visualize user's interest across categories
MATCH (u:User {id: 1})-[:RATED]->(i:Item)-[:IN_CATEGORY]->(c:Category)
RETURN u, i, c
LIMIT 30;
```

### Multi-level Interest Exploration

```cypher
// Visualize connections across multiple levels - user to categories to other items to other users
MATCH path = (u:User {id: 1})-[:RATED]->(i1:Item)-[:IN_CATEGORY]->(c:Category)<-[:IN_CATEGORY]-(i2:Item)<-[:RATED]-(u2:User)
WHERE i1 <> i2 AND u <> u2 AND NOT EXISTS((u)-[:RATED]->(i2))
RETURN path
LIMIT 15;
```

## Inference and Advanced Recommendation Queries

### Personalized Multi-factor Recommendations

```cypher
// Recommendation combining user similarity, item co-occurrence, and category preferences
MATCH (u:User {id: 1})

// Similar user recommendations (collaborative filtering)
OPTIONAL MATCH (u)-[:SIMILAR_TASTE]->(similar:User)-[r1:RATED]->(i1:Item)
WHERE r1.rating > 4 AND NOT EXISTS((u)-[:RATED]->(i1))
WITH u, i1, count(similar) * 0.5 AS similarUserScore

// Co-occurrence based recommendations
OPTIONAL MATCH (u)-[:RATED]->(rated:Item)-[:FREQUENTLY_LIKED_TOGETHER]->(i2:Item)
WHERE NOT EXISTS((u)-[:RATED]->(i2))
AND (i1 IS NULL OR i2.id <> i1.id)
WITH u, i1, similarUserScore, i2, count(rated) * 0.3 AS coOccurrenceScore

// Category preference based recommendations
OPTIONAL MATCH (u)-[:RATED]->(ci:Item)-[:IN_CATEGORY]->(c:Category)<-[:IN_CATEGORY]-(i3:Item)
WHERE NOT EXISTS((u)-[:RATED]->(i3))
AND (i1 IS NULL OR i3.id <> i1.id)
AND (i2 IS NULL OR i3.id <> i2.id)
WITH 
  i1, similarUserScore, 
  i2, coOccurrenceScore, 
  i3, count(ci) * 0.2 AS categoryScore

// Combine all recommendations with their scores
WITH 
  COLLECT({item: i1, score: similarUserScore, type: 'Similar Users'}) +
  COLLECT({item: i2, score: coOccurrenceScore, type: 'Co-occurrence'}) +
  COLLECT({item: i3, score: categoryScore, type: 'Category'}) AS recommendations
UNWIND recommendations AS recommendation
WHERE recommendation.item IS NOT NULL
RETURN 
  recommendation.item.id AS ItemID,
  recommendation.item.name AS ItemName,
  recommendation.type AS RecommendationType,
  recommendation.score AS Score
ORDER BY Score DESC
LIMIT 15;
```

### Customer Journey Recommendations

```cypher
// Recommend items based on typical customer journey/purchase sequences
MATCH (u:User {id: 1})-[:RATED]->(latest:Item)
WITH u, latest ORDER BY latest.release_date DESC LIMIT 1
MATCH p = (latest)<-[:RATED]-(:User)-[:RATED]->(next:Item)
WHERE NOT EXISTS((u)-[:RATED]->(next))
WITH next, count(p) AS frequency, collect(p) AS paths
RETURN next.id AS ItemID, next.name AS ItemName, 
       next.price AS Price, frequency AS Frequency
ORDER BY frequency DESC
LIMIT 10;
```

### Multi-hop Knowledge Graph Recommendations

```cypher
// Find recommendations through multiple relationship hops
MATCH path = (u:User {id: 1})-[:RATED]->()-[:IN_CATEGORY]->()<-[:IN_CATEGORY]-()<-[:RATED]-(similar:User)-[:RATED]->(rec:Item)
WHERE NOT EXISTS((u)-[:RATED]->(rec))
AND similar.id <> u.id
WITH rec, count(DISTINCT path) AS pathCount
RETURN rec.id AS ItemID, rec.name AS ItemName, pathCount AS RelevanceScore
ORDER BY pathCount DESC
LIMIT 10;
```

### Time-aware Recommendations

```cypher
// Recommend items based on recency and seasonality
MATCH (u:User {id: 1})-[:RATED]->(i:Item)
WITH u, max(i.release_date) AS latestInteraction
MATCH (newer:Item)
WHERE newer.release_date > latestInteraction
AND NOT EXISTS((u)-[:RATED]->(newer))
WITH newer
// Optionally consider season/month for seasonal recommendations
WHERE date().month >= 9 AND date().month <= 12 // Fall/Winter season
RETURN newer.id AS ItemID, newer.name AS ItemName, 
       newer.release_date AS ReleaseDate
ORDER BY newer.release_date DESC
LIMIT 10;
```

### Recommendations for New Users (Cold Start)

```cypher
// Recommend popular items for users with few or no ratings
MATCH (u:User {id: 1})
OPTIONAL MATCH (u)-[r:RATED]->(i:Item)
WITH u, count(r) AS ratingCount
WHERE ratingCount < 3
MATCH (pop:Item)
WHERE pop.avg_rating > 4 AND pop.num_ratings > 50
AND NOT EXISTS((u)-[:RATED]->(pop))
RETURN pop.id AS ItemID, pop.name AS ItemName, 
       pop.avg_rating AS Rating, pop.num_ratings AS NumberOfRatings
ORDER BY pop.avg_rating * log10(pop.num_ratings) DESC
LIMIT 15;
```

### Brand Affinity Recommendations

```cypher
// Recommend items based on brand preferences
MATCH (u:User {id: 1})-[:RATED]->(i:Item)-[:BY_BRAND]->(b:Brand)
WITH u, b, count(i) AS brandInteractions, avg(i.avg_rating) AS brandSatisfaction
ORDER BY brandInteractions * brandSatisfaction DESC
LIMIT 3
MATCH (b)<-[:BY_BRAND]-(rec:Item)
WHERE NOT EXISTS((u)-[:RATED]->(rec))
RETURN rec.id AS ItemID, rec.name AS ItemName, b.name AS Brand,
       brandInteractions AS BrandInteractions, brandSatisfaction AS BrandSatisfaction
ORDER BY brandInteractions * brandSatisfaction DESC, rec.avg_rating DESC
LIMIT 10;
```

### Preference-based Inference

```cypher
// Infer price sensitivity based on rating patterns
MATCH (u:User {id: 1})-[r:RATED]->(i:Item)
WITH u, avg(i.price) AS avgPrice, stddev(i.price) AS priceDeviation,
     avg(CASE WHEN r.rating >= 4 THEN i.price ELSE 0 END) AS preferredPrice
MATCH (rec:Item)
WHERE rec.price BETWEEN (preferredPrice - priceDeviation) AND (preferredPrice + priceDeviation)
AND NOT EXISTS((u)-[:RATED]->(rec))
RETURN rec.id AS ItemID, rec.name AS ItemName, rec.price AS Price,
       abs(rec.price - preferredPrice) AS PriceMatch
ORDER BY rec.avg_rating DESC, PriceMatch
LIMIT 10;
```

### Cross-category Recommendations

```cypher
// Recommend items from categories complementary to those the user has already purchased
MATCH (u:User {id: 1})-[:RATED]->(i:Item)-[:IN_CATEGORY]->(c:Category)
WITH u, collect(DISTINCT c.name) AS userCategories
MATCH (c1:Category)-[:FREQUENTLY_PURCHASED_WITH]->(c2:Category)
WHERE c1.name IN userCategories AND NOT c2.name IN userCategories
WITH u, c2, count(c1) AS categoryRelevance
ORDER BY categoryRelevance DESC
LIMIT 3
MATCH (c2)<-[:IN_CATEGORY]-(rec:Item)
WHERE NOT EXISTS((u)-[:RATED]->(rec))
RETURN rec.id AS ItemID, rec.name AS ItemName, c2.name AS NewCategory,
       categoryRelevance AS Relevance, rec.avg_rating AS Rating
ORDER BY categoryRelevance DESC, rec.avg_rating DESC
LIMIT 10;
```

### Seasonal Trend Recommendations

```cypher
// Recommend trending items that match user preferences
MATCH (u:User {id: 1})-[:RATED]->(i:Item)-[:IN_CATEGORY]->(c:Category)
WITH u, collect(DISTINCT c.name) AS userCategories
MATCH (trend:Item)
WHERE trend.release_date > date().epochSeconds - 7776000 // Last 90 days
AND EXISTS((trend)-[:IN_CATEGORY]->(:Category {name: userCategories}))
AND trend.num_ratings > 20
AND NOT EXISTS((u)-[:RATED]->(trend))
RETURN trend.id AS ItemID, trend.name AS ItemName, 
       trend.avg_rating AS Rating, trend.release_date AS ReleaseDate
ORDER BY trend.num_ratings * trend.avg_rating DESC
LIMIT 10;
```

### Social Influence-based Recommendations

```cypher
// Recommend items popular among influencers in the network
CALL gds.pageRank.stream({
  nodeProjection: 'User',
  relationshipProjection: {
    SIMILAR_TASTE: {
      type: 'SIMILAR_TASTE',
      orientation: 'UNDIRECTED',
      properties: {
        weight: {
          property: 'weight',
          defaultValue: 1.0
        }
      }
    }
  },
  relationshipWeightProperty: 'weight'
})
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS influencer, score
WHERE score > 0.5
MATCH (u:User {id: 1})
MATCH (influencer)-[:RATED]->(i:Item)
WHERE influencer.id <> u.id
AND NOT EXISTS((u)-[:RATED]->(i))
RETURN i.id AS ItemID, i.name AS ItemName, 
       count(DISTINCT influencer) AS InfluencerCount,
       score AS InfluencerScore
ORDER BY InfluencerCount * score DESC
LIMIT 10;
```

## Tips for Exploring the Knowledge Graph

1. **Start Small**: Begin with simple queries that return a limited number of nodes before expanding
2. **Use Graph View**: In Neo4j Browser, click the graph icon to visualize your query results
3. **Save Queries**: Save frequently used queries in Neo4j Browser for later use
4. **Use Parameters**: For repeated queries with different values, use parameters like `$userId` instead of hardcoded values
5. **Check Performance**: Use `PROFILE` before a query to see execution plans and identify bottlenecks
6. **Use LIMIT**: Always use LIMIT when exploring to avoid overwhelming the browser with too many results 