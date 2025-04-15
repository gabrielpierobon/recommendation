import pandas as pd
import os
from neo4j import GraphDatabase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jImporter:
    def __init__(self, uri, username, password):
        """Initialize the Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        logger.info("Connected to Neo4j database")
    
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
        logger.info("Closed Neo4j connection")
    
    def clear_database(self):
        """Clear all nodes and relationships in the database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")
    
    def create_constraints(self):
        """Create constraints for fast lookups"""
        with self.driver.session() as session:
            # Create constraints
            try:
                session.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
                session.run("CREATE CONSTRAINT item_id IF NOT EXISTS FOR (i:Item) REQUIRE i.id IS UNIQUE")
                session.run("CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")
                session.run("CREATE CONSTRAINT subcategory_name IF NOT EXISTS FOR (s:Subcategory) REQUIRE s.name IS UNIQUE")
                session.run("CREATE CONSTRAINT brand_name IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE")
                logger.info("Constraints created")
            except Exception as e:
                # Some versions of Neo4j use different syntax for constraints
                logger.warning(f"Error creating constraints with new syntax, trying legacy syntax: {e}")
                try:
                    session.run("CREATE CONSTRAINT ON (u:User) ASSERT u.id IS UNIQUE")
                    session.run("CREATE CONSTRAINT ON (i:Item) ASSERT i.id IS UNIQUE")
                    session.run("CREATE CONSTRAINT ON (c:Category) ASSERT c.name IS UNIQUE")
                    session.run("CREATE CONSTRAINT ON (s:Subcategory) ASSERT s.name IS UNIQUE")
                    session.run("CREATE CONSTRAINT ON (b:Brand) ASSERT b.name IS UNIQUE")
                    logger.info("Constraints created using legacy syntax")
                except Exception as e2:
                    logger.error(f"Failed to create constraints with legacy syntax: {e2}")
    
    def import_users(self, users_file):
        """Import users from CSV or Excel file"""
        logger.info(f"Importing users from {users_file}")
        
        if users_file.endswith('.csv'):
            df = pd.read_csv(users_file)
        else:
            df = pd.read_excel(users_file)
        
        # Ensure user_id is numeric
        if not pd.api.types.is_numeric_dtype(df['user_id'].dtype):
            # Skip header if present
            if df.iloc[0]['user_id'] == 'user_id':
                df = df.iloc[1:].reset_index(drop=True)
            # Convert to numeric
            df['user_id'] = pd.to_numeric(df['user_id'])
        
        # Create users
        with self.driver.session() as session:
            for _, row in df.iterrows():
                # Handle possible NaN values
                user_id = int(row['user_id'])
                properties = {col: val for col, val in row.items() if pd.notna(val)}
                
                # Ensure int values for some fields
                if 'age_group' in properties and pd.notna(properties['age_group']):
                    properties['age_group'] = str(properties['age_group'])
                
                # Convert list-like strings to actual lists
                if 'interests' in properties and isinstance(properties['interests'], str):
                    if ',' in properties['interests']:
                        properties['interests'] = [i.strip() for i in properties['interests'].split(',')]
                
                # Create user node
                cypher = """
                MERGE (u:User {id: $id})
                SET u += $properties
                RETURN u.id
                """
                result = session.run(cypher, id=user_id, properties=properties)
                result.single()
            
        logger.info(f"Imported {len(df)} users")
    
    def import_items(self, items_file):
        """Import items from CSV or Excel file"""
        logger.info(f"Importing items from {items_file}")
        
        if items_file.endswith('.csv'):
            df = pd.read_csv(items_file)
        else:
            df = pd.read_excel(items_file)
        
        # Ensure item_id is numeric
        if not pd.api.types.is_numeric_dtype(df['item_id'].dtype):
            # Skip header if present
            if df.iloc[0]['item_id'] == 'item_id':
                df = df.iloc[1:].reset_index(drop=True)
            # Convert to numeric
            df['item_id'] = pd.to_numeric(df['item_id'])
        
        # Create items
        with self.driver.session() as session:
            for _, row in df.iterrows():
                item_id = int(row['item_id'])
                properties = {col: val for col, val in row.items() if pd.notna(val) and col not in ['main_category', 'subcategory', 'brand', 'tags']}
                
                # Get category, subcategory, and brand
                category = row.get('main_category', None)
                subcategory = row.get('subcategory', None)
                brand = row.get('brand', None)
                
                # Handle tags
                tags = []
                if 'tags' in row and pd.notna(row['tags']):
                    if isinstance(row['tags'], str) and ',' in row['tags']:
                        tags = [tag.strip() for tag in row['tags'].split(',')]
                    else:
                        tags = [str(row['tags'])]
                
                # Create the item and its relationships in a single transaction
                cypher = """
                MERGE (i:Item {id: $id})
                SET i += $properties
                
                WITH i
                
                // Create and connect to category
                FOREACH (categoryName IN CASE WHEN $category IS NOT NULL THEN [$category] ELSE [] END |
                  MERGE (c:Category {name: categoryName})
                  MERGE (i)-[:IN_CATEGORY]->(c)
                )
                
                // Create and connect to subcategory
                FOREACH (subcategoryName IN CASE WHEN $subcategory IS NOT NULL THEN [$subcategory] ELSE [] END |
                  MERGE (s:Subcategory {name: subcategoryName})
                  MERGE (i)-[:IN_SUBCATEGORY]->(s)
                )
                
                // Create and connect to brand
                FOREACH (brandName IN CASE WHEN $brand IS NOT NULL THEN [$brand] ELSE [] END |
                  MERGE (b:Brand {name: brandName})
                  MERGE (i)-[:MADE_BY]->(b)
                )
                
                // Create and connect tags
                FOREACH (tagName IN $tags |
                  MERGE (t:Tag {name: tagName})
                  MERGE (i)-[:HAS_TAG]->(t)
                )
                
                RETURN i.id
                """
                
                result = session.run(cypher, 
                                    id=item_id, 
                                    properties=properties,
                                    category=category, 
                                    subcategory=subcategory,
                                    brand=brand,
                                    tags=tags)
                result.single()
                
        logger.info(f"Imported {len(df)} items")
    
    def import_ratings(self, ratings_file):
        """Import ratings from CSV or Excel file"""
        logger.info(f"Importing ratings from {ratings_file}")
        
        try:
            if ratings_file.endswith('.csv'):
                # Explicitly handle header
                df = pd.read_csv(ratings_file, header=0)
            else:
                df = pd.read_excel(ratings_file, header=0)
            
            # Convert columns to numeric, coercing errors to NaN
            df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')
            df['item_id'] = pd.to_numeric(df['item_id'], errors='coerce')
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            
            # Drop rows with NaN values
            df = df.dropna(subset=['user_id', 'item_id', 'rating'])
            
            # Create ratings
            with self.driver.session() as session:
                processed_count = 0
                error_count = 0
                
                for _, row in df.iterrows():
                    try:
                        user_id = int(row['user_id'])
                        item_id = int(row['item_id'])
                        rating = float(row['rating'])
                        
                        # Set properties
                        properties = {'rating': rating}
                        if 'timestamp' in row and pd.notna(row['timestamp']):
                            properties['timestamp'] = str(row['timestamp'])
                        
                        # Create rating relationship
                        cypher = """
                        MATCH (u:User {id: $user_id})
                        MATCH (i:Item {id: $item_id})
                        MERGE (u)-[r:RATED]->(i)
                        SET r += $properties
                        RETURN r.rating
                        """
                        
                        result = session.run(cypher, 
                                            user_id=user_id, 
                                            item_id=item_id, 
                                            properties=properties)
                        result.single()
                        processed_count += 1
                    except Exception as e:
                        error_count += 1
                        if error_count < 5:  # Limit log spam
                            logger.error(f"Error importing rating from user {row.get('user_id')} to item {row.get('item_id')}: {e}")
                        elif error_count == 5:
                            logger.error("Additional errors suppressed...")
                    
            logger.info(f"Imported {processed_count} ratings successfully (with {error_count} errors)")
        except Exception as e:
            logger.error(f"Error processing ratings file: {e}")
            raise
    
    def create_additional_relationships(self):
        """Create additional relationships between nodes based on data mining"""
        logger.info("Creating additional relationships based on data mining")
        
        with self.driver.session() as session:
            # Connect categories to subcategories
            cypher_category_subcategory = """
            MATCH (c:Category)<-[:IN_CATEGORY]-(i:Item)-[:IN_SUBCATEGORY]->(s:Subcategory)
            WITH c, s, COUNT(i) AS num_items
            WHERE num_items > 0
            MERGE (s)-[r:BELONGS_TO]->(c)
            RETURN COUNT(r) AS num_relationships
            """
            result = session.run(cypher_category_subcategory)
            num_cat_subcat = result.single()[0]
            logger.info(f"Created {num_cat_subcat} relationships between subcategories and categories")
            
            # Connect users with similar tastes (shared ratings)
            cypher_similar_users = """
            MATCH (u1:User)-[r1:RATED]->(i:Item)<-[r2:RATED]-(u2:User)
            WHERE u1.id < u2.id  // To avoid duplicate relationships
            AND r1.rating > 3 AND r2.rating > 3  // Both users liked the item
            WITH u1, u2, COUNT(i) AS num_shared_items
            WHERE num_shared_items > 1  // Require at least 2 shared liked items
            MERGE (u1)-[s:SIMILAR_TASTE]-(u2)
            SET s.weight = num_shared_items
            RETURN COUNT(s) AS num_relationships
            """
            result = session.run(cypher_similar_users)
            num_similar_users = result.single()[0]
            logger.info(f"Created {num_similar_users} similarity relationships between users")
            
            # Connect items that are commonly liked together
            cypher_similar_items = """
            MATCH (i1:Item)<-[r1:RATED]-(u:User)-[r2:RATED]->(i2:Item)
            WHERE i1.id < i2.id  // To avoid duplicate relationships
            AND r1.rating > 3 AND r2.rating > 3  // Both items were liked
            WITH i1, i2, COUNT(u) AS num_users
            WHERE num_users > 1  // Require at least 2 users who liked both
            MERGE (i1)-[s:FREQUENTLY_LIKED_TOGETHER]-(i2)
            SET s.weight = num_users
            RETURN COUNT(s) AS num_relationships
            """
            result = session.run(cypher_similar_items)
            num_similar_items = result.single()[0]
            logger.info(f"Created {num_similar_items} co-occurrence relationships between items")
    
    def create_recommendations_view(self):
        """Create a view with personalized recommendations"""
        logger.info("Creating recommendations view")
        
        with self.driver.session() as session:
            # Item-based collaborative filtering recommendations
            cypher_item_based_cf = """
            MATCH (u:User)-[r1:RATED]->(i1:Item)-[sim:FREQUENTLY_LIKED_TOGETHER]-(i2:Item)
            WHERE NOT EXISTS((u)-[:RATED]->(i2))
            WITH u, i2, SUM(r1.rating * sim.weight) AS score
            ORDER BY score DESC
            RETURN u.id AS user_id, i2.id AS item_id, i2.name AS item_name, score AS recommendation_strength
            LIMIT 100
            """
            result = session.run(cypher_item_based_cf)
            recommendations = [record for record in result]
            logger.info(f"Generated {len(recommendations)} item-based CF recommendations")
            
            # Demographic-based recommendations
            cypher_demographic = """
            MATCH (u:User)
            MATCH (i:Item)-[:IN_CATEGORY]->(:Category)
            WHERE NOT EXISTS((u)-[:RATED]->(i))
            WITH u, i, 
                CASE 
                    WHEN u.country = i.country THEN 1.0 ELSE 0.0 
                END +
                CASE 
                    WHEN u.gender = 'M' AND i.main_category IN ['Electronics', 'Sports'] THEN 0.5
                    WHEN u.gender = 'F' AND i.main_category IN ['Fashion', 'Beauty'] THEN 0.5
                    ELSE 0.0 
                END +
                CASE 
                    WHEN (u.age_group = '18-24' AND i.main_category = 'Gaming') THEN 0.5
                    WHEN (u.age_group = '25-34' AND i.main_category = 'Electronics') THEN 0.5
                    WHEN (u.age_group = '35-44' AND i.main_category = 'Home') THEN 0.5
                    ELSE 0.0 
                END AS demographic_score
            WHERE demographic_score > 0.5
            RETURN u.id AS user_id, i.id AS item_id, i.name AS item_name, demographic_score
            LIMIT 100
            """
            result = session.run(cypher_demographic)
            demographic_recs = [record for record in result]
            logger.info(f"Generated {len(demographic_recs)} demographic-based recommendations")

def main():
    """Main entry point for the script"""
    # Neo4j connection details
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "password"  # Replace with your actual password
    
    # File paths
    data_dir = "data"
    users_file = os.path.join(data_dir, "users.csv")
    items_file = os.path.join(data_dir, "items.csv")
    ratings_file = os.path.join(data_dir, "ratings.csv")
    
    # Check if CSV exists, otherwise use Excel
    if not os.path.exists(users_file):
        users_file = os.path.join(data_dir, "users.xlsx")
    if not os.path.exists(items_file):
        items_file = os.path.join(data_dir, "items.xlsx")
    if not os.path.exists(ratings_file):
        ratings_file = os.path.join(data_dir, "ratings.xlsx")
    
    # Create Neo4j importer
    try:
        importer = Neo4jImporter(uri, username, password)
        
        try:
            # Set up graph database
            importer.clear_database()
            importer.create_constraints()
            
            # Import data
            try:
                importer.import_users(users_file)
            except Exception as e:
                logger.error(f"Error importing users: {e}")
                
            try:
                importer.import_items(items_file)
            except Exception as e:
                logger.error(f"Error importing items: {e}")
                
            try:
                importer.import_ratings(ratings_file)
            except Exception as e:
                logger.error(f"Error importing ratings: {e}")
            
            # Create additional relationships
            try:
                importer.create_additional_relationships()
            except Exception as e:
                logger.error(f"Error creating additional relationships: {e}")
            
            # Create recommendations view
            try:
                importer.create_recommendations_view()
            except Exception as e:
                logger.error(f"Error creating recommendations view: {e}")
            
            logger.info("Knowledge graph successfully created!")
        except Exception as e:
            logger.error(f"Error creating knowledge graph: {e}")
        finally:
            importer.close()
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {e}")
        logger.error("Make sure Neo4j container is running (docker-compose up -d)")

if __name__ == "__main__":
    main() 