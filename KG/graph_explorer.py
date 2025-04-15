import argparse
import matplotlib.pyplot as plt
import networkx as nx
from neo4j import GraphDatabase
import logging
import pandas as pd
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jExplorer:
    def __init__(self, uri, username, password):
        """Initialize the Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        logger.info("Connected to Neo4j database")
    
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
        logger.info("Closed Neo4j connection")
    
    def get_graph_summary(self):
        """Get a summary of the graph structure"""
        with self.driver.session() as session:
            # Get node counts by label
            node_query = """
            CALL apoc.meta.stats()
            YIELD labels
            RETURN labels
            """
            try:
                result = session.run(node_query)
                labels = result.single()[0]
                
                # Format the output
                logger.info("Knowledge Graph Summary:")
                logger.info("-----------------------")
                logger.info("Node Types:")
                for label, count in labels.items():
                    logger.info(f"  - {label}: {count} nodes")
                
                # Get relationship counts
                rel_query = """
                CALL apoc.meta.stats()
                YIELD relTypes
                RETURN relTypes
                """
                result = session.run(rel_query)
                rel_types = result.single()[0]
                
                logger.info("\nRelationship Types:")
                for rel_type, count in rel_types.items():
                    logger.info(f"  - {rel_type}: {count} relationships")
                
                return labels, rel_types
            except Exception as e:
                logger.error(f"Error getting graph summary: {e}")
                logger.error("This might happen if the APOC plugin is not available in Neo4j.")
                
                # Fallback to simpler queries
                logger.info("Trying fallback queries...")
                node_counts = {}
                rel_counts = {}
                
                # Get node labels
                label_query = """
                MATCH (n) 
                RETURN DISTINCT labels(n) AS label, count(*) AS count
                """
                result = session.run(label_query)
                for record in result:
                    label = '.'.join(record["label"])
                    count = record["count"]
                    node_counts[label] = count
                
                # Get relationship types
                rel_query = """
                MATCH ()-[r]->() 
                RETURN DISTINCT type(r) AS type, count(*) AS count
                """
                result = session.run(rel_query)
                for record in result:
                    rel_type = record["type"]
                    count = record["count"]
                    rel_counts[rel_type] = count
                
                logger.info("Knowledge Graph Summary:")
                logger.info("-----------------------")
                logger.info("Node Types:")
                for label, count in node_counts.items():
                    logger.info(f"  - {label}: {count} nodes")
                
                logger.info("\nRelationship Types:")
                for rel_type, count in rel_counts.items():
                    logger.info(f"  - {rel_type}: {count} relationships")
                
                return node_counts, rel_counts
    
    def visualize_subgraph(self, query, filename="knowledge_graph_viz.png", limit=50):
        """
        Visualize a subgraph of the knowledge graph
        
        Args:
            query: Cypher query that returns nodes and relationships
            filename: Output filename for the visualization
            limit: Maximum number of nodes to include
        """
        with self.driver.session() as session:
            # Run the query
            result = session.run(query)
            
            # Create a networkx graph
            G = nx.DiGraph()
            
            # Process the results
            nodes_added = set()
            
            for record in result:
                # Skip if we've reached the limit
                if len(nodes_added) >= limit:
                    logger.info(f"Reached limit of {limit} nodes. Some data will not be shown.")
                    break
                
                # Add nodes and relationships based on the query results
                # This depends on what your query returns
                if 'n1' in record and 'n2' in record:
                    # If the query returns nodes as 'n1' and 'n2'
                    node1 = record['n1']
                    node2 = record['n2']
                    rel = record.get('r', None)
                    
                    # Add node 1
                    node1_id = f"{list(node1.labels)[0]}_{node1.get('id', node1.get('name', str(node1.id)))}"
                    if node1_id not in nodes_added:
                        G.add_node(node1_id, 
                                  label=list(node1.labels)[0],
                                  properties=dict(node1))
                        nodes_added.add(node1_id)
                    
                    # Add node 2
                    node2_id = f"{list(node2.labels)[0]}_{node2.get('id', node2.get('name', str(node2.id)))}"
                    if node2_id not in nodes_added:
                        G.add_node(node2_id, 
                                  label=list(node2.labels)[0],
                                  properties=dict(node2))
                        nodes_added.add(node2_id)
                    
                    # Add relationship
                    if rel:
                        G.add_edge(node1_id, node2_id, 
                                  label=type(rel).__name__,
                                  properties=dict(rel))
                elif 'path' in record:
                    # If the query returns a path
                    path = record['path']
                    
                    for i, node in enumerate(path.nodes):
                        node_id = f"{list(node.labels)[0]}_{node.get('id', node.get('name', str(node.id)))}"
                        if node_id not in nodes_added:
                            G.add_node(node_id, 
                                      label=list(node.labels)[0],
                                      properties=dict(node))
                            nodes_added.add(node_id)
                        
                        # Add relationship if there's a next node
                        if i < len(path.nodes) - 1:
                            next_node = path.nodes[i+1]
                            next_id = f"{list(next_node.labels)[0]}_{next_node.get('id', next_node.get('name', str(next_node.id)))}"
                            # Find the relationship between these nodes
                            for rel in path.relationships:
                                if rel.start_node.id == node.id and rel.end_node.id == next_node.id:
                                    G.add_edge(node_id, next_id,
                                             label=type(rel).__name__,
                                             properties=dict(rel))
            
            # Check if graph is empty
            if len(G) == 0:
                logger.warning("No data returned by the query or the query doesn't return graph data in the expected format.")
                return
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Set node colors based on label
            node_labels = [G.nodes[n]['label'] for n in G.nodes()]
            unique_labels = list(set(node_labels))
            color_map = plt.cm.get_cmap('tab10', len(unique_labels))
            node_colors = [unique_labels.index(label) for label in node_labels]
            
            # Create the layout
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, 
                                node_color=node_colors,
                                cmap=color_map,
                                node_size=500,
                                alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
            
            # Add labels
            labels = {}
            for node in G.nodes():
                if 'name' in G.nodes[node]['properties']:
                    labels[node] = G.nodes[node]['properties']['name']
                elif 'id' in G.nodes[node]['properties']:
                    labels[node] = f"{G.nodes[node]['label']} {G.nodes[node]['properties']['id']}"
                else:
                    labels[node] = node
            nx.draw_networkx_labels(G, pos, labels, font_size=10)
            
            # Add edge labels
            edge_labels = {}
            for u, v, data in G.edges(data=True):
                edge_labels[(u, v)] = data['label']
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
            
            # Add a legend
            legend_elements = []
            for i, label in enumerate(unique_labels):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=color_map(i), markersize=10, label=label))
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.title('Knowledge Graph Visualization')
            plt.axis('off')
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {filename}")
            
            # Close the plot
            plt.close()
    
    def get_user_recommendations(self, user_id, method="hybrid", limit=10):
        """
        Get recommendations for a user
        
        Args:
            user_id: The user ID to get recommendations for
            method: The recommendation method to use (collaborative, content, demographic, or hybrid)
            limit: The maximum number of recommendations to return
        """
        with self.driver.session() as session:
            if method == "collaborative":
                # Collaborative filtering recommendations
                query = """
                MATCH (u:User {id: $user_id})-[r1:RATED]->(i1:Item)-[sim:FREQUENTLY_LIKED_TOGETHER]-(i2:Item)
                WHERE NOT EXISTS((u)-[:RATED]->(i2))
                WITH i2, SUM(r1.rating * sim.weight) AS score
                ORDER BY score DESC
                LIMIT $limit
                RETURN i2.id AS item_id, i2.name AS item_name, score
                """
                result = session.run(query, user_id=user_id, limit=limit)
                
            elif method == "content":
                # Content-based recommendations
                query = """
                MATCH (u:User {id: $user_id})-[r:RATED]->(i1:Item)-[:IN_CATEGORY]->(c:Category)<-[:IN_CATEGORY]-(i2:Item)
                WHERE NOT EXISTS((u)-[:RATED]->(i2))
                WITH i2, COUNT(DISTINCT c) AS category_matches, AVG(r.rating) AS avg_rating
                ORDER BY category_matches * avg_rating DESC
                LIMIT $limit
                RETURN i2.id AS item_id, i2.name AS item_name, category_matches, avg_rating
                """
                result = session.run(query, user_id=user_id, limit=limit)
                
            elif method == "demographic":
                # Demographic-based recommendations
                query = """
                MATCH (u:User {id: $user_id})
                MATCH (i:Item)
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
                RETURN i.id AS item_id, i.name AS item_name, demographic_score
                ORDER BY demographic_score DESC
                LIMIT $limit
                """
                result = session.run(query, user_id=user_id, limit=limit)
                
            else:  # hybrid
                # Hybrid recommendations
                query = """
                // Collaborative filtering
                MATCH (u:User {id: $user_id})-[r1:RATED]->(i1:Item)-[sim:FREQUENTLY_LIKED_TOGETHER]-(i2:Item)
                WHERE NOT EXISTS((u)-[:RATED]->(i2))
                WITH i2, SUM(r1.rating * sim.weight) * 0.6 AS cf_score
                
                // Content-based
                OPTIONAL MATCH (u:User {id: $user_id})-[r:RATED]->(i:Item)-[:IN_CATEGORY]->(c:Category)<-[:IN_CATEGORY]-(i2)
                WITH i2, cf_score, COUNT(DISTINCT c) * AVG(coalesce(r.rating, 3)) * 0.3 AS cb_score
                
                // Demographic
                MATCH (u:User {id: $user_id})
                WITH u, i2, cf_score, cb_score
                
                WITH i2, cf_score, cb_score,
                    CASE 
                        WHEN u.country = i2.country THEN 1.0 ELSE 0.0 
                    END +
                    CASE 
                        WHEN u.gender = 'M' AND i2.main_category IN ['Electronics', 'Sports'] THEN 0.5
                        WHEN u.gender = 'F' AND i2.main_category IN ['Fashion', 'Beauty'] THEN 0.5
                        ELSE 0.0 
                    END +
                    CASE 
                        WHEN (u.age_group = '18-24' AND i2.main_category = 'Gaming') THEN 0.5
                        WHEN (u.age_group = '25-34' AND i2.main_category = 'Electronics') THEN 0.5
                        WHEN (u.age_group = '35-44' AND i2.main_category = 'Home') THEN 0.5
                        ELSE 0.0 
                    END * 0.1 AS demo_score
                
                // Combine scores
                WITH i2, cf_score + cb_score + demo_score AS total_score
                WHERE total_score > 0
                RETURN i2.id AS item_id, i2.name AS item_name, total_score
                ORDER BY total_score DESC
                LIMIT $limit
                """
                result = session.run(query, user_id=user_id, limit=limit)
            
            # Process the results
            recommendations = []
            for record in result:
                recommendations.append(dict(record))
            
            # Log the recommendations
            logger.info(f"Found {len(recommendations)} recommendations for user {user_id} using {method} method")
            for i, rec in enumerate(recommendations):
                logger.info(f"{i+1}. {rec['item_name']} (ID: {rec['item_id']}, Score: {rec.get('total_score', rec.get('score', rec.get('demographic_score', 'N/A')))})")
            
            return recommendations
    
    def export_recommendations_to_csv(self, user_ids=None, methods=None, output_dir="recommendations"):
        """
        Export recommendations for multiple users to CSV files
        
        Args:
            user_ids: List of user IDs to export recommendations for. If None, export for all users.
            methods: List of recommendation methods to use. If None, use all methods.
            output_dir: Directory to save the CSV files
        """
        if methods is None:
            methods = ["collaborative", "content", "demographic", "hybrid"]
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all user IDs if not specified
        if user_ids is None:
            with self.driver.session() as session:
                result = session.run("MATCH (u:User) RETURN u.id AS user_id")
                user_ids = [record["user_id"] for record in result]
        
        # Export recommendations for each user and method
        for method in methods:
            all_recommendations = []
            
            for user_id in user_ids:
                try:
                    recommendations = self.get_user_recommendations(user_id, method=method)
                    
                    # Add user_id to each recommendation
                    for rec in recommendations:
                        rec["user_id"] = user_id
                    
                    all_recommendations.extend(recommendations)
                except Exception as e:
                    logger.error(f"Error getting {method} recommendations for user {user_id}: {e}")
            
            # Export to CSV
            if all_recommendations:
                df = pd.DataFrame(all_recommendations)
                output_file = os.path.join(output_dir, f"{method}_recommendations.csv")
                df.to_csv(output_file, index=False)
                logger.info(f"Exported {len(all_recommendations)} {method} recommendations to {output_file}")
            else:
                logger.warning(f"No {method} recommendations found to export")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Neo4j Knowledge Graph Explorer")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--username", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    parser.add_argument("--action", choices=["summary", "visualize", "recommend", "export"], 
                       default="summary", help="Action to perform")
    parser.add_argument("--query", help="Cypher query for visualization")
    parser.add_argument("--output", default="knowledge_graph_viz.png", help="Output filename for visualization")
    parser.add_argument("--user_id", type=int, help="User ID for recommendations")
    parser.add_argument("--method", choices=["collaborative", "content", "demographic", "hybrid"],
                       default="hybrid", help="Recommendation method")
    parser.add_argument("--limit", type=int, default=10, help="Limit for recommendations or visualization nodes")
    
    return parser.parse_args()

def main():
    """Main entry point for the script"""
    args = parse_args()
    
    # Create Neo4j explorer
    explorer = Neo4jExplorer(args.uri, args.username, args.password)
    
    try:
        if args.action == "summary":
            explorer.get_graph_summary()
            
        elif args.action == "visualize":
            if args.query:
                explorer.visualize_subgraph(args.query, args.output, args.limit)
            else:
                # Use a default query if none is provided
                default_query = """
                MATCH path = (n1)-[r]->(n2)
                RETURN path, n1, n2, r
                LIMIT 50
                """
                logger.info("No query provided. Using default query:")
                logger.info(default_query)
                explorer.visualize_subgraph(default_query, args.output, args.limit)
                
        elif args.action == "recommend":
            if args.user_id:
                explorer.get_user_recommendations(args.user_id, args.method, args.limit)
            else:
                logger.error("User ID is required for recommendations")
                
        elif args.action == "export":
            explorer.export_recommendations_to_csv(
                user_ids=None if not args.user_id else [args.user_id],
                methods=[args.method] if args.method != "hybrid" else None
            )
            
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        explorer.close()

if __name__ == "__main__":
    main() 