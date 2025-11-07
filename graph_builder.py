"""
Graph Builder
=============

Constructs knowledge graphs from processed text chunks using LLMGraphTransformer.
Handles Neo4j database operations and graph enrichment.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") 
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class GraphBuilder:
    """Handles knowledge graph construction and enrichment."""
    
    def __init__(self):
        self.graph = None
        self.llm = None
        self.transformer = None
        
    def connect_to_neo4j(self):
        """Connect to Neo4j database."""
        try:
            self.graph = Neo4jGraph(
                url=NEO4J_URL,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD
            )
            print("✅ Connected to Neo4j database")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def setup_llm_transformer(self, model="gpt-3.5-turbo"):
        """Setup LLM and graph transformer with correct prompt template."""
        try:
            self.llm = ChatOpenAI(temperature=0, model=model)
            
            self.transformer = LLMGraphTransformer(
                llm=self.llm,
                strict_mode=False,
                node_properties=["description", "latitude", "longitude", "family", 
                               "subfamily", "genus", "country_id", "iso_code", "macroarea"],
                relationship_properties=[]
            )
            print(f"✅ LLM Graph Transformer ready (model: {model})")
            return True
        except Exception as e:
            print(f"❌ Failed to setup LLM transformer: {e}")
            return False
    
    def clear_graph(self):
        """Clear all data from the graph."""
        if not self.graph:
            print("❌ No graph connection")
            return False
            
        try:
            result = self.graph.query("MATCH (n) RETURN count(n) as count")
            old_count = result[0]['count'] if result else 0
            
            self.graph.query("MATCH (n) DETACH DELETE n")
            print(f"🗑️  Cleared {old_count} nodes from database")
            return True
        except Exception as e:
            print(f"❌ Error clearing graph: {e}")
            return False
    
    def _fix_entity_ids(self, graph_documents):
        """Fix problematic entity IDs like 'ID' to avoid conflicts."""
        for doc in graph_documents:
            for node in doc.nodes:
                # Fix ambiguous 'ID' entities 
                if hasattr(node, 'id') and node.id == 'ID':
                    # Check if it's referring to Indonesia
                    if any(prop in str(node.properties).lower() for prop in ['indonesia', 'indonesian']):
                        node.id = 'Indonesia_Country'
                    else:
                        node.id = f'Entity_ID_{hash(str(node.properties)) % 10000}'
        return graph_documents
    
    def build_from_chunks(self, chunk_files, preserve_existing=False):
        """Build knowledge graph from text chunks."""
        if not self.graph or not self.transformer:
            print("❌ Graph or transformer not initialized")
            return 0, 0
        
        if not preserve_existing:
            self.clear_graph()
        
        total_nodes = 0
        total_relationships = 0
        
        print(f"🏗️  Building graph from {len(chunk_files)} chunks...")
        
        with tqdm(total=len(chunk_files), desc="Processing chunks") as pbar:
            for chunk_file in chunk_files:
                try:
                    # Read chunk content
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Extract graph elements
                    documents = [Document(page_content=text)]
                    graph_documents = self.transformer.convert_to_graph_documents(documents)
                    
                    # Fix problematic entity IDs
                    graph_documents = self._fix_entity_ids(graph_documents)
                    
                    if graph_documents and graph_documents[0].nodes:
                        # Add to graph with merge strategy for duplicates
                        try:
                            self.graph.add_graph_documents(
                                graph_documents, 
                                baseEntityLabel="Language",
                                include_source=False
                            )
                        except Exception as merge_error:
                            # If merge fails, try manual processing to handle duplicates
                            print(f"⚠️  Merge issue in {os.path.basename(chunk_file)}, skipping...")
                            continue
                        
                        nodes_added = len(graph_documents[0].nodes)
                        rels_added = len(graph_documents[0].relationships)
                        
                        total_nodes += nodes_added
                        total_relationships += rels_added
                        
                        pbar.set_postfix({
                            'Nodes': total_nodes,
                            'Rels': total_relationships
                        })
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"❌ Error processing {chunk_file}: {e}")
                    continue
        
        print(f"✅ Graph built: {total_nodes} nodes, {total_relationships} relationships")
        return total_nodes, total_relationships
    
    def build_from_enhanced_chunks(self):
        """Build graph from enhanced chunks with fallback to regular chunks."""
        if not self.graph or not self.transformer:
            print("❌ Graph or transformer not initialized")
            return 0, 0
        
        # Clear existing graph
        self.clear_graph()
        
        # Try to find enhanced chunk files
        chunk_files = []
        
        # Priority order for chunk file locations
        chunk_locations = [
            'output/logs/enhanced_chunk_list.txt',
            'output/logs/chunk_list.txt'
        ]
        
        chunk_list_file = None
        for location in chunk_locations:
            if os.path.exists(location):
                chunk_list_file = location
                break
        
        if chunk_list_file:
            print(f"📋 Using chunk list: {chunk_list_file}")
            with open(chunk_list_file, 'r') as f:
                chunk_files = [line.strip() for line in f if line.strip()]
            
            # Verify files exist
            chunk_files = [f for f in chunk_files if os.path.exists(f)]
        
        # Fallback: look for chunk files directly
        if not chunk_files:
            import glob
            patterns = [
                'output/chunks/enhanced_chunk_*.txt',
                'output/chunks/*.txt'
            ]
            
            for pattern in patterns:
                files = glob.glob(pattern)
                if files:
                    chunk_files = files
                    break
        
        if not chunk_files:
            print("❌ No chunk files found!")
            return 0, 0
        
        print(f"🏗️  Building graph from {len(chunk_files)} enhanced chunks...")
        return self.build_from_chunks(chunk_files)
    
    def complete_missing_languages(self):
        """Complete the graph with any missing languages from CSV."""
        print("🔄 Completing graph with missing languages...")
        
        try:
            if not os.path.exists('data/languages.csv'):
                print("❌ data/languages.csv not found")
                return False
            
            languages_df = pd.read_csv('data/languages.csv')
            
            # Get existing languages in graph
            existing_result = self.graph.query("MATCH (l:Language) RETURN l.id as name")
            existing_languages = {row['name'] for row in existing_result}
            
            # Find missing languages
            all_languages = set(languages_df['Name'].astype(str))
            missing_languages = all_languages - existing_languages
            
            if not missing_languages:
                print("✅ No missing languages - graph is complete!")
                return True
            
            print(f"🔍 Found {len(missing_languages)} missing languages")
            print(f"📊 Coverage before: {len(existing_languages)}/{len(all_languages)} ({len(existing_languages)/len(all_languages)*100:.1f}%)")
            
            # Add missing languages
            added_count = 0
            for _, lang in tqdm(languages_df.iterrows(), desc="Checking languages", total=len(languages_df)):
                name = str(lang['Name'])
                
                if name in missing_languages:
                    # Create language node with all available data
                    family = str(lang.get('Family', 'Unknown'))
                    subfamily = str(lang.get('Subfamily', ''))
                    genus = str(lang.get('Genus', 'Unknown'))
                    country_id = str(lang.get('Country_ID', ''))
                    iso_code = str(lang.get('ISO639P3code', ''))
                    macroarea = str(lang.get('Macroarea', ''))
                    
                    # Clean name for Cypher
                    clean_name = name.replace('"', '\\"').replace("'", "\\'")
                    
                    create_query = f"""
                    MERGE (l:Language {{id: "{clean_name}"}})
                    SET 
                        l.family = "{family}",
                        l.subfamily = "{subfamily}",
                        l.genus = "{genus}",
                        l.country_id = "{country_id}",
                        l.iso_code = "{iso_code}",
                        l.macroarea = "{macroarea}"
                    """
                    
                    self.graph.query(create_query)
                    
                    # Create country relationship if available
                    if pd.notna(lang.get('Country_ID')) and country_id:
                        country_query = f"""
                        MERGE (c:Country {{id: "{country_id}"}})
                        WITH c
                        MATCH (l:Language {{id: "{clean_name}"}})
                        MERGE (l)-[:LOCATED_IN]->(c)
                        """
                        self.graph.query(country_query)
                    
                    # Create family relationship
                    if family and family != 'Unknown':
                        family_name = f"{family}Family"
                        family_query = f"""
                        MERGE (f:Languagefamily {{id: "{family_name}"}})
                        WITH f
                        MATCH (l:Language {{id: "{clean_name}"}})
                        MERGE (l)-[:BELONGS_TO]->(f)
                        """
                        self.graph.query(family_query)
                    
                    added_count += 1
            
            # Final coverage check
            final_result = self.graph.query("MATCH (l:Language) RETURN count(l) as count")
            final_count = final_result[0]['count'] if final_result else 0
            
            print(f"✅ Added {added_count} missing languages")
            print(f"📊 Final coverage: {final_count}/{len(all_languages)} ({final_count/len(all_languages)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error completing languages: {e}")
            return False
    
    def enrich_with_csv_data(self, languages_csv='data/languages.csv', countries_csv='data/countries.csv'):
        """Enrich graph with complete data from CSV files."""
        if not self.graph:
            print("❌ No graph connection")
            return False
        
        print("🔄 Enriching graph with CSV data...")
        
        try:
            # Load data
            languages_df = pd.read_csv(languages_csv)
            countries_df = pd.read_csv(countries_csv)
            
            # Create Country nodes
            print("1️⃣ Creating Country nodes...")
            for _, country in tqdm(countries_df.iterrows(), desc="Countries", total=len(countries_df)):
                # Escape quotes properly for Cypher
                country_id = str(country['ID']).replace('"', '\\"').replace("'", "\\'")
                country_name = str(country['Name']).replace('"', '\\"').replace("'", "\\'")
                
                country_query = f"""
                MERGE (c:Country {{id: "{country_id}", name: "{country_name}"}})
                """
                self.graph.query(country_query)
            
            # Enrich Language nodes
            print("2️⃣ Enriching Language nodes...")
            enriched_count = 0
            
            for _, lang in tqdm(languages_df.iterrows(), desc="Languages", total=len(languages_df)):
                try:
                    # Escape all string values properly
                    name = str(lang['Name']).replace('"', '\\"').replace("'", "\\'")
                    family = str(lang.get('Family', 'Unknown')).replace('"', '\\"').replace("'", "\\'")
                    subfamily = str(lang.get('Subfamily', '')).replace('"', '\\"').replace("'", "\\'")
                    genus = str(lang.get('Genus', 'Unknown')).replace('"', '\\"').replace("'", "\\'")
                    macroarea = str(lang.get('Macroarea', 'Unknown')).replace('"', '\\"').replace("'", "\\'")
                    country_id = str(lang.get('Country_ID', '')).replace('"', '\\"').replace("'", "\\'")
                    iso_code = str(lang.get('ISO639P3code', '')).replace('"', '\\"').replace("'", "\\'")
                    
                    # Update language with complete data
                    update_query = f"""
                    MATCH (l:Language {{id: "{name}"}})
                    SET 
                        l.family = "{family}",
                        l.subfamily = "{subfamily}",
                        l.genus = "{genus}",
                        l.macroarea = "{macroarea}",
                        l.iso_code = "{iso_code}",
                        l.country_id = "{country_id}"
                    """
                    self.graph.query(update_query)
                    
                    # Create relationships with proper escaping
                    if pd.notna(lang.get('Country_ID')):
                        country_query = f"""
                        MATCH (l:Language {{id: "{name}"}})
                        MATCH (c:Country {{id: "{country_id}"}})
                        MERGE (l)-[:LOCATED_IN]->(c)
                        """
                        self.graph.query(country_query)
                    
                    if pd.notna(lang.get('Family')) and family != 'Unknown':
                        family_name = f"{family}Family".replace('"', '\\"').replace("'", "\\'")
                        family_query = f"""
                        MERGE (f:Languagefamily {{id: "{family_name}"}})
                        WITH f
                        MATCH (l:Language {{id: "{name}"}})
                        MERGE (l)-[:BELONGS_TO]->(f)
                        """
                        self.graph.query(family_query)
                    
                    enriched_count += 1
                    
                except Exception as e:
                    print(f"⚠️  Error processing {lang.get('Name', 'unknown')}: {e}")
                    continue
            
            print(f"✅ Enriched {enriched_count} languages with CSV data")
            return True
            
        except Exception as e:
            print(f"❌ Error enriching with CSV data: {e}")
            return False
    
    def get_graph_statistics(self):
        """Get comprehensive graph statistics."""
        if not self.graph:
            return {}
        
        try:
            stats = {}
            
            # Node counts
            node_types = ['Language', 'Country', 'Languagefamily']
            for node_type in node_types:
                result = self.graph.query(f"MATCH (n:{node_type}) RETURN count(n) as count")
                stats[f'{node_type.lower()}_count'] = result[0]['count'] if result else 0
            
            # Relationship counts
            rel_types = ['LOCATED_IN', 'BELONGS_TO']
            for rel_type in rel_types:
                result = self.graph.query(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                stats[f'{rel_type.lower()}_count'] = result[0]['count'] if result else 0
            
            # Language families
            family_result = self.graph.query("""
            MATCH (l:Language)-[:BELONGS_TO]->(f:Languagefamily)
            RETURN f.id as family, count(l) as language_count
            ORDER BY language_count DESC
            LIMIT 10
            """)
            stats['top_families'] = family_result
            
            # Countries with most languages
            country_result = self.graph.query("""
            MATCH (l:Language)-[:LOCATED_IN]->(c:Country)
            RETURN c.name as country, count(l) as language_count
            ORDER BY language_count DESC
            LIMIT 10
            """)
            stats['top_countries'] = country_result
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return {}

def main():
    """Main function for testing graph builder."""
    builder = GraphBuilder()
    
    print("🚀 Graph Builder Test")
    print("=" * 30)
    
    # Initialize
    if not builder.connect_to_neo4j():
        return
    
    if not builder.setup_llm_transformer():
        return
    
    # Test with sample chunks (if they exist)
    import glob
    chunk_files = glob.glob("output/chunks/*.txt")
    
    if chunk_files:
        print(f"📁 Found {len(chunk_files)} chunk files")
        nodes, rels = builder.build_from_chunks(chunk_files[:2])  # Test with first 2 chunks
        
        # Enrich with CSV data
        builder.enrich_with_csv_data()
        
        # Show statistics
        stats = builder.get_graph_statistics()
        print("\n📊 Graph Statistics:")
        for key, value in stats.items():
            if isinstance(value, list):
                print(f"   {key}: {len(value)} items")
            else:
                print(f"   {key}: {value}")
    else:
        print("❌ No chunk files found in output/chunks/")
        print("🚀 Run wals.py first to generate chunks")

if __name__ == "__main__":
    main()