"""
Graph RAG with LLMGraphTransformer
=================================

Automatic knowledge graph extraction and storage using LangChain's LLMGraphTransformer,
Neo4j AuraDB, and OpenAI. This implementation provides fully automated entity and 
relationship extraction from unstructured text.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") 
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def extract_with_llm_transformer(text, llm, graph):
    """
    Extract knowledge graph using LLMGraphTransformer.
    
    Args:
        text (str): Input text for knowledge extraction
        llm: OpenAI language model instance
        graph: Neo4j graph database connection
        
    Returns:
        tuple: (number of nodes created, number of relationships created)
    """
    
    # Clear previous data
    graph.query("MATCH (n) DETACH DELETE n")
    print("🗑️  Database cleared")
    
    # Create document from text
    documents = [Document(page_content=text)]
    print(f"📄 Created document with {len(text)} characters")
    
    # Initialize LLMGraphTransformer
    print("🤖 Initializing LLMGraphTransformer...")
    llm_transformer = LLMGraphTransformer(llm=llm)
    
    # Convert to graph documents
    print("🧠 Extracting knowledge graph automatically...")
    try:
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        print(f"✅ LLMGraphTransformer extracted {len(graph_documents)} graph documents")
        
        if not graph_documents:
            print("⚠️  No graph documents created - text may lack extractable knowledge")
            return 0, 0
        
        # Add to Neo4j database
        print("📊 Adding to Neo4j database...")
        graph.add_graph_documents(graph_documents)
        
        # Count created entities and relationships
        result = graph.query("MATCH (n) RETURN count(n) as node_count")
        node_count = result[0]['node_count'] if result else 0
        
        result = graph.query("MATCH ()-[r]->() RETURN count(r) as rel_count")
        rel_count = result[0]['rel_count'] if result else 0
        
        print(f"✅ Automatically created {node_count} nodes and {rel_count} relationships")
        return node_count, rel_count
        
    except Exception as e:
        print(f"❌ LLMGraphTransformer failed: {e}")
        return 0, 0

def main():
    """
    Main function for Graph RAG knowledge extraction.
    
    Processes input text through LLMGraphTransformer to create a knowledge graph
    in Neo4j, then displays extraction results and usage instructions.
    """
    
    # Input text for analysis - modify this to process different content
    text = """
    The most spoken languages in the world include Mandarin Chinese with over 918 million native speakers, primarily in China. Spanish is spoken by approximately 460 million people in countries like Spain, Mexico, Argentina, Colombia and many others. English, with 379 million native speakers, is the most studied second language, used in countries like the United States, United Kingdom, Canada and Australia.

    Hindi is official in India along with 21 other constitutionally recognized languages. Arabic is spoken in 22 countries of the Arab League, including Egypt, Saudi Arabia and Morocco. Bengali is the official language of Bangladesh and the second most spoken in India.

    In Europe, German is spoken in Germany, Austria and Switzerland. French is official in France, Canada, Belgium and many African countries. Italian is spoken in Italy, Switzerland and San Marino. Russian is official in Russia, Belarus and Kazakhstan.
    """

    print("🚀 Graph RAG with LLMGraphTransformer")
    print("="*50)
    print("📝 Input text preview:")
    print("-" * 30)
    print(text[:200] + "..." if len(text) > 200 else text)
    print("-" * 30)

    # Initialize services
    try:
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        graph = Neo4jGraph(
            url=NEO4J_URL,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        print("✅ Connected to OpenAI and Neo4j")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return

    # Extract knowledge graph using LLMGraphTransformer
    nodes_created, rels_created = extract_with_llm_transformer(text, llm, graph)

    if nodes_created > 0 or rels_created > 0:
        print("\n🎉 SUCCESS! LLMGraphTransformer created the knowledge graph!")
        
        # Display extraction results
        print(f"\n📊 Automatic Extraction Results:")
        print(f"   • Nodes: {nodes_created}")
        print(f"   • Relationships: {rels_created}")
        
        # Show automatically detected entity types
        result = graph.query("MATCH (n) RETURN labels(n) as labels, count(n) as count ORDER BY count DESC")
        if result:
            print(f"\n🏷️  Auto-detected Entity Types:")
            for row in result:
                labels = row['labels']
                if labels:
                    label = labels[0] if len(labels) > 0 else 'Unknown'
                    print(f"   • {label}: {row['count']} entities")
        
        # Show automatically detected relationship types
        result = graph.query("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC")
        if result:
            print(f"\n🔗 Auto-detected Relationship Types:")
            for row in result:
                print(f"   • {row['rel_type']}: {row['count']} connections")
        
        # Show sample extracted facts
        result = graph.query("""
            MATCH (a)-[r]->(b) 
            RETURN a.id + ' ' + type(r) + ' ' + b.id as fact 
            LIMIT 5
        """)
        if result:
            print(f"\n💡 Sample Auto-extracted Facts:")
            for row in result:
                print(f"   • {row['fact']}")
        
        print(f"\n✅ Key Advantages of LLMGraphTransformer:")
        print(f"   🎯 Zero manual entity/relationship code")
        print(f"   🎯 Automatic schema detection")
        print(f"   🎯 Built-in best practices")
        print(f"   🎯 Production-ready approach")
        
        print(f"\n🔍 Query your graph:")
        print(f"   • Interactive: python query_interactive.py")
        print(f"   • Neo4j Browser: AuraDB web interface")
        
    else:
        print("\n❌ LLMGraphTransformer could not extract knowledge")
        print("💭 This can happen when:")
        print("   • Text lacks clear factual relationships")
        print("   • Content is too abstract or opinion-based")
        print("   • LLM cannot identify extractable patterns")
        
        print(f"\n💡 For better results, use text with:")
        print(f"   • Clear entities (people, places, organizations)")
        print(f"   • Explicit relationships between entities")
        print(f"   • Factual rather than descriptive content")
        
        print(f"\n🔄 To retry:")
        print(f"   1. Edit the 'text' variable in this file")
        print(f"   2. Run: uv run python retriever_llm_transformer.py")

if __name__ == "__main__":
    main()