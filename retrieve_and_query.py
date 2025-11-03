"""
Graph RAG Query System
=====================

Advanced retrieval and querying system using LangChain with Neo4j.
Implements graph-based RAG for contextual information retrieval.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from typing import Optional, Dict, Any

def setup_rag_system() -> Optional[GraphCypherQAChain]:
    """
    Initialize and configure the RAG system.
    
    Returns:
        GraphCypherQAChain: Configured chain for RAG queries, or None if setup fails
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Connect to Neo4j
        graph = Neo4jGraph(
            url=os.getenv('NEO4J_URI') or os.getenv('NEO4J_URL'),
            username=os.getenv('NEO4J_USERNAME'),
            password=os.getenv('NEO4J_PASSWORD')
        )
        
        # Create GraphCypherQAChain for intelligent querying
        cypher_chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True  # Acknowledging security risks
        )
        
        return cypher_chain
        
    except Exception as e:
        print(f"❌ Error setting up RAG system: {str(e)}")
        return None

def query_and_synthesize(query: str, cypher_chain: GraphCypherQAChain) -> str:
    """
    Query the knowledge graph and synthesize a response.
    
    Args:
        query: Natural language question
        cypher_chain: Configured GraphCypherQAChain instance
        
    Returns:
        str: Synthesized answer from the knowledge graph
    """
    try:
        # Execute the query
        result = cypher_chain.invoke({"query": query})
        
        # Extract the answer
        answer = result.get('result', 'No answer found')
        
        return answer
        
    except Exception as e:
        return f"Error processing query: {str(e)}"

def run_sample_queries():
    """
    Run sample queries to demonstrate the system.
    This function is used when the script is run directly.
    """
    print("🔗 Connected to Neo4j Graph Database for RAG")
    print("="*50)
    
    # Setup RAG system
    cypher_chain = setup_rag_system()
    if not cypher_chain:
        print("❌ Failed to setup RAG system")
        return
    
    print("🚀 Starting Graph RAG Query System")
    print("="*50)
    
    # Get node count for context
    try:
        graph = Neo4jGraph(
            url=os.getenv('NEO4J_URI') or os.getenv('NEO4J_URL'),
            username=os.getenv('NEO4J_USERNAME'),
            password=os.getenv('NEO4J_PASSWORD')
        )
        node_result = graph.query("MATCH (n) RETURN COUNT(n) as count")
        node_count = node_result[0]['count'] if node_result else 0
        print(f"📈 Graph contains {node_count} nodes")
        print("-" * 50)
    except:
        print("📈 Graph connection established")
        print("-" * 50)
    
    # Sample queries
    queries = [
        "What is spoken in Spain?",
        "How many countries do we have in the graph?", 
        "How many languages do we have in the graph?",
        "What are the relationships between countries and languages?",
        "Tell me about the entities in the knowledge graph"
    ]
    
    for query in queries:
        print(f"\n🔍 Processing: {query}")
        print("-" * 50)
        answer = query_and_synthesize(query, cypher_chain)
        print(f"💡 Answer: {answer}")
        print("-" * 75)
        print()

if __name__ == "__main__":
    run_sample_queries()

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") 
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize LLM and Graph
llm = ChatOpenAI(temperature=0, model="gpt-4")
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

print("🔗 Connected to Neo4j Graph Database for RAG")
print("="*50)

if __name__ == "__main__":
    run_sample_queries()