"""
Interactive Graph Query Tool
===========================

Interactive interface for exploring Neo4j knowledge graphs created by LLMGraphTransformer.
Provides predefined queries and custom Cypher query execution capabilities.
"""

from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
NEO4J_URL = os.environ.get("NEO4J_URL")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

# Initialize Neo4j connection
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

print("🔗 Connected to Neo4j Graph Database")
print("="*50)

def run_query(query_text, parameters=None):
    """
    Execute a Cypher query and display results.
    
    Args:
        query_text (str): Cypher query to execute
        parameters (dict, optional): Query parameters
        
    Returns:
        list: Query results or None if error occurred
    """
    try:
        if parameters:
            result = graph.query(query_text, parameters)
        else:
            result = graph.query(query_text)
        
        if result:
            print(f"📊 Found {len(result)} results:")
            for i, row in enumerate(result, 1):
                print(f"   {i}. {dict(row)}")
        else:
            print("   No results found.")
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def show_menu():
    """Display the interactive menu options."""
    print("\n" + "="*50)
    print("🔍 INTERACTIVE QUERY MENU")
    print("="*50)
    print("1. Show all countries")
    print("2. Show all languages")
    print("3. Find languages in a country")
    print("4. Find countries for a language")
    print("5. Show database statistics")
    print("6. Custom Cypher query")
    print("7. Query examples")
    print("0. Exit")
    print("="*50)

def show_examples():
    """Display example Cypher queries with descriptions."""
    print("\n📖 QUERY EXAMPLES:")
    print("="*30)
    
    examples = [
        {
            "title": "All nodes and their types",
            "query": "MATCH (n) RETURN labels(n)[0] as type, count(n) as count ORDER BY count DESC"
        },
        {
            "title": "Languages with most countries",
            "query": "MATCH (l:Language)-[]->(c:Country) RETURN l.id as language, count(c) as countries ORDER BY countries DESC LIMIT 5"
        },
        {
            "title": "Countries in Europe with languages",
            "query": "MATCH (c:Country)<-[]-(l:Language) WHERE c.id IN ['Germany', 'France', 'Spain', 'Italy'] RETURN c.id as country, collect(l.id) as languages"
        },
        {
            "title": "Find multilingual countries",
            "query": "MATCH (c:Country)<-[]-(l:Language) WITH c, count(l) as lang_count WHERE lang_count > 1 RETURN c.id as country, lang_count ORDER BY lang_count DESC"
        },
        {
            "title": "Languages with special characters",
            "query": "MATCH (l:Language) WHERE l.id =~ '.*[^a-zA-Z\\\\s].*' RETURN l.id as language ORDER BY l.id"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}:")
        print(f"   Query: {example['query']}")

def main():
    """Main interactive loop for graph exploration."""
    while True:
        show_menu()
        choice = input("\n🎯 Choose an option (0-7): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
            
        elif choice == "1":
            print("\n🌍 All countries in the database:")
            run_query("MATCH (c:Country) RETURN c.id as name ORDER BY c.id")
            
        elif choice == "2":
            print("\n🗣️ All languages in the database:")
            run_query("MATCH (l:Language) RETURN l.id as name ORDER BY l.id")
            
        elif choice == "3":
            country = input("Enter country name: ").strip()
            print(f"\n🗣️ Languages in {country}:")
            run_query("""
                MATCH (country:Country {id: $country})<-[r]-(language:Language)
                RETURN language.id as language, type(r) as relationship
                ORDER BY language.id
            """, {"country": country})
            
        elif choice == "4":
            language = input("Enter language name: ").strip()
            print(f"\n🌍 Countries where {language} is spoken:")
            run_query("""
                MATCH (language:Language {id: $language})-[r]->(country:Country)
                RETURN country.id as country, type(r) as relationship
                ORDER BY country.id
            """, {"language": language})
            
        elif choice == "5":
            print("\n📊 Database statistics:")
            print("\nNode counts:")
            run_query("MATCH (n) RETURN labels(n)[0] as type, count(n) as count ORDER BY count DESC")
            print("\nRelationship counts:")
            run_query("MATCH ()-[r]->() RETURN type(r) as relationship, count(r) as count ORDER BY count DESC")
            
        elif choice == "6":
            print("\n⚡ Enter your custom Cypher query:")
            print("Example: MATCH (n:Language) RETURN n.id LIMIT 5")
            custom_query = input("Query: ").strip()
            if custom_query:
                print(f"\n🔍 Executing: {custom_query}")
                run_query(custom_query)
            else:
                print("Empty query, skipping...")
                
        elif choice == "7":
            show_examples()
            
        else:
            print("❌ Invalid option. Please choose 0-7.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    print("🚀 Welcome to the Interactive Graph Query Tool!")
    print("💡 This tool lets you explore the knowledge graph created by LLMGraphTransformer")
    print("🔧 You can run predefined queries or write your own Cypher queries")
    main()