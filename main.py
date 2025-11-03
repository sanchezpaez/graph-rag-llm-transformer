"""
Graph RAG System - Main Interface
=================================

Simple main script that orchestrates the Graph RAG pipeline:
1. Build knowledge graph from text
2. Query the graph using natural language

Usage:
    python main.py                    # Interactive mode with menu
    python main.py --build            # Build graph only
    python main.py --query            # Query mode only  
    python main.py --demo             # Run demo queries
    python main.py --full             # Build + demo queries
"""

import argparse
import sys
from typing import Optional

def build_graph() -> bool:
    """
    Build the knowledge graph.
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("🏗️  Building Knowledge Graph...")
    print("=" * 50)
    
    try:
        import graph_builder
        if hasattr(graph_builder, 'build_knowledge_graph'):
            return graph_builder.build_knowledge_graph()
        else:
            # If it's the current script style, just import and run
            print("✅ Knowledge graph built successfully!")
            return True
            
    except ImportError:
        print("❌ Error: graph_builder.py not found!")
        return False
    except Exception as e:
        print(f"❌ Error building graph: {str(e)}")
        return False

def run_queries(interactive: bool = True) -> bool:
    """
    Run RAG queries on the knowledge graph.
    
    Args:
        interactive: If True, run interactive queries. If False, run demo queries.
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("🤖 Starting RAG Query System...")
    print("=" * 50)
    
    try:
        from retrieve_and_query import setup_rag_system, query_and_synthesize, run_sample_queries
        
        if not interactive:
            # Run demo queries
            run_sample_queries()
            return True
        
        # Interactive mode
        cypher_chain = setup_rag_system()
        if not cypher_chain:
            print("❌ Failed to setup RAG system")
            return False
            
        print("✅ RAG system ready!")
        print("\n💡 Ask questions about your knowledge graph")
        print("📝 Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                query = input("\n🤔 Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if query:
                    answer = query_and_synthesize(query, cypher_chain)
                    print(f"💡 Answer: {answer}")
                else:
                    print("❌ Please enter a valid question")
                    
            except KeyboardInterrupt:
                break
                
        return True
        
    except ImportError:
        print("❌ Error: retrieve_and_query.py not found!")
        return False
    except Exception as e:
        print(f"❌ Error in query system: {str(e)}")
        return False

def show_menu() -> str:
    """Show interactive menu and get user choice."""
    print("\n🎯 Graph RAG System")
    print("=" * 30)
    print("1. Build Knowledge Graph")
    print("2. Query Graph (Interactive)")
    print("3. Run Demo Queries")
    print("4. Full Pipeline (Build + Demo)")
    print("5. Exit")
    print("-" * 30)
    
    while True:
        choice = input("� Choose option (1-5): ").strip()
        if choice in ['1', '2', '3', '4', '5']:
            return choice
        print("❌ Invalid choice. Please enter 1-5.")

def main():
    """Main function to orchestrate the Graph RAG system."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Graph RAG System')
    parser.add_argument('--build', action='store_true', help='Build knowledge graph only')
    parser.add_argument('--query', action='store_true', help='Interactive query mode')
    parser.add_argument('--demo', action='store_true', help='Run demo queries')
    parser.add_argument('--full', action='store_true', help='Build graph + run demo')
    
    args = parser.parse_args()
    
    # Command line mode
    if args.build:
        success = build_graph()
        sys.exit(0 if success else 1)
        
    elif args.query:
        success = run_queries(interactive=True)
        sys.exit(0 if success else 1)
        
    elif args.demo:
        success = run_queries(interactive=False)
        sys.exit(0 if success else 1)
        
    elif args.full:
        print("🚀 Running Full Pipeline...")
        if build_graph():
            print("\n" + "="*50)
            success = run_queries(interactive=False)
            sys.exit(0 if success else 1)
        else:
            sys.exit(1)
    
    # Interactive menu mode (default)
    try:
        while True:
            choice = show_menu()
            
            if choice == '1':
                build_graph()
                input("\n📎 Press Enter to continue...")
                
            elif choice == '2':
                run_queries(interactive=True)
                
            elif choice == '3':
                run_queries(interactive=False)
                input("\n📎 Press Enter to continue...")
                
            elif choice == '4':
                print("🚀 Running Full Pipeline...")
                if build_graph():
                    print("\n" + "="*50)
                    run_queries(interactive=False)
                input("\n📎 Press Enter to continue...")
                
            elif choice == '5':
                print("👋 Goodbye!")
                break
                
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()