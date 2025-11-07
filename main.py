"""
Linguistic Graph RAG System
============================

Main orchestration script for building and querying knowledge graphs from WALS linguistic data.
Supports multiple execution modes: data processing, graph building, and interactive queries.
"""

import argparse
import sys
from wals import WALSDataProcessor
from graph_builder import GraphBuilder
from graph_explorer import GraphExplorer

def process_wals_data():
    """Process WALS data and generate optimized chunks."""
    print("🌍 Linguistic Graph RAG System")
    print("=" * 40)
    print("📁 Output directories ready")
    
    processor = WALSDataProcessor()
    
    # Load WALS data
    if not processor.load_data():
        print("❌ Failed to load WALS data")
        return False
    
    processor.setup_output_structure()
    
    # Generate chunks
    chunk_files = processor.generate_chunks()
    print(f"✅ Generated {len(chunk_files)} chunks with WALS features")
    
    # Show statistics
    stats = processor.get_statistics()
    print(f"📊 Total languages: {stats['total_languages']}")
    print(f"📊 Language families: {stats['total_families']}")
    print(f"📊 Countries represented: {stats['total_countries']}")
    
    return len(chunk_files) > 0

def build_knowledge_graph():
    """Build the knowledge graph from processed chunks."""
    print("\n🏗️  Building Knowledge Graph")
    print("=" * 30)
    
    builder = GraphBuilder()
    
    # Initialize connections
    if not builder.connect_to_neo4j():
        return False
    
    if not builder.setup_llm_transformer():
        return False
    
    # Build graph from chunks
    print("🔍 Building from chunks with linguistic features...")
    nodes, relationships = builder.build_from_enhanced_chunks()
    
    if nodes == 0:
        print("❌ No graph elements created")
        return False
    
    # Show final statistics
    stats = builder.get_graph_statistics()
    print(f"\n📊 Final Graph Statistics:")
    print(f"   Languages: {stats.get('language_count', 0)}")
    print(f"   Countries: {stats.get('country_count', 0)}")
    print(f"   Language Families: {stats.get('languagefamily_count', 0)}")
    
    return True

def interactive_query_mode():
    """Start interactive query session."""
    print("\n🔍 Query Mode")
    print("=" * 30)
    
    explorer = GraphExplorer()
    
    # Initialize
    if not explorer.connect_to_neo4j():
        return False
    
    # Setup LLM for natural language queries
    if not explorer.setup_qa_chain():
        print("⚠️  LLM not available, using predefined queries only")
    
    # Start interactive session
    explorer.interactive_query()
    return True

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Linguistic Graph RAG System")
    parser.add_argument(
        "--mode", 
        choices=["full", "process", "build", "query"],
        default="full",
        help="Execution mode: full (all steps), process (data only), build (graph only), query (queries only)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "full":
            # Complete pipeline
            print("🚀 Running complete pipeline...")
            if not process_wals_data():
                sys.exit(1)
            if not build_knowledge_graph():
                sys.exit(1)
            interactive_query_mode()
            
        elif args.mode == "process":
            # Data processing only
            if not process_wals_data():
                sys.exit(1)
                
        elif args.mode == "build":
            # Graph building only
            if not build_knowledge_graph():
                sys.exit(1)
                
        elif args.mode == "query":
            # Query mode only
            if not interactive_query_mode():
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
