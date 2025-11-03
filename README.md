# Graph RAG with LLMGraphTransformer

Automatic Knowledge Graph generation and retrieval system using **LangChain's LLMGraphTransformer**, **Neo4j AuraDB**, and **OpenAI**.

## Overview

This project implements a complete Graph RAG (Retrieval-Augmented Generation) pipeline that automatically extracts entities and relationships from text documents and stores them in a Neo4j graph database. Unlike traditional RAG systems that rely on vector similarity, Graph RAG leverages the semantic relationships between entities to provide more contextual and accurate information retrieval.

### Key Technologies

- **LangChain**: Provides the LLMGraphTransformer for automated knowledge graph construction from unstructured text
- **Neo4j AuraDB**: Cloud-hosted graph database offering powerful graph query capabilities with Cypher
- **OpenAI GPT**: Drives the intelligent entity and relationship extraction process
- **UV Package Manager**: Ensures reproducible Python environments and dependency management

### Benefits

- **Automatic Schema Discovery**: No manual entity/relationship definition required
- **Semantic Relationships**: Captures complex connections between concepts that vector search might miss
- **Scalable Architecture**: Cloud-based Neo4j AuraDB handles large-scale graph operations
- **Production Ready**: Built on enterprise-grade tools (LangChain, Neo4j, OpenAI)

## Quick Setup

### 1. Install UV package manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and setup project
```bash
git clone [your-repo]
cd graph_rag
```

### 3. Create environment and install dependencies
```bash
uv venv .venv --python 3.11
uv pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

## Usage

### **Step 1: Create the knowledge graph (REQUIRED FIRST)**
```bash
uv run python graph_builder.py
```
This script:
- Connects to Neo4j and clears any existing data
- Processes the input text using LLMGraphTransformer
- Automatically extracts entities and relationships
- Stores the knowledge graph in Neo4j

### **Step 2: Query and explore the graph**
```bash
uv run python graph_explorer.py
```
This script:
- Provides an interactive menu for graph exploration
- **Requires the database to have data from Step 1**
- Offers predefined queries and custom Cypher query execution

### **Important Notes:**
- ⚠️ **Always run `graph_builder.py` first** to populate the database
- 🔄 Running `graph_builder.py` will clear and recreate the entire graph
- 💡 LLMGraphTransformer uses `id` property (not `name`) for entity names

## Required Credentials

1. **OpenAI API Key**: Get one at https://platform.openai.com/api-keys
2. **Neo4j AuraDB**: Create a free instance at https://neo4j.com/cloud/aura/

## Features

- ✅ Automatic entity and relationship extraction
- ✅ Zero manual schema configuration
- ✅ Support for multiple languages (English/Spanish)
- ✅ Interactive query interface
- ✅ Cloud-based graph database (Neo4j AuraDB)
- ✅ Production-ready architecture

## Troubleshooting

If you encounter nested environment issues, always use:
```bash
uv run python [script_name].py
```

Instead of manually activating environments.

**Manual activation (alternative):**
```bash
source .venv/bin/activate
python graph_builder.py
python graph_explorer.py
deactivate
```

## Future Roadmap

### Phase 1: Enhanced Retrieval
- [ ] Implement graph-based RAG retriever
- [ ] Add similarity search with graph traversal
- [ ] Integrate vector embeddings with graph relationships

### Phase 2: User Interface
- [ ] Web-based graph visualization
- [ ] Interactive query builder
- [ ] Real-time graph exploration

### Phase 3: Advanced Features
- [ ] Multi-document graph merging
- [ ] Incremental graph updates
- [ ] Custom entity/relationship types
- [ ] Graph-based summarization

### Phase 4: Production Enhancements
- [ ] Batch processing capabilities
- [ ] API endpoint creation
- [ ] Performance optimization
- [ ] Monitoring and analytics