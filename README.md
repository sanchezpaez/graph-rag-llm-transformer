# WALS Knowledge Graph 🌍

A comprehensive Graph RAG system for exploring World Atlas of Language Structures (WALS) data through intelligent natural language queries. Built with LangChain, Neo4j, and OpenAI GPT.

## Quick Start

```bash
# 1. Setup
git clone [repo-url] && cd graph-rag-llm-transformer
uv sync
cp .env.example .env  # Add Neo4j + OpenAI keys

# 2. Build knowledge graph (first time only)
uv run python main.py --mode build

# 3. Start interactive queries
uv run python main.py --mode query
```

### Pipeline Options
```bash
uv run python main.py --mode full     # Complete: build + query
uv run python main.py --mode process  # WALS data processing only
uv run python main.py --mode build    # Graph building only
uv run python main.py --mode query    # Interactive queries only
```

## Features

- **Large Coverage of WALS Dataset**: 2,639 languages from World Atlas of Language Structures
- **Natural Language Queries**: "What languages are spoken in Spain?" → 15 languages
- **Smart Country Mapping**: "Germany" → "DE", "France" → "FR" automatic conversion
- **Geographic Queries**: Search by country, region, or macroarea
- **Family Queries**: Explore linguistic families and classifications
- **4 Query Modes**: Statistics, Geographic, Families, Advanced Cypher

## Example Queries

- **Geographic**: "Spain" → 15 languages | "France" → 7 languages | "Africa" → 547 languages
- **Families**: "Romance" → 24 languages | "Germanic" → 20+ languages | "Niger-Congo" → 190+ languages
- **Statistics**: Regional distribution, top families, data coverage insights

## Architecture

```
main.py              # Main orchestrator and mode selection
├── graph_builder.py # WALS data processing → Neo4j graph
├── graph_explorer.py # Interactive query interface with LLM integration
└── wals.py          # WALS data processing utilities
```

**Graph Structure**: Single `Language` node type with properties: `id`, `family`, `genus`, `country_id`, `macroarea`, `coordinates`

## Performance

- **Coverage**: 2,639 out of 3,573 languages (73.8%)
- **Build Time**: ~45 minutes for full dataset
- **Success Rate**: 99.3% chunk processing
- **Geographic Coverage**: 100+ countries across 6 macroareas

## Requirements

- **Neo4j AuraDB**: Create free instance at https://neo4j.com/cloud/aura/
- **OpenAI API Key**: Get one at https://platform.openai.com/api-keys
- **Python 3.11+** with uv package manager

## Future Roadmap

**Enhanced Queries**:
- Bidirectional queries (language → countries where spoken)
- Multi-country mapping and complex geographic filters
- Advanced linguistic feature queries

**Data Expansion**:
- Additional linguistic features from WALS
- Speaker population data
- Language endangerment status
- Etymological relationships

**UI Improvements**:
- Web-based interface
- Interactive maps and visualizations
- Query builder with autocomplete
- Export capabilities (CSV, JSON)

**Advanced Features**:
- Voice query interface
- API endpoints for external integration
- Performance optimization and caching
- Comprehensive testing and monitoring

## Data Source

This project uses data from the World Atlas of Language Structures (WALS):

**Dryer, Matthew S. & Haspelmath, Martin (eds.) 2013. The World Atlas of Language Structures Online. Leipzig: Max Planck Institute for Evolutionary Anthropology.** (Available online at https://wals.info)

## License

MIT License
