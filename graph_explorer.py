"""
Graph Explorer
==============

Handles intelligent querying of the linguistic knowledge graph using LLM-enhanced Cypher generation.
Provides natural language interface for complex linguistic queries.
"""

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

class GraphExplorer:
    """Intelligent graph querying with natural language interface."""
    
    def __init__(self):
        self.graph = None
        self.llm = None
        self.qa_chain = None
        
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
    
    def setup_qa_chain(self, model="gpt-3.5-turbo"):
        """Setup LLM and QA chain with enhanced prompts."""
        try:
            self.llm = ChatOpenAI(temperature=0, model=model)
            
            # Enhanced Cypher generation prompt for linguistic queries
            cypher_prompt = PromptTemplate(
                input_variables=["schema", "question"],
                template="""You are an expert in both linguistics and Cypher queries. 
Generate precise Cypher queries for the WALS linguistic knowledge graph.

CRITICAL DATABASE STRUCTURE:
• ONLY Language nodes exist (no other node types)
• NO relationships between nodes
• ALL data is stored as properties of Language nodes

LANGUAGE NODE PROPERTIES:
• id: Language identifier/name (e.g., "Plains-Indians Sign Language", "Arapaho", "Basque (Bidasoa Valley)")
• name: Language name (often NULL - use id instead)
• family: Language family (e.g., "Indo-European", "Austronesian", "Niger-Congo")
• subfamily: Language subfamily (e.g., "Romance", "Germanic") - limited coverage
• genus: Language genus (e.g., "Germanic", "Romance", "Semitic")
• macroarea: Geographic region (e.g., "Africa", "Eurasia", "Papunesia", "North America", "South America", "Australia")
• country_id: Country identifier (e.g., "United States", "ES", "PG", "Indonesia")
• iso_code: ISO language codes
• latitude, longitude: Geographic coordinates

IMPORTANT: Language names are stored in the 'id' field, NOT in 'name' (which is usually NULL).
For queries requesting language names, use l.id instead of l.name.

QUERY PATTERNS FOR NAMES:
✅ List language names: MATCH (l:Language) WHERE l.country_id = 'United States' RETURN l.id as language_name
✅ Count with fallback: MATCH (l:Language) WHERE l.country_id = 'United States' RETURN count(l) as count
✅ Names with details: MATCH (l:Language) WHERE l.country_id = 'United States' RETURN l.id as name, l.family, l.genus

RESPONSE FORMATTING:
- For counting questions (how many): Return just the count number
- For listing questions (list languages, show languages): Present as numbered list with total at end
  Example: "Languages spoken in Spain:
  1. Basque (Bidasoa Valley)
  2. Basque (Gernica) 
  3. Basque (Hondarribia)
  4. Basque (Basaburua And Imoz)
  5. Basque (Lekeitio)
  ...
  Total: 15 languages"
- For mixed queries: Combine both formats appropriately

When you receive a list of language names, ALWAYS format them as a numbered list with clear structure and include the total count at the end.

REAL DATA EXAMPLES:
Language: "Plains-Indians Sign Language" → family: "other", genus: "Sign Languages", country_id: "United States", macroarea: "North America"
Language: "Basque (Bidasoa Valley)" → family: "Isolate", genus: "Basque", country_id: "ES", macroarea: "Eurasia"
Language: "Asmat" → family: "Trans-New Guinea", genus: "Asmat-Kamoro", country_id: "Indonesia", macroarea: "Papunesia"
Language: "Arapaho" → family: "Algic", genus: "Algonquian", country_id: "United States", macroarea: "North America"

COUNTRY_ID VALUES (exact strings to use):
• USA queries → "United States"
• Spain queries → "ES" 
• Papua New Guinea → "PG"
• Indonesia → "Indonesia"
• Mexico → "MX"
• India → "India"
• Ethiopia → "ET"
• Chad → "TD"

MACROAREA VALUES (exact strings):
"Africa", "Eurasia", "Papunesia", "North America", "South America", "Australia"

FAMILY VALUES (exact strings):
"Indo-European", "Austronesian", "Niger-Congo", "Sino-Tibetan", "Afro-Asiatic", "Torricelli"

CORRECT QUERY PATTERNS:
✅ Count languages in USA: MATCH (l:Language) WHERE l.country_id = 'United States' RETURN count(l)
✅ List languages in USA: MATCH (l:Language) WHERE l.country_id = 'United States' RETURN l.id as language_name
✅ Count Indo-European languages: MATCH (l:Language) WHERE l.family = 'Indo-European' RETURN count(l)
✅ Languages in Africa: MATCH (l:Language) WHERE l.macroarea = 'Africa' RETURN count(l)
✅ List countries: MATCH (l:Language) WHERE l.country_id IS NOT NULL RETURN DISTINCT l.country_id ORDER BY l.country_id
✅ Languages with details: MATCH (l:Language) WHERE l.country_id = 'United States' RETURN l.id as name, l.family, l.genus LIMIT 10

FORBIDDEN PATTERNS:
❌ (l)-[:LOCATED_IN]->(c:Country) - NO relationships exist
❌ (c:Country) - NO Country nodes exist
❌ WHERE c.name = "..." - NO Country nodes exist

Schema information:
{schema}

Question: {question}

IMPORTANT INSTRUCTIONS:
1. Generate a precise Cypher query using ONLY Language nodes and their properties
2. Use l.id for language names (NOT l.name which is usually NULL)
3. When the question asks for a list of languages, format the response as:
   - Title with context
   - Numbered list of languages (show first 10-15, then "..." if more)
   - Total count at the end
4. When the question asks "how many", just return the count clearly
5. Use the exact country_id values provided above (e.g., "United States" not "USA")

Generate a precise Cypher query:
"""
            )
            
            self.qa_chain = GraphCypherQAChain.from_llm(
                llm=self.llm,
                graph=self.graph,
                verbose=True,
                cypher_prompt=cypher_prompt,
                return_intermediate_steps=True,
                allow_dangerous_requests=True  # Required for Neo4j operations
            )
            
            print(f"✅ Graph QA Chain ready (model: {model})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup QA chain: {e}")
            return False
    
    def query_natural_language(self, question):
        """Query the graph using natural language with LLM-generated Cypher."""
        if not self.graph:
            print("❌ No graph connection")
            return None
        
        try:
            print(f"🔍 Query: {question}")
            print("-" * 50)
            
            # Use LLM to generate Cypher
            if not self.qa_chain:
                print("❌ QA chain not initialized")
                return None
            
            result = self.qa_chain.invoke({"query": question})
            
            # Extract information
            answer = result.get('result', 'No answer found')
            cypher_query = None
            
            if 'intermediate_steps' in result:
                for step in result['intermediate_steps']:
                    if 'query' in step:
                        cypher_query = step['query']
                        break
            
            print(f"📊 Generated Cypher: {cypher_query}")
            
            return {
                'answer': answer,
                'cypher': cypher_query,
                'raw_result': result
            }
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return None
    
    def query_cypher_direct(self, cypher_query):
        """Execute Cypher query directly."""
        if not self.graph:
            print("❌ No graph connection")
            return None
        
        try:
            print(f"⚡ Executing: {cypher_query}")
            result = self.graph.query(cypher_query)
            print(f"📋 Results: {len(result)} rows")
            return result
            
        except Exception as e:
            print(f"❌ Cypher error: {e}")
            return None
    
    def query_cypher_silent(self, cypher_query):
        """Execute Cypher query silently without technical output."""
        if not self.graph:
            return None
        
        try:
            result = self.graph.query(cypher_query)
            return result
            
        except Exception as e:
            return None
    
    def get_languages_in_country(self, country_name):
        """Get all languages spoken in a specific country."""
        cypher = f"""
        MATCH (l:Language)-[:LOCATED_IN]->(c:Country)
        WHERE toLower(c.name) CONTAINS toLower("{country_name}")
        RETURN l.id as language, c.name as country, l.family as family, l.subfamily as subfamily
        ORDER BY l.id
        """
        return self.query_cypher_direct(cypher)
    
    def get_graph_overview(self):
        """Get overview of the linguistic graph."""
        queries = {
            "total_languages": "MATCH (l:Language) RETURN count(l) as count",
            "total_countries": "MATCH (c:Country) RETURN count(c) as count",
            "total_families": "MATCH (l:Language) RETURN count(DISTINCT l.family) as count",
            "top_families": """
                MATCH (l:Language) 
                WHERE l.family IS NOT NULL AND l.family <> ""
                RETURN l.family as family, count(l) as language_count 
                ORDER BY language_count DESC 
                LIMIT 10
            """,
            "countries_with_most_languages": """
                MATCH (l:Language)-[:LOCATED_IN]->(c:Country)
                RETURN c.name as country, count(l) as language_count
                ORDER BY language_count DESC
                LIMIT 10
            """
        }
        
        overview = {}
        for key, query in queries.items():
            try:
                result = self.graph.query(query)
                overview[key] = result
            except Exception as e:
                overview[key] = f"Error: {e}"
        
    def interactive_query(self):
        """Start interactive query session with organized menu interface."""
        if not self.graph:
            print("❌ No graph connection")
            return False
        
        self.show_welcome_menu()
        
        while True:
            try:
                choice = input("\n🔍 Select an option (1-5): ").strip()
                
                if choice in ['5', 'exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                
                if choice == '1':
                    self.show_all_statistics()
                elif choice == '2':
                    location = input("Enter country or region name: ").strip()
                    if location:
                        self._handle_geographic_query(location)
                elif choice == '3':
                    family = input("Enter linguistic family name: ").strip()
                    if family:
                        self._handle_family_query(family)
                elif choice == '4':
                    self.handle_cypher_query()
                elif choice.lower() in ['menu', 'm']:
                    self.show_welcome_menu()
                else:
                    print("❌ Invalid option. Use numbers 1-5 or 'menu' to see options.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        return True
    
    def show_all_statistics(self):
        """Display comprehensive statistics about the database."""
        print("\n📊 WALS Database Statistics")
        print("=" * 50)
        
        # Total languages
        result = self.query_cypher_silent("MATCH (l:Language) RETURN count(l) as total")
        total_languages = result[0]['total'] if result else 0
        print(f"🌍 Total Languages: {total_languages:,}")
        
        # Regional distribution
        print(f"\n📍 Languages by Region:")
        result = self.query_cypher_silent("MATCH (l:Language) WHERE l.macroarea IS NOT NULL RETURN l.macroarea as region, count(l) as count ORDER BY count DESC")
        if result:
            for row in result:
                print(f"   • {row['region']}: {row['count']:,} languages")
        
        # Top language families
        print(f"\n👨‍👩‍👧‍👦 Largest Language Families:")
        result = self.query_cypher_silent("MATCH (l:Language) WHERE l.family IS NOT NULL AND l.family <> 'nan' RETURN l.family as family, count(l) as count ORDER BY count DESC LIMIT 8")
        if result:
            for i, row in enumerate(result, 1):
                print(f"   {i}. {row['family']}: {row['count']:,} languages")
        
        # Data coverage summary
        print(f"\n📊 Data Coverage:")
        coverage_data = [
            ("Languages with country data", "MATCH (l:Language) WHERE l.country_id IS NOT NULL AND l.country_id <> 'nan' RETURN count(l) as count"),
            ("Languages with family data", "MATCH (l:Language) WHERE l.family IS NOT NULL AND l.family <> 'nan' RETURN count(l) as count"),
            ("Languages with geographic coordinates", "MATCH (l:Language) WHERE l.latitude IS NOT NULL AND l.longitude IS NOT NULL RETURN count(l) as count")
        ]
        
        for label, query in coverage_data:
            result = self.query_cypher_silent(query)
            if result:
                count = result[0]['count']
                percentage = (count / total_languages) * 100
                print(f"   • {label}: {count:,} ({percentage:.1f}%)")
        
        print("=" * 50)
    
    def show_welcome_menu(self):
        """Display the main interactive menu."""
        print("\n" + "="*60)
        print("🌍 WALS Linguistic Database Explorer")
        print("="*60)
        print("📊 Available queries:")
        print()
        print("1️⃣  📈 Show General Statistics")
        print("     • Shows total languages, regional distribution, and language family data")
        print()
        print("2️⃣  🌏 Query Languages by Geography") 
        print("     • Available regions: Africa, Australia, Eurasia, North America, Papunesia, South America")
        print("     • Try any country: 'Spain', 'France', 'Germany', 'Indonesia', 'Brazil', 'China'")
        print("     • System will intelligently find the correct country code")
        print()
        print("3️⃣  👨‍👩‍👧‍👦 Query Languages by Linguistic Family")
        print("     • Try: 'Romance', 'Germanic', 'Indo-European', 'Niger-Congo'")
        print("     • En español: 'indoeuropeo', 'románico', 'germánico', 'austronesio'")
        print()
        print("4️⃣  ⚡ Direct Cypher Query (Advanced)")
        print("     • For experienced users: Write Neo4j Cypher queries directly")
        print()
        print("5️⃣  🚪 Exit")
        print()
        print("💡 Tip: Type 'menu' at any time to return here")
        print("="*60)
    
    def handle_cypher_query(self):
        """Handle direct Cypher queries."""
        print("\n🔧 Direct Cypher Query (Advanced)")
        print("-" * 40)
        print("Write Cypher queries directly against the database.")
        print("Examples:")
        print("• MATCH (l:Language) RETURN count(l)")
        print("• MATCH (l:Language) WHERE l.country_id = 'ES' RETURN l.id, l.family")
        print("• MATCH (l:Language) WHERE l.macroarea = 'Africa' RETURN DISTINCT l.family")
        print()
        print("💡 Special commands: 'schema' (database structure), 'examples' (more queries)")
        print()
        
        cypher = input("Enter Cypher query: ").strip()
        
        if not cypher:
            print("❌ No query entered")
            return
        
        if cypher.lower() == 'schema':
            self._show_schema_info()
            return
        
        if cypher.lower() == 'examples':
            self._show_cypher_examples()
            return
        
        try:
            print("🤔 Executing query...")
            result = self.query_cypher_direct(cypher)
            
            if result:
                print(f"✅ Results ({len(result)} rows):")
                for i, row in enumerate(result[:15]):  # Show first 15 results
                    print(f"   {i+1}. {row}")
                if len(result) > 15:
                    print(f"   ... and {len(result)-15} more results")
            else:
                print("✅ Query executed successfully. No results returned.")
            print()
            
        except Exception as e:
            print(f"❌ Query error: {e}")
            print("💡 Check Cypher syntax")
            print()
    
    def _display_result(self, result):
        """Display query result in a formatted way."""
        if result and result.get('answer'):
            print(f"\n✅ Answer:\n{result['answer']}\n")
        else:
            print("\n❌ Could not find an answer to that question.\n")
    
    def _handle_geographic_query(self, location):
        """Handle geographic queries with intelligent country mapping via LLM."""
        print(f"\n🌍 Languages in {location}")
        print("=" * 50)
        
        # First try macroarea (regions) - these are straightforward
        macroarea_query = f"MATCH (l:Language) WHERE LOWER(l.macroarea) = LOWER('{location}') RETURN l.id ORDER BY l.id LIMIT 20"
        result = self.query_cypher_silent(macroarea_query)
        
        if result and len(result) > 0:
            print(f"Found {len(result)} languages in {location} region:")
            for i, row in enumerate(result, 1):
                print(f"   {i}. {row['l.id']}")
            if len(result) == 20:
                print("   ... (showing first 20 results)")
            print()
            return
        
        # If not a region, use LLM to intelligently map country name to country code
        if self.qa_chain:
            try:
                # Create a prompt to help LLM understand the task
                country_mapping_prompt = f"""
                Find languages spoken in {location}. 
                
                The database uses country_id field with various formats:
                - ISO codes: "ES", "FR", "DE", "IT", "US", "MX", "IN", etc.
                - Full names: "United States", "Indonesia", "India", etc.
                - Mixed formats depending on the country
                
                Write a Cypher query that tries multiple possible country identifiers for {location}.
                Use OR conditions to check different possible formats:
                - ISO 2-letter codes (e.g., DE for Germany, IT for Italy, FR for France)
                - Full country names
                - Partial matches with CONTAINS
                
                Example for Germany: l.country_id = 'DE' OR l.country_id = 'Germany' OR l.country_id CONTAINS 'German'
                Example for Italy: l.country_id = 'IT' OR l.country_id = 'Italy' OR l.country_id CONTAINS 'Ital'
                
                Format: MATCH (l:Language) WHERE [multiple OR conditions] RETURN l.id ORDER BY l.id LIMIT 20
                """
                
                result = self.qa_chain.invoke({"query": country_mapping_prompt})
                
                # The result contains both the generated Cypher and the data
                if 'intermediate_steps' in result:
                    for step in result['intermediate_steps']:
                        if 'context' in step and step['context']:
                            # We have results! Display them
                            languages = step['context']
                            if languages and len(languages) > 0:
                                print(f"Found {len(languages)} languages:")
                                for i, lang_data in enumerate(languages, 1):
                                    # Handle different possible field names in the result
                                    language_name = lang_data.get('l.id', lang_data.get('language_name', lang_data.get('name', str(lang_data))))
                                    print(f"   {i}. {language_name}")
                                if len(languages) == 20:
                                    print("   ... (showing first 20 results)")
                                print()
                                return
                        
            except Exception as e:
                pass  # Fall back to simple search if LLM fails
        
        # Fallback: try simple country name matching
        simple_queries = [
            f"MATCH (l:Language) WHERE LOWER(l.country_id) CONTAINS LOWER('{location}') RETURN l.id ORDER BY l.id LIMIT 20",
            f"MATCH (l:Language) WHERE l.country_id = '{location.upper()}' RETURN l.id ORDER BY l.id LIMIT 20",
            f"MATCH (l:Language) WHERE l.country_id = '{location}' RETURN l.id ORDER BY l.id LIMIT 20"
        ]
        
        for query in simple_queries:
            result = self.query_cypher_silent(query)
            if result and len(result) > 0:
                print(f"Found {len(result)} languages:")
                for i, row in enumerate(result, 1):
                    print(f"   {i}. {row['l.id']}")
                if len(result) == 20:
                    print("   ... (showing first 20 results)")
                print()
                return
        
        # If nothing found
        print(f"❌ No languages found for '{location}'")
        print("💡 Try: Africa, Spain, France, Germany, Indonesia, Mexico, USA")
        print("💡 Or regions: Africa, Australia, Eurasia, North America, Papunesia, South America")
        print()
    
    def _handle_family_query(self, family_name):
        """Handle linguistic family queries with clean output."""
        print(f"\n👨‍👩‍👧‍👦 Languages in {family_name} family")
        print("=" * 50)
        
        # Family name mappings for common variations
        family_mappings = {
            'indoeuropeo': 'Indo-European',
            'indoeuropean': 'Indo-European',
            'indo-european': 'Indo-European',
            'indoeuropea': 'Indo-European',
            'romance': 'Romance',
            'romances': 'Romance',
            'románico': 'Romance',
            'románicas': 'Romance',
            'germanic': 'Germanic',
            'germanico': 'Germanic',
            'germánico': 'Germanic',
            'germánicas': 'Germanic',
            'nigercongoese': 'Niger-Congo',
            'nigercongo': 'Niger-Congo',
            'niger-congo': 'Niger-Congo',
            'afroasiatic': 'Afro-Asiatic',
            'afro-asiatic': 'Afro-Asiatic',
            'afroasiático': 'Afro-Asiatic',
            'sinotibetan': 'Sino-Tibetan',
            'sino-tibetan': 'Sino-Tibetan',
            'sino-tibetano': 'Sino-Tibetan',
            'austronesian': 'Austronesian',
            'austronesio': 'Austronesian',
            'austronésico': 'Austronesian',
            'semitic': 'Semitic',
            'semítico': 'Semitic',
            'bantu': 'Bantu',
            'bantú': 'Bantu',
            'celtic': 'Celtic',
            'céltico': 'Celtic',
            'slavic': 'Slavic',
            'eslavo': 'Slavic'
        }
        
        family_lower = family_name.lower().strip()
        
        # Try mapped family names first
        if family_lower in family_mappings:
            mapped_name = family_mappings[family_lower]
            print(f"💡 Searching for: {mapped_name}")
            
            # Try genus field first (for specific groups like Romance, Germanic, etc.)
            genus_query = f"MATCH (l:Language) WHERE l.genus = '{mapped_name}' RETURN l.id ORDER BY l.id LIMIT 20"
            result = self.query_cypher_silent(genus_query)
            
            if result and len(result) > 0:
                print(f"Found {len(result)} languages in {mapped_name} group:")
                for i, row in enumerate(result, 1):
                    print(f"   {i}. {row['l.id']}")
                if len(result) == 20:
                    print("   ... (showing first 20 results)")
                print()
                return
            
            # Try family field
            family_query = f"MATCH (l:Language) WHERE l.family = '{mapped_name}' RETURN l.id ORDER BY l.id LIMIT 20"
            result = self.query_cypher_silent(family_query)
            
            if result and len(result) > 0:
                print(f"Found {len(result)} languages in {mapped_name} family:")
                for i, row in enumerate(result, 1):
                    print(f"   {i}. {row['l.id']}")
                if len(result) == 20:
                    print("   ... (showing first 20 results)")
                print()
                return
        
        # Try original name with flexible matching
        # Try family field first
        family_query = f"MATCH (l:Language) WHERE LOWER(l.family) CONTAINS LOWER('{family_name}') RETURN l.id ORDER BY l.id LIMIT 20"
        result = self.query_cypher_silent(family_query)
        
        if result and len(result) > 0:
            print(f"Found {len(result)} languages in {family_name} family:")
            for i, row in enumerate(result, 1):
                print(f"   {i}. {row['l.id']}")
            if len(result) == 20:
                print("   ... (showing first 20 results)")
        else:
            # Try genus field (for specific groups like Romance, Germanic, etc.)
            genus_query = f"MATCH (l:Language) WHERE LOWER(l.genus) CONTAINS LOWER('{family_name}') RETURN l.id ORDER BY l.id LIMIT 20"
            result = self.query_cypher_silent(genus_query)
            
            if result and len(result) > 0:
                print(f"Found {len(result)} languages in {family_name} group:")
                for i, row in enumerate(result, 1):
                    print(f"   {i}. {row['l.id']}")
                if len(result) == 20:
                    print("   ... (showing first 20 results)")
            else:
                print(f"❌ No languages found for '{family_name}'")
                print("💡 Try: Romance, Germanic, Indo-European, Niger-Congo, Austronesian")
                print("💡 Or in Spanish: indoeuropeo, románico, germánico, austronesio")
        print()
    
    def _show_data_coverage(self):
        """Show data coverage statistics."""
        print("\n📊 Data Coverage by Field:")
        print("-" * 40)
        
        coverage_queries = [
            ("Total languages", "MATCH (l:Language) RETURN count(l) as total"),
            ("With ISO code", "MATCH (l:Language) WHERE l.iso_code IS NOT NULL AND l.iso_code <> 'nan' RETURN count(l) as count"),
            ("With country", "MATCH (l:Language) WHERE l.country_id IS NOT NULL AND l.country_id <> 'nan' RETURN count(l) as count"),
            ("With macroarea", "MATCH (l:Language) WHERE l.macroarea IS NOT NULL AND l.macroarea <> 'nan' RETURN count(l) as count"),
            ("With family", "MATCH (l:Language) WHERE l.family IS NOT NULL AND l.family <> 'nan' RETURN count(l) as count"),
            ("With subfamily", "MATCH (l:Language) WHERE l.subfamily IS NOT NULL AND l.subfamily <> 'nan' RETURN count(l) as count"),
            ("With genus", "MATCH (l:Language) WHERE l.genus IS NOT NULL AND l.genus <> 'nan' RETURN count(l) as count")
        ]
        
        total_languages = None
        
        for label, query in coverage_queries:
            try:
                result = self.query_cypher_direct(query)
                if result:
                    count = result[0]['total'] if 'total' in result[0] else result[0]['count']
                    if total_languages is None:
                        total_languages = count
                        print(f"   {label}: {count:,}")
                    else:
                        percentage = (count / total_languages) * 100
                        print(f"   {label}: {count:,} ({percentage:.1f}%)")
                else:
                    print(f"   {label}: Error")
            except Exception as e:
                print(f"   {label}: Error - {e}")
        print()
    
    def _show_schema_info(self):
        """Show Neo4j schema information."""
        print("\n📋 Esquema de la Base de Datos:")
        print("-" * 40)
        print("Nodos: Language")
        print("\nPropiedades principales:")
        print("• id (nombre de la lengua)")
        print("• iso_code (código ISO)")
        print("• country_id (código del país)")
        print("• macroarea (región geográfica)")
        print("• family (familia lingüística)")
        print("• subfamily (subfamilia)")
        print("• genus (clasificación específica)")
        print("• latitude, longitude (coordenadas)")
        print("\nEjemplo de nodo:")
        print("(:Language {id: 'Spanish', family: 'Indo-European', genus: 'Romance'})")
        print()
    
    def _show_cypher_examples(self):
        """Show example Cypher queries."""
        print("\n📝 Cypher Query Examples:")
        print("-" * 40)
        
        examples = [
            ("Contar lenguas por familia", "MATCH (l:Language) WHERE l.family IS NOT NULL RETURN l.family, count(l) as count ORDER BY count DESC LIMIT 10"),
            ("Lenguas en un país específico", "MATCH (l:Language) WHERE l.country_id = 'ES' RETURN l.id, l.family LIMIT 10"),
            ("Lenguas por macroárea", "MATCH (l:Language) WHERE l.macroarea = 'Africa' RETURN count(l) as total"),
            ("Familias más diversas", "MATCH (l:Language) WHERE l.family IS NOT NULL RETURN l.family, count(DISTINCT l.country_id) as countries ORDER BY countries DESC LIMIT 5"),
            ("Lenguas Romance", "MATCH (l:Language) WHERE l.genus = 'Romance' RETURN l.id, l.country_id LIMIT 10"),
            ("Coordenadas de lenguas", "MATCH (l:Language) WHERE l.latitude IS NOT NULL RETURN l.id, l.latitude, l.longitude LIMIT 5")
        ]
        
        for i, (description, query) in enumerate(examples, 1):
            print(f"\n{i}. {description}:")
            print(f"   {query}")
        print()


def main():
    """Main function for testing graph explorer."""
    explorer = GraphExplorer()
    
    print("🔍 Graph Explorer Test")
    print("=" * 30)
    
    # Initialize
    if not explorer.connect_to_neo4j():
        return
    
    if not explorer.setup_qa_chain():
        return
    
    # Test queries
    test_queries = [
        "What languages are spoken in Spain?",
        "List all Romance languages",
        "What languages belong to the Indo-European family?"
    ]
    
    for question in test_queries:
        print(f"\n🔍 Testing: {question}")
        result = explorer.query_natural_language(question)
        if result:
            print(f"✅ Answer: {result['answer']}")
        print("-" * 50)
    
    # Direct query test
    print("\n📊 Spain languages (direct query):")
    spain_langs = explorer.get_languages_in_country("Spain")
    if spain_langs:
        for lang in spain_langs[:5]:  # Show first 5
            print(f"   {lang['language']} ({lang['family']})")
        print(f"   ... and {len(spain_langs) - 5} more") if len(spain_langs) > 5 else None
    
    # Overview
    print("\n📈 Graph Overview:")
    overview = explorer.get_graph_overview()
    for key, value in overview.items():
        if isinstance(value, list) and value:
            if key in ['total_languages', 'total_countries', 'total_families']:
                print(f"   {key}: {value[0]['count']}")
            else:
                print(f"   {key}: {len(value)} items")

if __name__ == "__main__":
    main()