"""
WALS Data Processing
===================

Load, process, and prepare WALS linguistic data for graph construction.
Handles all data transformations and chunk generation.
"""

import pandas as pd
import os
import shutil
import urllib.request
import zipfile
from tqdm import tqdm

class WALSDataProcessor:
    """Handles all WALS data loading and processing."""
    
    def __init__(self):
        self.data = {}
        self.output_dir = 'output'
        self.data_dir = 'data'
        self.wals_github_base = 'https://raw.githubusercontent.com/cldf-datasets/wals/master/cldf'
        
    def download_and_extract_wals(self):
        """Download WALS data from GitHub CLDF repository."""
        print("📥 Downloading WALS data from GitHub...")
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Files to download from CLDF-WALS GitHub
        files_to_download = {
            'languages.csv': f'{self.wals_github_base}/languages.csv',
            'countries.csv': f'{self.wals_github_base}/codes.csv',  # Countries are in codes.csv
            'parameters.csv': f'{self.wals_github_base}/parameters.csv',
            'values.csv': f'{self.wals_github_base}/values.csv'
        }
        
        # Check if files already exist
        if all(os.path.exists(f"{self.data_dir}/{file}") for file in files_to_download.keys()):
            print("✅ WALS CSV files already exist in data/ directory")
            return True
        
        # Download each file
        for filename, url in files_to_download.items():
            target_path = f"{self.data_dir}/{filename}"
            
            if os.path.exists(target_path):
                print(f"✅ {filename} already exists")
                continue
                
            try:
                print(f"⬇️  Downloading {filename}...")
                urllib.request.urlretrieve(url, target_path)
                print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
                return False
        
        # Download codes.csv for value definitions
        try:
            codes_url = f'{self.wals_github_base}/codes.csv'
            urllib.request.urlretrieve(codes_url, f"{self.data_dir}/codes.csv")
            print("✅ Downloaded codes.csv")
        except Exception as e:
            print(f"⚠️  Could not download codes.csv: {e}")
        
        # Process countries data (it's in codes.csv format)
        self._process_countries_data()
        
        print("✅ WALS data download completed")
        return True
    
    def _process_countries_data(self):
        """Process countries data from codes.csv format."""
        try:
            # Read the codes.csv which contains country information
            codes_path = f"{self.data_dir}/countries.csv"
            if os.path.exists(codes_path):
                codes_df = pd.read_csv(codes_path)
                
                # Filter for country codes and create a proper countries.csv
                if 'Name' in codes_df.columns and 'ID' in codes_df.columns:
                    # Keep only relevant country data
                    countries_df = codes_df[['ID', 'Name']].drop_duplicates()
                    countries_df.to_csv(f"{self.data_dir}/countries.csv", index=False)
                    print("✅ Processed countries data")
                else:
                    print("⚠️  Countries data format unexpected")
                    
        except Exception as e:
            print(f"⚠️  Could not process countries data: {e}")
        
    def load_data(self):
        """Load all WALS data files."""
        print("📊 Loading WALS dataset...")
        
        # First, ensure we have the data files
        if not self.download_and_extract_wals():
            print("❌ Failed to obtain WALS data")
            return False
        
        try:
            # Core data files
            self.data['languages'] = pd.read_csv('data/languages.csv')
            self.data['countries'] = pd.read_csv('data/countries.csv')
            
            print(f"✅ Loaded {len(self.data['languages'])} languages")
            print(f"✅ Loaded {len(self.data['countries'])} countries")
            
            # Optional WALS feature files
            optional_files = {
                'parameters': 'data/parameters.csv',
                'values': 'data/values.csv',
                'codes': 'data/codes.csv'
            }
            
            for key, filename in optional_files.items():
                if os.path.exists(filename):
                    self.data[key] = pd.read_csv(filename)
                    print(f"✅ Loaded {filename}: {len(self.data[key])} records")
                else:
                    print(f"⚠️  Optional file {filename} not found")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading WALS data: {e}")
            return False
    
    def setup_output_structure(self):
        """Create clean output directory structure."""
        print("📁 Setting up output structure...")
        
        # Create directories
        directories = [
            'output',
            'output/chunks', 
            'output/logs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Clean ALL old chunk files (comprehensive cleanup)
        chunk_patterns = [
            'massive_chunk_*.txt',
            'chunk_*.txt', 
            'enhanced_chunk_*.txt',
            'wals_chunk_*.txt'
        ]
        
        old_files = []
        # Clean from root directory
        for file in os.listdir('.'):
            for pattern in chunk_patterns:
                if file.startswith(pattern.replace('*.txt', '')) and file.endswith('.txt'):
                    old_files.append(file)
        
        # Clean from output/chunks directory
        chunks_dir = 'output/chunks'
        if os.path.exists(chunks_dir):
            for file in os.listdir(chunks_dir):
                if file.endswith('.txt'):
                    old_files.append(os.path.join(chunks_dir, file))
        
        if old_files:
            print(f"🧹 Removing {len(old_files)} old chunk files...")
            for file in old_files:
                try:
                    os.remove(file)
                except:
                    pass  # Ignore errors
        
        # Clean old log files
        log_files = ['chunk_list.txt', 'massive_chunk_list.txt', 'enhanced_chunk_list.txt']
        for file in log_files:
            # Check both root and logs directory
            for location in [file, f'output/logs/{file}']:
                if os.path.exists(location):
                    os.remove(location)
        
        print("✅ Output structure clean and ready")
    
    def generate_chunks(self, batch_size=10):
        """Generate chunks with complete WALS linguistic data."""
        print("🧩 Generating linguistic chunks with WALS features...")
        print(f"📏 Using batch size of {batch_size} for optimal LLM processing...")
        
        if 'languages' not in self.data:
            print("❌ No language data loaded")
            return []
        
        # Country code to name mapping for common problematic codes  
        country_names = {
            'ID': 'Indonesia',
            'US': 'United States', 
            'CA': 'Canada',
            'AU': 'Australia',
            'IN': 'India',
            'CN': 'China',
            'BR': 'Brazil',
            'AR': 'Argentina'
        }
        
        languages_df = self.data['languages']
        chunk_files = []
        chunk_num = 1
        
        # Group by language family for better organization
        families = languages_df.groupby('Family')
        
        for family_name, family_langs in tqdm(families, desc="Processing families"):
            if pd.isna(family_name):
                family_name = "UnknownFamily"
            
            # Create family-based chunks
            for i in range(0, len(family_langs), batch_size):
                batch = family_langs.iloc[i:i+batch_size]
                
                # Generate chunk with linguistic features
                chunk_text = self._create_chunk_content(batch, family_name, chunk_num, country_names)
                
                # Save chunk
                chunk_file = f"{self.output_dir}/chunks/enhanced_chunk_{chunk_num:03d}.txt"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(chunk_text)
                
                chunk_files.append(chunk_file)
                chunk_num += 1
        
        # Save enhanced chunk list  
        list_file = f'{self.output_dir}/logs/enhanced_chunk_list.txt'
        with open(list_file, 'w') as f:
            for chunk_file in chunk_files:
                f.write(f"{chunk_file}\n")
        
        print(f"✅ Generated {len(chunk_files)} chunks")
        return chunk_files
    
    def _create_chunk_content(self, languages_batch, family_name, chunk_num, country_names):
        """Create chunk with complete linguistic information."""
        
        chunk_text = f"""
LINGUISTIC KNOWLEDGE GRAPH DATA - CHUNK {chunk_num}:

Family: {family_name}
Languages in this chunk: {len(languages_batch)}

=== DETAILED LANGUAGE INFORMATION ===
"""
        
        for _, lang in languages_batch.iterrows():
            name = str(lang['Name'])
            family = str(lang.get('Family', 'Unknown'))
            subfamily = str(lang.get('Subfamily', ''))
            genus = str(lang.get('Genus', 'Unknown'))
            latitude = lang.get('Latitude', 0)
            longitude = lang.get('Longitude', 0)
            country_id = str(lang.get('Country_ID', ''))
            # Convert problematic country codes to full names
            country_name = country_names.get(country_id, country_id)
            iso_code = str(lang.get('ISO639P3code', ''))
            macroarea = str(lang.get('Macroarea', ''))
            
            # Add linguistic features if available
            feature_info = ""
            if 'values' in self.data and 'parameters' in self.data:
                feature_info = self._get_linguistic_features(lang)
            
            lang_entry = f"""
Language: {name}
- Family: {family}
- Subfamily: {subfamily}
- Genus: {genus}
- Coordinates: {latitude}, {longitude}
- Country: {country_name}
- ISO Code: {iso_code}
- Macroarea: {macroarea}
{feature_info}
"""
            chunk_text += lang_entry
        
        # Enhanced relationship instructions
        chunk_text += f"""
=== RELATIONSHIPS TO EXTRACT ===
- Each {family_name} language BELONGS_TO {family_name}Family
- Each language LOCATED_IN its country (if country specified)
- Languages of same subfamily form SUBFAMILY_OF relationships
- Languages in same macroarea form MACROAREA_OF relationships
- Create Country entities with proper names
- Create LanguageFamily entities for linguistic classification
"""
        
        return chunk_text.strip()
    
    def _get_linguistic_features(self, language):
        """Get linguistic features for a language from WALS data."""
        if 'values' not in self.data or 'codes' not in self.data:
            return ""
        
        lang_id = language['ID'] if 'ID' in language else language.name
        features = []
        
        # Get word order (81A parameter)
        word_order = self._get_feature_value(lang_id, '81A')
        if word_order:
            features.append(f"- Word Order: {word_order}")
        
        # Get other interesting features
        feature_params = {
            '82A': 'Subject-Verb Order',
            '83A': 'Object-Verb Order', 
            '85A': 'Adposition-Noun Order',
            '86A': 'Genitive-Noun Order',
            '87A': 'Adjective-Noun Order'
        }
        
        for param_id, param_name in feature_params.items():
            value = self._get_feature_value(lang_id, param_id)
            if value:
                features.append(f"- {param_name}: {value}")
        
        if features:
            return "\n" + "\n".join(features)
        else:
            return "\n- Linguistic features: Available in WALS database"
    
    def _get_feature_value(self, language_id, parameter_id):
        """Get the value of a specific parameter for a language."""
        if 'values' not in self.data or 'codes' not in self.data:
            return None
        
        # Find the value for this language and parameter
        values_df = self.data['values']
        codes_df = self.data['codes']
        
        # Look for the value
        lang_values = values_df[
            (values_df['Language_ID'] == language_id) & 
            (values_df['Parameter_ID'] == parameter_id)
        ]
        
        if not lang_values.empty:
            code_id = lang_values.iloc[0]['Code_ID']
            
            # Find the meaning in codes
            code_meaning = codes_df[codes_df['ID'] == code_id]
            if not code_meaning.empty:
                return code_meaning.iloc[0]['Name']
        
        return None
    
    def get_statistics(self):
        """Get comprehensive statistics about the WALS data."""
        if 'languages' not in self.data:
            return {}
        
        languages_df = self.data['languages']
        
        stats = {
            'total_languages': len(languages_df),
            'total_families': languages_df['Family'].nunique(),
            'total_countries': languages_df['Country_ID'].nunique(),
        }
        
        return stats

def main():
    """Main function for testing WALS data processing."""
    processor = WALSDataProcessor()
    
    print("🚀 WALS Data Processing")
    print("=" * 40)
    
    # Load data
    if not processor.load_data():
        return
    
    # Setup output structure
    processor.setup_output_structure()
    
    # Generate chunks
    chunk_files = processor.generate_chunks()
    
    print(f"\n✅ WALS processing complete!")
    print(f"📁 Generated {len(chunk_files)} chunks in: output/chunks/")
    print(f"🚀 Next: Run graph construction")

if __name__ == "__main__":
    main()