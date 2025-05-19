"""
Enhanced Named Entity Recognition (NER) Analysis Tool
This tool uses spaCy to perform advanced NER on text, with additional features:
- Entity extraction and classification
- Relationship extraction between entities
- Entity frequency analysis
- Custom entity recognition
- Interactive visualization
- Sentiment analysis for entities
- Export options (CSV, JSON)
- Batch processing capabilities
"""

# Import necessary libraries
import spacy
from spacy import displacy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import os
from collections import Counter
from tqdm import tqdm  # For progress bars
import networkx as nx  # For relationship graphs
from textblob import TextBlob  # For sentiment analysis

class NERAnalyzer:
    def __init__(self, model="en_core_web_sm"):
        """Initialize the NER analyzer with the specified model"""
        print(f"Loading spaCy model: {model}...")
        self.nlp = spacy.load(model)
        self.custom_entities = {}
        print("NER Analyzer initialized successfully!")
        
    def add_custom_entity(self, entity_name, patterns):
        """Add custom entities to the NER system"""
        if "entity_ruler" not in self.nlp.pipe_names:
            self.nlp.add_pipe("entity_ruler", before="ner")
            
        ruler = self.nlp.get_pipe("entity_ruler")
        pattern_list = [{"label": entity_name, "pattern": pattern} for pattern in patterns]
        ruler.add_patterns(pattern_list)
        self.custom_entities[entity_name] = patterns
        print(f"Added custom entity type: {entity_name} with {len(patterns)} patterns")
        
    def process_text(self, text):
        """Process text with spaCy NLP pipeline"""
        return self.nlp(text)
    
    def extract_entities(self, doc):
        """Extract entities from processed document"""
        entities = []
        for ent in doc.ents:
            entities.append({
                'Entity': ent.text,
                'Label': ent.label_,
                'Start': ent.start_char,
                'End': ent.end_char,
                'Explanation': spacy.explain(ent.label_)
            })
        return pd.DataFrame(entities)
    
    def analyze_entity_frequency(self, entities_df):
        """Analyze frequency of entities by type"""
        return entities_df.groupby('Label').size().sort_values(ascending=False)
    
    def extract_relationships(self, doc):
        """Extract potential relationships between entities"""
        relationships = []
        entities = list(doc.ents)
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Find sentences containing both entities
                for sent in doc.sents:
                    if entity1.start < sent.end and entity2.start < sent.end and entity1.end > sent.start and entity2.end > sent.start:
                        # Simple relationship extraction based on proximity
                        words_between = len([token for token in doc[entity1.end:entity2.start] if not token.is_punct and not token.is_space])
                        
                        if words_between < 10:  # If entities are close to each other
                            relationships.append({
                                'Entity1': entity1.text,
                                'Entity1_Type': entity1.label_,
                                'Entity2': entity2.text,
                                'Entity2_Type': entity2.label_,
                                'Distance': words_between,
                                'Sentence': sent.text
                            })
        
        return pd.DataFrame(relationships) if relationships else pd.DataFrame()
    
    def analyze_entity_sentiment(self, doc, entities_df):
        """Analyze sentiment around each entity"""
        sentiments = []
        
        for _, row in entities_df.iterrows():
            entity = row['Entity']
            start = row['Start']
            end = row['End']
            
            # Get text window around entity (50 chars on each side)
            context_start = max(0, start - 50)
            context_end = min(len(doc.text), end + 50)
            context = doc.text[context_start:context_end]
            
            # Calculate sentiment using TextBlob
            sentiment = TextBlob(context).sentiment.polarity
            
            sentiments.append({
                'Entity': entity,
                'Label': row['Label'],
                'Sentiment': sentiment,
                'Context': context
            })
            
        return pd.DataFrame(sentiments)
    
    def visualize_entities(self, doc, jupyter=True):
        """Visualize named entities in the document"""
        return displacy.render(doc, style="ent", jupyter=jupyter)
    
    def visualize_entity_distribution(self, entities_df):
        """Create a bar chart of entity types"""
        plt.figure(figsize=(12, 6))
        entity_counts = entities_df['Label'].value_counts()
        sns.barplot(x=entity_counts.index, y=entity_counts.values)
        plt.title('Distribution of Entity Types')
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('entity_distribution.png')
        plt.close()
        return 'entity_distribution.png'
    
    def visualize_entity_network(self, relationships_df):
        """Create a network graph of entity relationships"""
        if relationships_df.empty:
            return None
            
        G = nx.Graph()
        
        # Add nodes and edges
        for _, row in relationships_df.iterrows():
            G.add_node(row['Entity1'], type=row['Entity1_Type'])
            G.add_node(row['Entity2'], type=row['Entity2_Type'])
            G.add_edge(row['Entity1'], row['Entity2'], weight=1/max(1, row['Distance']))
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G)
        
        # Get node colors based on entity type
        node_types = {node: data['type'] for node, data in G.nodes(data=True)}
        unique_types = list(set(node_types.values()))
        color_map = {t: plt.cm.tab10(i/len(unique_types)) for i, t in enumerate(unique_types)}
        node_colors = [color_map[node_types[node]] for node in G.nodes()]
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color_map[t], label=t, markersize=10) 
                          for t in unique_types]
        plt.legend(handles=legend_elements)
        
        plt.title('Entity Relationship Network')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('entity_network.png')
        plt.close()
        return 'entity_network.png'
    
    def export_data(self, data, filename, format='csv'):
        """Export processed data to file"""
        if format.lower() == 'csv':
            data.to_csv(filename, index=False)
        elif format.lower() == 'json':
            data.to_json(filename, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'")
        return filename
    
    def batch_process_files(self, directory, pattern=r'.*\.txt'):
        """Process all text files in a directory"""
        results = []
        
        # Get all matching files
        files = [f for f in os.listdir(directory) if re.match(pattern, f)]
        
        for filename in tqdm(files, desc="Processing files"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
                
                doc = self.process_text(text)
                entities = self.extract_entities(doc)
                
                results.append({
                    'filename': filename,
                    'doc': doc,
                    'entities': entities
                })
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        
        return results
    
    def summarize_entities(self, entities_df):
        """Generate a summary of extracted entities"""
        summary = []
        
        # Group entities by type
        grouped = entities_df.groupby('Label')
        
        for entity_type, group in grouped:
            # Get unique entities of this type
            unique_entities = group['Entity'].unique()
            
            # Create summary entry
            summary.append({
                'Entity_Type': entity_type,
                'Count': len(group),
                'Unique_Count': len(unique_entities),
                'Examples': ', '.join(unique_entities[:5])
            })
        
        return pd.DataFrame(summary)

# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = NERAnalyzer()
    
    # Add custom entity types (examples)
    analyzer.add_custom_entity("TECH_PRODUCT", ["iPhone", "Galaxy", "Azure", "AWS", "Google Cloud"])
    analyzer.add_custom_entity("CRYPTOCURRENCY", ["Bitcoin", "Ethereum", "Dogecoin", "NFT"])
    
    # Sample text for NER
    text = """
    Amazon announced its quarterly earnings on July 30, 2023. 
    CEO Andy Jassy said the company is investing $4 billion in AI technology. 
    Google, based in Mountain View, California, also shared its financial report. 
    The 2024 Summer Olympics will be held in Paris, France.
    Apple's new iPhone 15 is expected to launch in September, while Samsung's Galaxy S23 
    is currently available in stores. Microsoft Azure and AWS are competing for cloud market share.
    Bitcoin reached an all-time high last month, and Ethereum is also gaining value.
    """
    
    # Process the text
    print("Processing text...")
    doc = analyzer.process_text(text)
    
    # Extract entities
    print("\nExtracting entities...")
    entities_df = analyzer.extract_entities(doc)
    print("\nExtracted Named Entities:")
    print(entities_df)
    
    # Analyze entity frequency
    print("\nEntity frequency analysis:")
    frequency = analyzer.analyze_entity_frequency(entities_df)
    print(frequency)
    
    # Extract relationships
    print("\nExtracting relationships between entities...")
    relationships_df = analyzer.extract_relationships(doc)
    if not relationships_df.empty:
        print("\nExtracted Relationships:")
        print(relationships_df[['Entity1', 'Entity2', 'Distance']])
    else:
        print("No significant relationships found.")
    
    # Analyze sentiment
    print("\nAnalyzing sentiment around entities...")
    sentiment_df = analyzer.analyze_entity_sentiment(doc, entities_df)
    print("\nEntity Sentiment Analysis:")
    print(sentiment_df[['Entity', 'Label', 'Sentiment']])
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.visualize_entities(doc, jupyter=True)
    dist_viz = analyzer.visualize_entity_distribution(entities_df)
    print(f"Entity distribution chart saved as: {dist_viz}")
    
    if not relationships_df.empty:
        net_viz = analyzer.visualize_entity_network(relationships_df)
        print(f"Entity network visualization saved as: {net_viz}")
    
    # Generate summary
    print("\nGenerating entity summary...")
    summary_df = analyzer.summarize_entities(entities_df)
    print(summary_df)
    
    # Export data
    print("\nExporting data to files...")
    analyzer.export_data(entities_df, "extracted_entities.csv")
    analyzer.export_data(entities_df, "extracted_entities.json", format="json")
    analyzer.export_data(sentiment_df, "entity_sentiment.csv")
    
    if not relationships_df.empty:
        analyzer.export_data(relationships_df, "entity_relationships.csv")
    
    print("\nAnalysis complete! Results saved to CSV and JSON files.")
    
    # Example of batch processing (commented out)
    """
    print("\nBatch processing example (uncomment to run):")
    # Create a directory for text files
    if not os.path.exists("sample_texts"):
        os.makedirs("sample_texts")
        
    # Example batch processing
    batch_results = analyzer.batch_process_files("sample_texts")
    print(f"Processed {len(batch_results)} files")
    """