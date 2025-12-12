"""
Knowledge Graph Visualization Tool

This script creates visual representations of JSON knowledge graphs using matplotlib and networkx.
Perfect for understanding the structure and relationships in your KAG systems.

Usage:
    python visualize_graph.py knowledge/movie_graph.json
    python visualize_graph.py music_knowledge/music_graph.json
"""

import json
import sys
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

def load_knowledge_graph(file_path):
    """Load knowledge graph from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{file_path}': {e}")
        return None

def create_graph_visualization(graph_data, title="Knowledge Graph"):
    """Create a visual representation of the knowledge graph."""
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes from entities
    entities = graph_data.get('entities', {})
    for entity_id, entity_data in entities.items():
        entity_type = entity_data.get('type', 'Unknown')
        G.add_node(entity_id, type=entity_type, data=entity_data)
    
    # Add edges from relationships
    relationships = graph_data.get('relationships', [])
    for rel in relationships:
        subject = rel.get('subject')
        predicate = rel.get('predicate')
        obj = rel.get('object')
        
        if subject and obj:
            G.add_edge(subject, obj, relation=predicate)
    
    # Set up the plot
    plt.figure(figsize=(16, 12))
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Choose layout algorithm
    if len(G.nodes()) <= 10:
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # Define colors for different entity types
    type_colors = {
        'PERSON': '#FF6B6B',      # Red for people
        'MOVIE': '#4ECDC4',       # Teal for movies
        'ALBUM': '#45B7D1',       # Blue for albums
        'ARTIST': '#96CEB4',      # Green for artists
        'GENRE': '#FFEAA7',       # Yellow for genres
        'Unknown': '#DDA0DD'      # Purple for unknown
    }
    
    # Get node colors based on type
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'Unknown')
        node_colors.append(type_colors.get(node_type, '#DDA0DD'))
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=3000,
        alpha=0.8,
        edgecolors='black',
        linewidths=2
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        width=2,
        alpha=0.6,
        connectionstyle="arc3,rad=0.1"
    )
    
    # Draw node labels (entity names)
    node_labels = {}
    for node in G.nodes():
        # Clean up the label for display
        clean_name = node.replace('_', '\n')
        if len(clean_name) > 15:
            clean_name = clean_name[:12] + '...'
        node_labels[node] = clean_name
    
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=9,
        font_weight='bold',
        font_color='white'
    )
    
    # Draw edge labels (relationships)
    edge_labels = {}
    for edge in G.edges():
        relation = G.edges[edge].get('relation', '')
        if relation:
            edge_labels[edge] = relation
    
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color='blue',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
    )
    
    # Create legend
    legend_elements = []
    for entity_type, color in type_colors.items():
        if any(G.nodes[node].get('type') == entity_type for node in G.nodes()):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=15, 
                                            label=entity_type))
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Remove axes
    plt.axis('off')
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    return G

def print_graph_stats(graph_data):
    """Print statistics about the knowledge graph."""
    entities = graph_data.get('entities', {})
    relationships = graph_data.get('relationships', [])
    
    print(f"\n Knowledge Graph Statistics:")
    print(f"   Entities: {len(entities)}")
    print(f"   Relationships: {len(relationships)}")
    
    # Count entity types
    type_counts = {}
    for entity_data in entities.values():
        entity_type = entity_data.get('type', 'Unknown')
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
    
    print(f"   Entity Types:")
    for entity_type, count in type_counts.items():
        print(f"     {entity_type}: {count}")
    
    # Count relationship types
    relation_counts = {}
    for rel in relationships:
        predicate = rel.get('predicate', 'Unknown')
        relation_counts[predicate] = relation_counts.get(predicate, 0) + 1
    
    print(f"   Relationship Types:")
    for relation_type, count in relation_counts.items():
        print(f"     {relation_type}: {count}")

def create_detailed_view(graph_data, output_file="graph_details.txt"):
    """Create a detailed text view of the graph structure."""
    entities = graph_data.get('entities', {})
    relationships = graph_data.get('relationships', [])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("KNOWLEDGE GRAPH DETAILED VIEW\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("ENTITIES:\n")
        f.write("-" * 20 + "\n")
        for entity_id, entity_data in entities.items():
            f.write(f"\n{entity_id}:\n")
            for key, value in entity_data.items():
                f.write(f"  {key}: {value}\n")
        
        f.write(f"\n\nRELATIONSHIPS:\n")
        f.write("-" * 20 + "\n")
        for rel in relationships:
            subject = rel.get('subject', 'Unknown')
            predicate = rel.get('predicate', 'Unknown')
            obj = rel.get('object', 'Unknown')
            f.write(f"{subject} --{predicate}--> {obj}\n")
    
    print(f" Detailed view saved to: {output_file}")

def main():
    """Main function to run the visualization."""
    
    # Check if file path provided
    if len(sys.argv) != 2:
        print("Usage: python visualize_graph.py <path_to_graph.json>")
        print("\nExamples:")
        print("  python visualize_graph.py knowledge/movie_graph.json")
        print("  python visualize_graph.py music_knowledge/music_graph.json")
        return
    
    file_path = sys.argv[1]
    
    # Load the knowledge graph
    print(f" Loading knowledge graph from: {file_path}")
    graph_data = load_knowledge_graph(file_path)
    
    if graph_data is None:
        return
    
    # Print statistics
    print_graph_stats(graph_data)
    
    # Create visualization
    print(f"\n Creating visualization...")
    
    # Get a nice title from the file name
    file_name = Path(file_path).stem
    title = f"Knowledge Graph: {file_name.replace('_', ' ').title()}"
    
    try:
        G = create_graph_visualization(graph_data, title)
        
        # Save the plot
        output_file = f"{file_name}_visualization.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f" Visualization saved as: {output_file}")
        
        # Create detailed text view
        detail_file = f"{file_name}_details.txt"
        create_detailed_view(graph_data, detail_file)
        
        # Show the plot
        plt.show()
        
        print(f"\n Visualization complete!")
        print(f"   Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
    except ImportError as e:
        print(f" Error: Missing required library.")
        print(f"Please install the required packages:")
        print(f"pip install matplotlib networkx")
        print(f"\nDetailed error: {e}")
    except Exception as e:
        print(f" Error creating visualization: {e}")

if __name__ == "__main__":
    main()