import json
import networkx as nx
import graphviz
from pathlib import Path

def load_semantic_tree(input_file: str = 'semantic_treev3.json') -> nx.DiGraph:
    """Load semantic tree from JSON file with hierarchical structure"""
    with open(input_file, 'r', encoding='utf-8') as f:
        tree_data = json.load(f)
    
    G = nx.DiGraph()
    
    def add_node_and_children(node_data, parent_label=None):
        """Recursively add nodes and their children to the graph"""
        current_label = node_data['label']
        G.add_node(current_label)
        
        if parent_label:
            G.add_edge(parent_label, current_label)
            
        # Recursively process children
        for child in node_data['children']:
            add_node_and_children(child, current_label)
    
    # Start processing from root
    add_node_and_children(tree_data)
    return G

def visualize_tree(G: nx.DiGraph, output_file: str = 'semantic_tree_viz.pdf'):
    """Visualize the tree using graphviz"""
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Semantic Tree')
    dot.attr(rankdir='TB')  # Top to bottom layout
    
    # Add nodes
    for node in G.nodes():
        dot.node(str(node), str(node), 
                style='filled',
                fillcolor='lightblue',
                shape='box',
                fontname='Arial')
    
    # Add edges
    for edge in G.edges():
        dot.edge(str(edge[0]), str(edge[1]))
    
    # Save the visualization
    dot.render(output_file.replace('.pdf', ''), format='pdf', cleanup=True)
    print(f"Visualization saved as {output_file}")

def print_tree_stats(G: nx.DiGraph):
    """打印树的统计信息"""
    print("\nTree Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # 找出根节点（入度为0的节点）
    root_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    print(f"Root nodes: {root_nodes}")
    
    # 找出叶子节点（出度为0的节点）
    leaf_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
    print(f"Number of leaf nodes: {len(leaf_nodes)}")
    
    # 计算树的深度
    depths = []
    for root in root_nodes:
        for leaf in leaf_nodes:
            if nx.has_path(G, root, leaf):
                path_length = len(nx.shortest_path(G, root, leaf)) - 1
                depths.append(path_length)
    
    if depths:
        max_depth = max(depths)
        print(f"Maximum tree depth: {max_depth}")
    else:
        print("Warning: Could not calculate tree depth - no valid paths from roots to leaves")

def main():
    # 确保输入文件存在
    input_file = 'semantic_treev3.json'
    if not Path(input_file).exists():
        print(f"Error: {input_file} not found!")
        return
    
    # 加载树
    try:
        G = load_semantic_tree(input_file)
    except json.JSONDecodeError:
        print(f"Error: {input_file} is not a valid JSON file!")
        return
    except Exception as e:
        print(f"Error loading tree: {e}")
        return
    
    # 打印树的统计信息
    print_tree_stats(G)
    
    # 可视化树
    try:
        visualize_tree(G, 'semantic_tree_viz.pdf')
    except Exception as e:
        print(f"Error visualizing tree: {e}")
        return

if __name__ == "__main__":
    main()