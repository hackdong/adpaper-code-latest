import json
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import davies_bouldin_score
import os
from datetime import datetime
from sklearn.decomposition import PCA
import pickle

class SemanticTreeCompressor(nn.Module):
    def __init__(self, input_dim=384, latent_dim=64):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim)
        )   
        
    def forward(self, x):
        # Get both latent representation and reconstruction
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction

def get_leaf_nodes_and_parents(node, parent_label=None):

    leaves = []
    if not node['children']:
        leaves.append({
            'label': node['label'].strip(),
            'parent': parent_label
        })
    else:
        for child in node['children']:
            leaves.extend(get_leaf_nodes_and_parents(child, node['label']))
    return leaves

def visualize_embeddings(embeddings, leaf_nodes, title, filename):


    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    

    unique_parents = list(set(node['parent'] for node in leaf_nodes))
    color_palette = sns.color_palette('husl', n_colors=len(unique_parents))
    parent_to_color = dict(zip(unique_parents, color_palette))
    

    plt.figure(figsize=(15, 10))
    for i, node in enumerate(leaf_nodes):
        color = parent_to_color[node['parent']]
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], c=[color], label=node['parent'])
        plt.annotate(node['label'], 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    fontsize=14)  
    

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
              bbox_to_anchor=(1.05, 1), 
              loc='upper left',
              prop={'size': 12})  
    
    plt.title(title, fontsize=16) 
    plt.xlabel('Dimension 1', fontsize=14) 
    plt.ylabel('Dimension 2', fontsize=14)  
    

    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300) 
    plt.close()

def calculate_tree_distance(node1_path, node2_path):

    if not node1_path or not node2_path:
        return 0  
        
    common_prefix_len = 0
    min_len = min(len(node1_path), len(node2_path))
    
    for i in range(min_len):
        if node1_path[i].strip() == node2_path[i].strip():
            common_prefix_len += 1
        else:
            break
    
    return len(node1_path) + len(node2_path) 

def get_node_path(node, semantic_tree):

    def find_node_path(current_node, target_label, current_path):
        if current_node['label'].strip() == target_label.strip():
            return current_path + [current_node['label']]
        
        if 'children' in current_node and current_node['children']:
            for child in current_node['children']:
                result = find_node_path(child, target_label, current_path + [current_node['label']])
                if result:
                    return result
        return None

    path = find_node_path(semantic_tree, node['label'], [])
    if path is None:
        print(f"Warning: Could not find path for node {node['label']}")
        return [node['label']]  
    return path

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels, weights):
        device = features.device
        batch_size = features.size(0)
        

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        

        similarity_matrix = similarity_matrix * weights
        

        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        

        loss = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        loss = -loss.mean()
        
        return loss

def calculate_dbi(embeddings, labels):
    """

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        labels: numpy array of shape (n_samples,)
    Returns:
        dbi: Davies-Bouldin index
    """
    try:
        return davies_bouldin_score(embeddings, labels)
    except ValueError as e:
        print(f"Warning: Failed to calculate DBI: {e}")
        return float('nan')

def create_output_directory():

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'training_output_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    return output_dir

def save_training_results(output_dir, history, best_epoch, compressed_embeddings, leaf_nodes):


    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'loss': [float(x) for x in history['loss']],
            'dbi': [float(x) for x in history['dbi']],
            'best_epoch': best_epoch
        }, f, indent=4)
    

    embeddings_path = os.path.join(output_dir, 'compressed_embeddings.npy')
    np.save(embeddings_path, compressed_embeddings)
    

    nodes_path = os.path.join(output_dir, 'leaf_nodes.json')
    with open(nodes_path, 'w', encoding='utf-8') as f:
        json.dump(leaf_nodes, f, ensure_ascii=False, indent=4)

def plot_training_progress(history, current_epoch, output_dir):

    plt.figure(figsize=(12, 5))
    

    plt.subplot(1, 2, 1)
    plt.plot(range(1, current_epoch + 1), history['loss'], 'b-', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    

    plt.subplot(1, 2, 2)
    plt.plot(range(1, current_epoch + 1), history['dbi'], 'r-', label='DBI')
    plt.xlabel('Epoch')
    plt.ylabel('Davies-Bouldin Index')
    plt.title('Davies-Bouldin Index Progress')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', f'training_progress_epoch_{current_epoch}.png'))
    plt.close()

def compress_embeddings(embeddings_to_compress, pca_model_path=None):

    if pca_model_path is None:

        output_dirs = [d for d in os.listdir('.') if d.startswith('training_output_')]
        if not output_dirs:
            raise FileNotFoundError("No training output directory found")
        latest_dir = max(output_dirs)
        pca_model_path = os.path.join(latest_dir, 'models', 'pca_model.pkl')
    

    with open(pca_model_path, 'rb') as f:
        pca = pickle.load(f)
    

    compressed_embeddings = pca.transform(embeddings_to_compress)
    return compressed_embeddings

def train_compressor():

    output_dir = create_output_directory()
    print(f"Created output directory: {output_dir}")
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open('semantic_treev3.json', 'r', encoding='utf-8') as f:
        semantic_tree = json.load(f)
    

    leaf_nodes = get_leaf_nodes_and_parents(semantic_tree)
    

    unique_parents = list(set(node['parent'] for node in leaf_nodes))
    print(f"Found {len(unique_parents)} unique parent nodes")
    

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    

    print("Generating original embeddings...")
    labels = [node['label'] for node in leaf_nodes]
    embeddings = model.encode(labels, convert_to_tensor=True)
    

    embeddings = embeddings.cpu().numpy()  # Convert to numpy array for PCA
    

    parent_labels = [node['parent'] for node in leaf_nodes]
    original_dbi = calculate_dbi(embeddings, parent_labels)
    print(f"\nDavies-Bouldin Index before PCA compression: {original_dbi:.4f}")
    

    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=32)
    compressed_embeddings = pca.fit_transform(embeddings)
    

    pca_model_path = os.path.join(output_dir, 'models', 'pca_model.pkl')
    with open(pca_model_path, 'wb') as f:
        pickle.dump(pca, f)
    

    compressed_dbi = calculate_dbi(compressed_embeddings, parent_labels)
    print(f"Davies-Bouldin Index after PCA compression: {compressed_dbi:.4f}")
    

    print("Visualizing compressed embeddings...")
    visualize_embeddings(
        compressed_embeddings,
        leaf_nodes,
        't-SNE Visualization of Compressed 64-dim Semantic Embeddings (PCA)',
        os.path.join(output_dir, 'plots', 'compressed_semantic_tree_visualization.png')
    )
    

    embeddings_path = os.path.join(output_dir, 'compressed_embeddings_pca.npy')
    np.save(embeddings_path, compressed_embeddings)
    
    print(f"\nPCA compression completed!")
    print(f"Compressed embeddings saved to: {embeddings_path}")
    print(f"Compressed embeddings visualization saved in {output_dir}/plots/")

if __name__ == "__main__":
    train_compressor() 