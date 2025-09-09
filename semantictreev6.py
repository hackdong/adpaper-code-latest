#使用自动k值选择的谱聚类构建语义树
import numpy as np
import requests
import json
from typing import List, Tuple, Dict
import networkx as nx
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TreeNode:
    def __init__(self, label: str, phrases: List[str] = None):
        self.label = label
        self.phrases = phrases or [label]
        self.children = []
        self.embedding = None
        
    def add_child(self, child_node: 'TreeNode'):
        self.children.append(child_node)
        
    def __repr__(self):
        return f"TreeNode(label='{self.label}', phrases={self.phrases})"

class ClusterInfo:
    def __init__(self, nodes: List[int], centroid: np.ndarray = None):
        self.node_indices = nodes  # 簇中节点的索引
        self.centroid = centroid   # 簇的中心向量

class SemanticTreeBuilder:
    def __init__(self, api_url: str = 'http://localhost:11434/api/chat'):
        self.api_url = api_url
        self.embeddings = {}  # 存储语的嵌入向量
        self.nodes = []  # Store TreeNode objects instead of clusters
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.similarity_matrix = None  # Add this line to store similarity matrix
        self.output_file = 'semantic_treev3.json'  # Add default output file path
        
    def get_semantic_label(self, nodes: List[TreeNode], excluded_labels: List[str] = None) -> str:
        """
        调用大模型获取语义标签
        Args:
            nodes: 需要合并的TreeNode列表
            excluded_labels: 不应该包含的标签列表
        """
        # 只使用子节点的label而不是所有phrases
        labels = [node.label for node in nodes]
        print(f"Getting semantic label for cluster of {len(labels)} labels...")
        print(f"Labels to include: {labels}")
        if excluded_labels:
            print(f"Labels to exclude: {excluded_labels}")

        prompt = (
            "You are an audio analysis expert. Generate 5 different category names for the given categories.\n"
            "Requirements for each category name:\n"
            "- Specific enough to distinguish from other categories\n"
            "- General enough to include all positive examples\n"
            "- Not so broad that it includes negative examples\n"
            "- Each name should be less than 5 words\n"
            "- Output format: one name per line, no numbers or bullet points\n"
            "- Different with the given children labels\n"
            f"Must include these categories: {', '.join(labels)}\n"
        )
        
        # if excluded_labels:
        #     prompt += f"Must NOT include these categories: {', '.join(excluded_labels)}"

        data = {
            "model": "llama3.1:8b",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate 5 appropriate category names in English, one per line."}
            ],
            "stream": False
        }
        
        response = requests.post(
            self.api_url, 
            data=json.dumps(data), 
            headers={'Content-Type': 'application/json'}
        )
        candidate_labels = json.loads(response.text)['message']["content"].strip().split('\n')
        print(f"Generated labels: {candidate_labels}")
        
        # 选择最佳标签
        return self._select_best_label(candidate_labels, nodes)

    def _select_best_label(self, candidate_labels: List[str], nodes: List[TreeNode]) -> str:
        """
        从候选标签中选择与子节点embedding平均值最接近的标签
        """
        # 计算子节点embeddings的平均值
        node_embeddings = np.array([node.embedding for node in nodes])
        centroid = np.mean(node_embeddings, axis=0)
        
        # 计算每个候选标签的embedding并找出最接近的
        best_similarity = -1
        best_label = candidate_labels[0]  # 默认使用第一个标签
        
        for label in candidate_labels:
            label_embedding = self._text_embedding(label)
            similarity = cosine_similarity([centroid], [label_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = label
        
        print(f"Selected best label: {best_label} (similarity: {best_similarity:.4f})")
        return best_label
    
    def build_tree(self, phrases: List[str]) -> TreeNode:
        """构建语义树"""
        print(f"Starting tree building with {len(phrases)} phrases...")
        # Initialize leaf nodes
        self.nodes = [TreeNode(phrase) for phrase in phrases]
        for node in self.nodes:
            node.embedding = self._text_embedding(node.label)
            
        while len(self.nodes) > 1:
            print(f"Current number of nodes: {len(self.nodes)}")
            self._update_similarity_matrix()
            print("similarity_matrix:",self.similarity_matrix)
            similarity_graph = self._build_similarity_graph()
            
            # Get connected components for merging
            connected_components = list(nx.connected_components(similarity_graph))
            if len(connected_components) == len(self.nodes):
                break
            print("number connected_components:",len(connected_components))
                
            # Process each component
            new_nodes = []
            processed_indices = set()
            
            for component in connected_components:
                if len(component) > 1:
                    # Merge nodes in component
                    merged_phrases = []
                    component_nodes = []
                    for idx in component:
                        merged_phrases.extend(self.nodes[idx].phrases)
                        component_nodes.append(self.nodes[idx])
                        processed_indices.add(idx)
                    
                    # Get labels from other nodes as exclusions
                    excluded_labels = [node.label for i, node in enumerate(self.nodes) 
                                    if i not in processed_indices]
                    
                    # Create new parent node
                    label = self.get_semantic_label(component_nodes, excluded_labels)
                    new_node = TreeNode(label, merged_phrases)
                    new_node.embedding = self._text_embedding(label)
                    
                    # Add children
                    for child_node in component_nodes:
                        new_node.add_child(child_node)
                    
                    new_nodes.append(new_node)
            
            # Add unprocessed nodes
            for i, node in enumerate(self.nodes):
                if i not in processed_indices:
                    new_nodes.append(node)
            
            self.nodes = new_nodes
            
        # Return the root node if there's only one node left, otherwise return the list of nodes
        return self.nodes[0] if len(self.nodes) == 1 else self.nodes
    
    def _calculate_cluster_similarity(self, node1: TreeNode, node2: TreeNode) -> float:
        """Calculate similarity between two nodes using their embeddings"""
        return cosine_similarity([node1.embedding], [node2.embedding])[0][0]
    
    def _update_similarity_matrix(self):
        """Update similarity matrix using node embeddings"""
        n = len(self.nodes)
        self.similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._calculate_cluster_similarity(self.nodes[i], self.nodes[j])
                self.similarity_matrix[i][j] = sim
                self.similarity_matrix[j][i] = sim
    
    def _build_similarity_graph(self) -> nx.Graph:
        """基于特征值斜率选择k值和k最近邻构建相似度图"""
        # 初始化最终的相似度图
        final_graph = nx.Graph()
        
        # 使用已有的相似度矩阵作为亲和矩阵
        affinity_matrix = self.similarity_matrix
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(affinity_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]  # 降序排列
        eigenvalues = eigenvalues[sorted_indices]
        
        # 计算斜率并找到最佳k值
        slopes = np.diff(eigenvalues)
        k = np.argmin(slopes) + 2  # +1 因为slopes长度比eigenvalues少1
        
        print(f"Selected k value: {k}")
        
        # 对每个节点找到k个最近邻
        n = len(self.nodes)
        for i in range(n):
            # 获取当前节点与所有其他节点的相似度
            similarities = affinity_matrix[i]
            # 找到k个最相似的节点（不包括自己）
            # 将自己的相似度设为最小值，这样不会选到自己
            similarities[i] = -np.inf
            nearest_neighbors = np.argsort(similarities)[-k:]
            
            # 如果两个节点互为k近邻，则添加边
            for j in nearest_neighbors:
                # 获取j的k个最近邻
                similarities_j = affinity_matrix[j].copy()
                similarities_j[j] = -np.inf
                nearest_neighbors_j = np.argsort(similarities_j)[-k:]
                
                # 如果i和j互为k近邻，则添加边
                if i in nearest_neighbors_j:
                    final_graph.add_edge(i, j)
        
        return final_graph
    
    
    def _text_embedding(self, text: str) -> np.ndarray:
        """获取文本的嵌入向量"""
        # 检查缓存中存在
        if text not in self.embeddings:
            # 使用预训练模型获取嵌入向量
            self.embeddings[text] = self.embedding_model.encode([text])[0]
        return self.embeddings[text]
    
    def save_tree(self, root: TreeNode, output_file: str = None) -> None:
        """保存语义树到JSON文件"""
        if output_file:
            self.output_file = output_file

        def node_to_dict(node: TreeNode) -> dict:
            return {
                'label': node.label,
                'phrases': node.phrases,
                'children': [node_to_dict(child) for child in node.children]
            }

        # Convert tree to serializable dictionary format
        tree_data = node_to_dict(root)

        # Save to JSON file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=2)
        print(f"Semantic tree saved to {self.output_file}")
    
    def _calculate_centroid(self, nodes: List[TreeNode]) -> np.ndarray:
        """计算一组节点的中心向量"""
        embeddings = np.array([node.embedding for node in nodes])
        return np.mean(embeddings, axis=0)


# 使用示例
if __name__ == "__main__":
    
    # 读取all_audio_classesv2.txt
    with open('all_audio_classesv2.txt', 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        # 跳过前两行
        all_classes = [line[1:] for line in lines[2:] if line.startswith('-')]
    print("number of all_classes:",len(all_classes))  

    # Create builder instance and build tree
    builder = SemanticTreeBuilder()
    tree = builder.build_tree(all_classes)
    
    # Save tree using the builder
    builder.save_tree(tree)
