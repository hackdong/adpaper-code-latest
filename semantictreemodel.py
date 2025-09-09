import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import logging
import torch
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
from sentence_transformers import SentenceTransformer
import torchaudio.transforms as T
import numpy as np
import networkx as nx

# 数据集类
class AudioDataset(Dataset):
    """
    Custom Dataset class for audio event detection.
    Handles loading and preprocessing of audio files and their corresponding labels.
    """
    text_encoder = None
    
    def __init__(self, csv_path, semantic_tree_path, split='train', sample_rate=16000, duration=10, train_ratio=0.9, random_seed=42):
        """
        Initialize the dataset.
        Args:
            csv_path: Path to the CSV file containing metadata
            semantic_tree_path: Path to semantic tree file
            split: 'train' or 'val'
            sample_rate: Target sample rate for audio
            duration: Duration of audio clips in seconds
            train_ratio: Ratio of training data (default: 0.9)
            random_seed: Random seed for reproducibility
        """
        print(f"Initializing AudioDataset with csv_path: {csv_path}")
        
        # 读取所有数据
        all_data = pd.read_csv(csv_path)
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 随机打乱数据
        shuffled_indices = np.random.permutation(len(all_data))
        train_size = int(len(all_data) * train_ratio)
        
        # 划分训练集和验证集
        if split == 'train':
            selected_indices = shuffled_indices[:train_size]
            print(f"Using {len(selected_indices)} samples for training")
        else:  # val
            selected_indices = shuffled_indices[train_size:]
            print(f"Using {len(selected_indices)} samples for validation")
        
        # 选择对应的数据
        self.data = all_data.iloc[selected_indices].reset_index(drop=True)
        
        self.audio_dir = os.path.dirname(csv_path)
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        
        # 加载语义树
        print(f"Loading semantic tree from: {semantic_tree_path}")
        with open(semantic_tree_path, 'r') as f:
            self.semantic_tree = json.load(f)
        
        # 初始化mel频谱图转换
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        
        # 初始化文本编码器
        if AudioDataset.text_encoder is None:
            print("Initializing text encoder...")
            AudioDataset.text_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # 从数据中获取唯一的设备类型和事件类别
        self.device_types = list(self.data['device_type'].dropna().unique())
        self.overlay_categories = list(self.data['overlay_category'].dropna().unique())
        
        print(f"Found {len(self.device_types)} device types and {len(self.overlay_categories)} event categories")
        
        # 生成文本嵌入
        self.device_embeddings = {
            device: torch.tensor(self.text_encoder.encode(f"Device type: {device}"), 
                               dtype=torch.float32)
            for device in self.device_types
        }
        
        self.overlay_embeddings = {
            cat: torch.tensor(self.text_encoder.encode(f"Event type: {cat}"), 
                            dtype=torch.float32)
            for cat in self.overlay_categories
        }
        
        # 获取嵌入维度
        self.embedding_dim = next(iter(self.device_embeddings.values())).shape[0]
        print(f"Dataset initialization completed. Split: {split}")

    def get_semantic_distance(self, category1, category2):
        """
        计算语义树中两个类别之间的距离
        这个方法将在损失函数中使用
        """
        # 这里可以实现基于语义树的距离计算
        # 可以是最近公共祖先的深度、路径长度等
        return 1.0  # 临时返回默认值
    
    def __getitem__(self, idx):
        """获取一个样本"""
        try:
            row = self.data.iloc[idx]
            
            # 加载和处理音频
            audio_path = os.path.join(self.audio_dir, row['file_name'])
            waveform, sr = torchaudio.load(audio_path)
            
            # 重采样如果需要
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            
            # 处理音频长度
            if waveform.shape[1] < self.num_samples:
                waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.shape[1]))
            else:
                waveform = waveform[:, :self.num_samples]
            
            # 确保音频是单声道
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 转换为mel频谱
            mel_spec = self.mel_spectrogram(waveform)
            mel_spec = self.amplitude_to_db(mel_spec)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            
            # 准备设备嵌入
            device_embedding = self.device_embeddings[row['device_type']]
            
            # 准备事件掩码和嵌入
            event_mask = torch.zeros(mel_spec.size(-1))  # 使用频谱图的时间维度
            overlay_embedding = None
            
            if pd.notna(row['overlay_start_time']) and pd.notna(row['overlay_end_time']):
                start_frame = int(float(row['overlay_start_time']) * self.sample_rate / self.mel_spectrogram.hop_length)
                end_frame = int(float(row['overlay_end_time']) * self.sample_rate / self.mel_spectrogram.hop_length)
                start_frame = min(start_frame, mel_spec.size(-1)-1)
                end_frame = min(end_frame, mel_spec.size(-1))
                # print(f"start_frame:{start_frame} and end_frame:{end_frame}")
                event_mask[start_frame:end_frame] = 1.0
                
                
                if pd.notna(row['overlay_category']):
                    overlay_embedding = self.overlay_embeddings[row['overlay_category']]
            
            # 打印维度信息
            # print(f"Audio shape: {mel_spec.shape}, Event mask shape: {event_mask.shape}")
            
            return {
                'audio': mel_spec,
                'labels': {
                    'device_type': row['device_type'],
                    'device_embedding': device_embedding,
                    'is_normal': torch.tensor(1 if row['is_normal'] else 0, dtype=torch.float32),
                    'event_mask': event_mask,
                    'overlay_info': {
                        'category': row['overlay_category'] if pd.notna(row['overlay_category']) else None,
                        'embedding': overlay_embedding,
                        'start_time': row['overlay_start_time'] if pd.notna(row['overlay_start_time']) else None,
                        'end_time': row['overlay_end_time'] if pd.notna(row['overlay_end_time']) else None
                    }
                }
            }
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            return self._get_default_sample()


    def __len__(self):
        return len(self.data)

# 事件检测模型
class EventDetectionModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256, num_layers=2):
        super().__init__()
        
        # 特征提取（注意下采样率）
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 下采样2倍
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 下采样2倍
        )
        
        # 计算时间维度的下采样率
        self.time_downsample_ratio = 4  # 2 * 2 = 4
        
        # 时序编码
        self.temporal_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 事件检测头
        self.event_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 音频特征编码
        self.audio_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 384)
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入的音频特征 [batch_size, 1, freq, time]
        Returns:
            dict: {
                'event_mask_logits': 事件存在的概率序列 [batch_size, time],
                'audio_embeddings': 音频特征嵌入 [batch_size, 384]
            }
        """
        # 特征提取
        features = self.feature_extractor(x)  # [batch, channels, freq, time/4]
        batch_size = features.shape[0]
        
        # 调整维度以适应LSTM
        features = features.permute(0, 2, 3, 1)  # [batch, freq, time/4, channels]
        features = features.reshape(batch_size, -1, features.size(-1))  # [batch, time/4, features]
        
        # 时序编码
        hidden_states, _ = self.temporal_encoder(features)  # [batch, time/4, hidden_dim*2]
        
        # 事件检测
        event_mask_logits = self.event_head(hidden_states).squeeze(-1)  # [batch, time/4]
        
        # 上采样事件掩码预测以匹配目标大小
        event_mask_logits = F.interpolate(
            event_mask_logits.unsqueeze(1),  # [batch, 1, time/4]
            size=x.size(-1),  # 原始时间维度
            mode='linear',
            align_corners=False
        ).squeeze(1)  # [batch, time]
        
        # 音频特征编码
        pooled_features = hidden_states.mean(dim=1)  # [batch, hidden_dim*2]
        audio_embeddings = self.audio_encoder(pooled_features)  # [batch, 384]
        
        return {
            'event_mask_logits': event_mask_logits,
            'audio_embeddings': audio_embeddings
        }

# 机器分析模型
class MachineAnalysisModel(nn.Module):
    def __init__(self, num_device_types, pretrained_event_model):
        super().__init__()
        self.event_model = pretrained_event_model
        for param in self.event_model.parameters():
            param.requires_grad = False
            
        self.machine_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        
        # 修改为生成设备嵌入而不是分类logits
        self.device_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 384)  # 384是设备名称嵌入的维度
        )
        
        self.anomaly_detector = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        with torch.no_grad():
            event_outputs = self.event_model(x)
            event_mask = 1 - (torch.sigmoid(event_outputs['event_mask_logits']) > 0.5).float()
            event_mask = event_mask.unsqueeze(1)
        
        masked_input = x * event_mask.unsqueeze(2)
        
        features = self.machine_feature_extractor(masked_input)
        pooled_features = torch.mean(features, dim=(2, 3))
        
        device_embedding = self.device_encoder(pooled_features)  # 生成设备嵌入
        anomaly_logits = self.anomaly_detector(pooled_features)
        
        return {
            'device_embedding': device_embedding,  # 设备嵌入
            'is_normal': anomaly_logits
        }

class ContrastiveLoss(nn.Module):
    """统一的对比学习损失"""
    def __init__(self, semantic_tree, temperature=0.07):
        super().__init__()
        self.semantic_tree = semantic_tree
        self.temperature = temperature

    def calculate_contrastive_loss(self, audio_embeddings, text_embeddings, categories, category_to_idx):
        """
        计算音频特征和文本语义之间的对比损失
        Args:
            audio_embeddings: 音频特征向量 [batch_size, embed_dim]
            text_embeddings: 文本嵌入向量 [num_categories, embed_dim]
            categories: 当前批次的类别标签
            category_to_idx: 类别索引的映射
        """
        # L2归一化
        audio_embeddings = F.normalize(audio_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)

        # 计算相似度矩阵 [batch_size, num_categories]
        logits = torch.matmul(audio_embeddings, text_embeddings.T) / self.temperature
        
        batch_size = audio_embeddings.size(0)
        num_categories = text_embeddings.size(0)
        device = audio_embeddings.device

        # 计算语义权重矩阵
        semantic_weights = torch.zeros(batch_size, num_categories, device=device)
        for i in range(batch_size):
            for j in range(num_categories):
                semantic_weights[i, j] = self.get_semantic_similarity(
                    categories[i],
                    list(category_to_idx.keys())[j]
                )

        # 正样本对的损失
        pos_mask = torch.zeros_like(logits, device=device)
        for i, category in enumerate(categories):
            category_idx = category_to_idx[category]
            pos_mask[i, category_idx] = 1
        
        pos_logits = logits[pos_mask.bool()]

        # 负样本对的损失（考虑义相似度）
        neg_mask = 1 - pos_mask
        neg_weights = torch.exp(logits) * (1 - semantic_weights) * neg_mask

        # InfoNCE损失
        denominator = neg_weights.sum(dim=1) + torch.exp(pos_logits)
        loss = -pos_logits + torch.log(denominator)
        
        return loss.mean()

    def get_semantic_similarity(self, category1, category2):
        """计算语义树中两个类别的相似度"""
        def get_path_to_root(category):
            path = []
            current = category
            while current in self.semantic_tree['parents']:
                path.append(current)
                current = self.semantic_tree['parents'][current]
            path.append(current)
            return path

        path1 = get_path_to_root(category1)
        path2 = get_path_to_root(category2)

        common_length = 0
        for a, b in zip(reversed(path1), reversed(path2)):
            if a == b:
                common_length += 1
            else:
                break

        similarity = common_length / max(len(path1), len(path2))
        return similarity


class EventDetectionLoss(nn.Module):
    def __init__(self, semantic_tree=None, temperature=0.07, fp_weight=2.0, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.fp_weight = fp_weight
        self.event_mask_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.temperature = temperature
        self.semantic_tree = semantic_tree
        
        if semantic_tree:
            self.G = self._build_graph(semantic_tree)
            self._build_distance_cache()

    def calculate_metrics(self, predictions, targets):
        """
        计算事件检测的评估指标
        Args:
            predictions: 预测的事件掩码 logits [B, T]
            targets: 真实的事件掩码 [B, T]
        Returns:
            dict: 包含各项指标的字典
        """
        with torch.no_grad():
            # 将logits转换为二值预测
            pred_mask = (torch.sigmoid(predictions) > self.threshold)
            
            # 计算各类样本数量
            true_positives = (pred_mask & targets.bool()).sum().float()
            false_positives = (pred_mask & ~targets.bool()).sum().float()
            false_negatives = (~pred_mask & targets.bool()).sum().float()
            true_negatives = (~pred_mask & ~targets.bool()).sum().float()
            
            # 计算指标
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            # 计算每个样本的准确率
            accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + 
                                                          false_positives + false_negatives)
            
            return {
                'precision': precision.item(),
                'recall': recall.item(),
                'f1': f1.item(),
                'accuracy': accuracy.item(),
                'true_positives': true_positives.item(),
                'false_positives': false_positives.item(),
                'false_negatives': false_negatives.item(),
                'true_negatives': true_negatives.item()
            }

    def _calculate_mask_loss(self, predictions, targets):
        """计算带权重的掩码损失"""
        base_loss = self.event_mask_loss(predictions, targets)
        
        # 创建权重矩阵
        weights = torch.ones_like(targets)
        false_positive_mask = (torch.sigmoid(predictions) > self.threshold) & (targets == 0)
        weights[false_positive_mask] = self.fp_weight
        
        return (base_loss * weights).mean()

    def _build_graph(self, tree_data):
        """构建有向图表示的语义树，只关注label和children
        Args:
            tree_data: JSON格式的语义树数据
        Returns:
            nx.DiGraph: 语义树的图表示
        """
        G = nx.DiGraph()
        
        def add_node_and_children(node_data, parent_label=None):
            """递归添加节点和子节点"""
            current_label = node_data['label'].strip()  # 去除前后空格
            G.add_node(current_label)
            
            if parent_label:
                G.add_edge(parent_label, current_label)
            
            # 递归处理子节点
            for child in node_data.get('children', []):
                add_node_and_children(child, current_label)
        
        add_node_and_children(tree_data)
        return G
    
    def _build_distance_cache(self):
        """构建节点间最短路径的缓存"""
        self.distance_cache = {}
        root = self._find_root()
        
        # 获取所有叶子节点
        leaf_nodes = [node for node in self.G.nodes() if self.G.out_degree(node) == 0]
        
        # 计算所有叶子节点对之间的相似度
        for node1 in leaf_nodes:
            self.distance_cache[node1] = {}
            path1 = nx.shortest_path(self.G, root, node1)
            
            for node2 in leaf_nodes:
                path2 = nx.shortest_path(self.G, root, node2)
                
                # 计算共同路径长度
                common_length = 0
                for p1, p2 in zip(path1, path2):
                    if p1 == p2:
                        common_length += 1
                    else:
                        break
                
                # 计算相似度
                max_length = max(len(path1), len(path2))
                similarity = common_length / max_length
                
                self.distance_cache[node1][node2] = similarity
    
    def _find_root(self):
        """找到图的根节点"""
        root_nodes = [node for node in self.G.nodes() if self.G.in_degree(node) == 0]
        assert len(root_nodes) == 1, "语义树应该只有一个根节点"
        return root_nodes[0]
    
    def _calculate_semantic_similarity(self, event1, event2):
        """计算两个事件的语义相似度
        Args:
            event1: 第一个事件标签
            event2: 第二个事件标签
        Returns:
            float: 相似度分数 [0,1]
        """
        # 标准化输入
        event1 = event1.strip()
        event2 = event2.strip()
        
        # 如果事件标签不在缓存中，返回0
        if event1 not in self.distance_cache or event2 not in self.distance_cache:
            return 0.0
        
        return self.distance_cache[event1][event2]
    
    def calculate_rank_metrics(self, audio_embeddings, text_embeddings, event_names):
        """
        计算对比学习的排序准确度
        Args:
            audio_embeddings: 音频特征 [N, D]
            text_embeddings: 文本特征 [N, D]
            event_names: 事件名称列表，用于计算语义相似度加权
        Returns:
            dict: 包含排序准确度的指标
        """
        # 归一化特征
        audio_emb = F.normalize(audio_embeddings, p=2, dim=1)
        text_emb = F.normalize(text_embeddings, p=2, dim=1)
        
        # 计算相似度矩阵 [N, N]
        similarity = torch.matmul(audio_emb, text_emb.T)
        
        # 获取每个样本的预测排序
        _, predictions = similarity.sort(dim=1, descending=True)
        
        # 正确的匹配应该在对角线上
        targets = torch.arange(similarity.size(0), device=similarity.device)
        
        # 计算Top-k准确率
        top1_correct = (predictions[:, 0] == targets).float().mean()
        top5_correct = torch.any(predictions[:, :5] == targets.unsqueeze(1), dim=1).float().mean()
        
        # 计算平均排名（Mean Rank）
        ranks = torch.where(predictions == targets.unsqueeze(1))[1].float() + 1
        mean_rank = ranks.mean()
        
        # 计算MRR (Mean Reciprocal Rank)
        mrr = (1.0 / ranks).mean()
        
        # 如果提供了事件名称，计算语义加权的准确率
        semantic_acc = torch.tensor(0.0, device=similarity.device)
        if event_names:
            semantic_weights = torch.zeros_like(similarity)
            for i, name1 in enumerate(event_names):
                for j, name2 in enumerate(event_names):
                    semantic_weights[i, j] = self._calculate_semantic_similarity(name1, name2)
            
            # 计算语义加权的准确率
            semantic_acc = (semantic_weights * (predictions == targets.unsqueeze(1)).float()).sum(dim=1).mean()
        
        return {
            'top1_acc': top1_correct.item(),
            'top5_acc': top5_correct.item(),
            'mean_rank': mean_rank.item(),
            'mrr': mrr.item(),
            'semantic_acc': semantic_acc.item()
        }

    def forward(self, predictions, labels):
        """
        计算总损失和所有指标
        Args:
            predictions: {
                'event_mask_logits': [B, T],
                'audio_embeddings': [B, D]
            }
            labels: {
                'event_mask': [B, T],
                'overlay_info': {
                    'category': List[str],
                    'embedding': [B, D]
                }
            }
        """
        # 1. 事件掩码损失和指标
        mask_loss = self._calculate_mask_loss(
            predictions['event_mask_logits'],
            labels['event_mask']
        )
        
        mask_metrics = self.calculate_metrics(
            predictions['event_mask_logits'],
            labels['event_mask']
        )
        
        # 2. 对比学习损失和排序指标
        contrastive_loss = torch.tensor(0.0, device=predictions['audio_embeddings'].device)
        rank_metrics = {}
        
        # 获取有效样本
        valid_samples = []
        valid_embeddings = []
        valid_categories = []
        
        for i, (category, embedding, has_event) in enumerate(zip(
            labels['overlay_info']['category'],
            labels['overlay_info']['embedding'],
            labels['event_mask'].any(dim=1)
        )):
            if category is not None and embedding is not None and has_event:
                valid_samples.append(i)
                valid_embeddings.append(embedding)
                valid_categories.append(category)
        
        if valid_samples:
            valid_samples = torch.tensor(valid_samples, device=predictions['audio_embeddings'].device)
            audio_emb = predictions['audio_embeddings'][valid_samples]
            text_emb = torch.stack(valid_embeddings)
            
            # 计算对比损失
            similarity = torch.matmul(
                F.normalize(audio_emb, p=2, dim=1),
                F.normalize(text_emb, p=2, dim=1).T
            ) / self.temperature
            
            # 计算语义相似度矩阵
            semantic_sim = torch.zeros_like(similarity)
            for i, cat1 in enumerate(valid_categories):
                for j, cat2 in enumerate(valid_categories):
                    semantic_sim[i, j] = self._calculate_semantic_similarity(cat1, cat2)
            
            log_probs = F.log_softmax(similarity, dim=1)
            contrastive_loss = -(semantic_sim * log_probs).sum(dim=1).mean()
            
            # 计算排序指标
            rank_metrics = self.calculate_rank_metrics(
                audio_emb, text_emb, valid_categories
            )
        
        # 3. 返回所有损失和指标
        return {
            'total_loss': mask_loss + contrastive_loss,
            'mask_loss': mask_loss,
            'contrastive_loss': contrastive_loss,
            'mask_metrics': mask_metrics,
            'rank_metrics': rank_metrics
        }


class MachineAnalysisLoss(nn.Module):
    def __init__(self, semantic_tree):
        super().__init__()
        self.anomaly_loss = nn.BCEWithLogitsLoss()
        self.semantic_tree = semantic_tree

    def forward(self, predictions, targets):
        # 计算设备嵌入损失
        pred_embed = F.normalize(predictions['device_embedding'], p=2, dim=1)
        target_embed = F.normalize(targets['device_embedding'], p=2, dim=1)
        device_loss = 1 - F.cosine_similarity(pred_embed, target_embed).mean()
        
        # 调整维度以匹配
        is_normal_pred = predictions['is_normal'].squeeze(1)  # [batch_size]
        is_normal_target = targets['is_normal']  # [batch_size]
        
        # 计算异常检测损失
        anomaly_loss = self.anomaly_loss(is_normal_pred, is_normal_target)
        
        # 总损失
        total_loss = device_loss + anomaly_loss
        
        return total_loss

# 训练函数
def train_event_detection(model, train_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in labels.items()}
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels, labels['text_embeddings']['overlay'])
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

def train_machine_analysis(model, train_loader, criterion, optimizer, num_epochs=25, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in labels.items()}
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = AudioDataset('train_metadata.csv', 'semantic_treev3.json', split='train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 第一阶段：训练事件检测模型
    event_model = EventDetectionModel()
    event_criterion = EventDetectionLoss(train_dataset.semantic_tree)
    event_optimizer = torch.optim.Adam(event_model.parameters(), lr=0.001)
    train_event_detection(event_model, train_loader, event_criterion, event_optimizer, device=device)
    
    # 第二阶段：训练机器分析模型
    machine_model = MachineAnalysisModel(
        num_device_types=len(train_dataset.device_types),
        pretrained_event_model=event_model
    )
    machine_criterion = MachineAnalysisLoss(train_dataset.semantic_tree)
    machine_optimizer = torch.optim.Adam(machine_model.parameters(), lr=0.001)
    train_machine_analysis(machine_model, train_loader, machine_criterion, machine_optimizer, device=device)



if __name__ == "__main__":
    main()
