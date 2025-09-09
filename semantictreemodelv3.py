import os
import json
import sys
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
from semantic_tree_embeddingv2 import compress_embeddings
import traceback

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# 数据集类
class AudioDataset(Dataset):
    """
    Custom Dataset class for audio event detection.
    Handles loading and preprocessing of audio files and their corresponding labels.
    """
    text_encoder = None
    category_embeddings = {}  # 类变量，用于存储类别嵌入
    initialized = False  # 添加初始化标志
    
    def __init__(self, metadata_path, semantic_tree_path, split='train', train_ratio=0.8, random_seed=42):
        """
        初始化数据集
        """
        # 设置音频参数
        self.sample_rate = 16000
        self.n_fft = 400
        self.hop_length = 160
        self.n_mels = 64
        self.f_min = 50
        self.f_max = 8000
        self.win_length = 400
        
        # 初始化梅尔频谱转换器
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            win_length=self.win_length
        )
        
        # 初始化文本编码器
        if AudioDataset.text_encoder is None:
            print("Initializing text encoder...")
            AudioDataset.text_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            if AudioDataset.text_encoder is None:
                print("Failed to initialize text encoder")
                sys.exit(1)
        
        # 读取元数据
        self.metadata = pd.read_csv(metadata_path)
        self.metadata_path = metadata_path
        
        # 数据清理：处理NaN值
        self.metadata['overlay_category'] = self.metadata['overlay_category'].fillna('NOC')
        self.metadata['device_type'] = self.metadata['device_type'].fillna('unknown')
        self.metadata['is_normal'] = self.metadata['is_normal'].fillna(True)
        
        # 数据集划分
        if split in ['train', 'val']:
            np.random.seed(random_seed)
            indices = np.random.permutation(len(self.metadata))
            train_size = int(len(indices) * train_ratio)
            
            if split == 'train':
                self.metadata = self.metadata.iloc[indices[:train_size]]
            else:  # val
                self.metadata = self.metadata.iloc[indices[train_size:]]
        
        # 加载语义树
        with open(semantic_tree_path, 'r') as f:
            self.semantic_tree = json.load(f)
            
        # 构建设备类型映射
        self.device_types = sorted(list(set(self.metadata['device_type'].astype(str))))
        self.device_type_to_idx = {device: idx for idx, device in enumerate(self.device_types)}
        
        # 构建事件类别映射
        self.event_categories = sorted(list(set(self.metadata['overlay_category'].astype(str))))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.event_categories)}
        
        # 预先生成所有类别的嵌入

        print("Generating category embeddings...")
        self.category_embeddings = {}  # 显式初始化
        for category in self.event_categories:
            try:
                compressed_embedding = compress_embeddings(
                    AudioDataset.text_encoder.encode(f" {category}").reshape(1, -1)
                )[0]
                self.category_embeddings[category] = torch.tensor(
                    compressed_embedding,
                    dtype=torch.float32
                )
                
            except Exception as e:
                print(f"Error generating embedding for category {category}: {str(e)}")
                self.category_embeddings[category] = torch.zeros(32)
            
        print("Category embeddings generated.")
        AudioDataset.initialized = True  # 设置初始化标志
        
        print("Generating device embeddings...")
        self.device_embeddings = {}  # 显式初始化
        for device in self.device_types:
            try:
                compressed_embedding = compress_embeddings(
                    AudioDataset.text_encoder.encode(f" {device}").reshape(1, -1)
                )[0]
                self.device_embeddings[device] = torch.tensor(
                    compressed_embedding,
                    dtype=torch.float32
                )
                
            except Exception as e:
                print(f"Error generating embedding for device {device}: {str(e)}")
                self.device_embeddings[device] = torch.zeros(32)
            
        print("Device embeddings generated.")
        AudioDataset.initialized = True  # 设置初始化标志

        # 打印数据集信息
        print(f"\nDataset Info ({split}):")
        print(f"Total samples: {len(self.metadata)}")
        print(f"Device types: {self.device_types}")
        print(f"Event categories: {self.event_categories}")
        print(f"Number of device types: {len(self.device_types)}")
        print(f"Number of event categories: {len(self.event_categories)}\n")

    def load_audio(self, file_path):
        """加载并预处理音频文件"""
        try:
            # 加载音频
            waveform, sr = torchaudio.load(file_path)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 重采样
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 计算梅尔频谱图
            mel_spec = self.mel_spectrogram(waveform)
            
            # 转换为分贝单位
            mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            
            # 标准化
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            
            return mel_spec
            
        except Exception as e:
            print(f"Error loading audio file {file_path}: {str(e)}")
            return torch.zeros(1, self.n_mels, 1000)  # 返回适当大小的零张量

    def __getitem__(self, idx):
        try:
            row = self.metadata.iloc[idx]
            
            # 构建完整的文件路径
            file_path = os.path.join(os.path.dirname(self.metadata_path), row['file_name'])
            
            # 确保类别是规范化的字符串
            device_type = str(row['device_type']).strip()
            event_category = str(row['overlay_category']).strip()
            
            # 验证类别是否存在
            if event_category not in self.category_embeddings:
                print(f"Missing category: '{event_category}'")
                print(f"Available categories: {list(self.category_embeddings.keys())}")
                # 动态添加缺失的类别
                try:
                    compressed_embedding = compress_embeddings(
                        AudioDataset.text_encoder.encode(f" {event_category}").reshape(1, -1)
                    )[0]
                    self.category_embeddings[event_category] = torch.tensor(
                        compressed_embedding,
                        dtype=torch.float32
                    )
                except Exception as e:
                    print(f"Error generating embedding for category {event_category}: {str(e)}")
                    self.category_embeddings[event_category] = torch.zeros(32)
            
            # 加载音频
            audio_tensor = self.load_audio(file_path)
            
            # 准备标签
            labels = {
                'device_type_idx': torch.tensor(
                    self.device_type_to_idx[device_type], 
                    dtype=torch.long
                ),
                'device_embedding': self.device_embeddings[device_type].clone(),
                'is_normal': torch.tensor(
                    1 if row['is_normal'] else 0, 
                    dtype=torch.float
                ),
                'event_mask': self.create_event_mask(row).clone().detach(), 
 
                'overlay_info': {
                    'category': event_category,
                    'category_idx': torch.tensor(
                        self.category_to_idx[event_category], 
                        dtype=torch.long
                    ),
                    'embedding': self.category_embeddings[event_category].clone()  # 添加 clone()
                }
            }
            
            return {
                'audio': audio_tensor,
                'labels': labels,
                'file_path': file_path
            }
            
        except Exception as e:
            print(f"Error loading item {idx}:")
            print(f"Row data: {row.to_dict()}")
            print(f"Error: {str(e)}")
            raise

    def create_event_mask(self, row):
        """创建事件掩码"""
        # 计算时间轴上的点数
        n_frames = int(self.sample_rate * 10 / self.hop_length) + 1  # 10秒音频
        mask = torch.zeros(n_frames)
        
        if pd.notna(row['overlay_start_time']) and pd.notna(row['overlay_end_time']):
            start_frame = int(row['overlay_start_time'] * self.sample_rate / self.hop_length)
            end_frame = int(row['overlay_end_time'] * self.sample_rate / self.hop_length)
            mask[start_frame:end_frame] = 1
            
        return mask

    def __len__(self):
        return len(self.metadata)

# 事件检测模型
class EventDetectionModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256, num_layers=2, num_event_categories=0):
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
            nn.Linear(hidden_dim, 32)  # 改为32维输出
        )
        
        # 添加事件分类头
        self.event_classifier = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_event_categories)
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入的音频特征 [batch_size, 1, freq, time]
        Returns:
            dict: {
                'event_mask_logits': 事件存在的概率序列 [batch_size, time],
                'audio_embeddings': 音频特征嵌入 [batch_size, 32]
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
        audio_embeddings = self.audio_encoder(pooled_features)  # [batch, 32]
        
        # 添加事件分类
        event_logits = self.event_classifier(audio_embeddings)
        
        return {
            'event_mask_logits': event_mask_logits,
            'audio_embeddings': audio_embeddings,
            'event_logits': event_logits
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
        
        # 修改设备编码器输出维度为32
        self.device_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32)  # 改为32维输出
        )
        
        self.anomaly_detector = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
        # 添加设备分类头
        self.device_classifier = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_device_types)
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
        
        # 添加设备分类
        device_logits = self.device_classifier(device_embedding)
        
        return {
            'device_embedding': device_embedding,  # 设备嵌入
            'is_normal': anomaly_logits,
            'device_logits': device_logits
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
    def __init__(self, semantic_tree, threshold=0.5, fp_weight=2.0, embedding_weight=0.5, classification_weight=1.0):
        """
        初始化事件检测损失函数
        Args:
            threshold: 事件检测的阈值
            fp_weight: 假阳性样本的权重
            embedding_weight: 嵌入损失的权重
            classification_weight: 分类损失的权重
        """
        super().__init__()
        self.threshold = threshold
        self.fp_weight = fp_weight
        self.embedding_weight = embedding_weight
        self.classification_weight = classification_weight
        self.event_mask_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, labels):
        try:
            # 获取设备信息
            device = predictions['event_logits'].device
            
            # 1. 掩码损失计算
            mask_loss = self._calculate_mask_loss(
                predictions['event_mask_logits'],
                labels['event_mask'].to(device)
            )
            
            # 2. 嵌入损失计算
            embedding_loss = torch.tensor(0.0, device=device)
            if isinstance(labels['overlay_info']['embedding'], torch.Tensor):
                embedding_loss = F.mse_loss(
                    predictions['audio_embeddings'],
                    labels['overlay_info']['embedding'].to(device)
                )
            
            # 3. 分类损失计算
            classification_loss = torch.tensor(0.0, device=device)
            if 'event_logits' in predictions and 'category_idx' in labels['overlay_info']:
                if labels['overlay_info']['category_idx'] is not None:
                    # Assuming event_logits are logits that need to be passed through softmax to get probabilities
                    predicted_probabilities = F.softmax(predictions['event_logits'], dim=1)
                    # Convert the true category index to a one-hot vector for comparison with probabilities
                    true_category_one_hot = F.one_hot(labels['overlay_info']['category_idx'].to(device), num_classes=predictions['event_logits'].size(1))
                    # Compute the classification loss between the predicted probabilities and the true category one-hot vector
                    classification_loss = self.classification_loss(
                        predicted_probabilities,
                        true_category_one_hot.float()
                    )
            
            # 4. 计算总损失
            total_loss = (mask_loss + 
                         self.embedding_weight * embedding_loss + 
                         self.classification_weight * classification_loss)
            
            # 5. 计算评估指标
            with torch.no_grad():
                mask_metrics = self.calculate_metrics(
                    predictions['event_mask_logits'].detach(),
                    labels['event_mask'].to(device)
                )
                
                # 添加分类准确率
                if 'event_logits' in predictions and 'category_idx' in labels['overlay_info']:
                    if labels['overlay_info']['category_idx'] is not None:
                        pred_classes = torch.argmax(predictions['event_logits'], dim=1)
                        correct = (pred_classes == labels['overlay_info']['category_idx'].to(device)).float().mean()
                        mask_metrics['classification_accuracy'] = correct.item()
            
            return {
                'total_loss': total_loss,
                'mask_loss': mask_loss.detach(),
                'embedding_loss': embedding_loss.detach(),
                'classification_loss': classification_loss.detach(),
                'mask_metrics': mask_metrics
            }
            
        except Exception as e:
            print(f"Error in loss calculation: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            # 返回默认损失值
            device = predictions['event_mask_logits'].device
            return {
                'total_loss': torch.tensor(0.0, requires_grad=True, device=device),
                'mask_loss': torch.tensor(0.0, device=device),
                'embedding_loss': torch.tensor(0.0, device=device),
                'classification_loss': torch.tensor(0.0, device=device),
                'mask_metrics': {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'accuracy': 0.0
                }
            }

    def calculate_metrics(self, predictions, targets):
        """计算评估指标"""
        try:
            with torch.no_grad():
                # 将预测转换为二值
                pred_probs = torch.sigmoid(predictions)
                pred_binary = (pred_probs > 0.5).float()
                
                # 计算混淆矩阵元素
                true_positives = (pred_binary * targets).sum()
                false_positives = (pred_binary * (1 - targets)).sum()
                false_negatives = ((1 - pred_binary) * targets).sum()
                true_negatives = ((1 - pred_binary) * (1 - targets)).sum()
                
                # 计算指标
                precision = true_positives / (true_positives + false_positives + 1e-8)
                recall = true_positives / (true_positives + false_negatives + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives + 1e-8)
                
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
                
        except Exception as e:
            print(f"Error in metric calculation: {str(e)}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': 0.0,
                'true_positives': 0.0,
                'false_positives': 0.0,
                'false_negatives': 0.0,
                'true_negatives': 0.0
            }

    def _calculate_mask_loss(self, predictions, targets):
        """计算掩码损失"""
        try:
            # 确保输入在同一设备上并且形状匹配
            predictions = predictions.to(targets.device)
            if predictions.shape != targets.shape:
                raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
            
            # 打印张量信息
           # print(f"Mask tensors - pred: {predictions.shape} ({predictions.dtype}), "
           #       f"target: {targets.shape} ({targets.dtype})")#
            
            # 使用 BCE 损失
            loss = F.binary_cross_entropy_with_logits(
                predictions.float(),
                targets.float(),
                reduction='none'
            )
            
            # 计算加权损失
            weights = torch.ones_like(targets)
            with torch.no_grad():
                pred_probs = torch.sigmoid(predictions)
                # 直使用浮点数进行比较，PyTorch会自动处理广播
                false_positives = (pred_probs > 0.5) & (targets == 0)
                weights[false_positives] = self.fp_weight
            
            # 返回平均损失
            return (loss * weights).mean()
            
        except Exception as e:
            print(f"Error in mask loss calculation: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    def _build_graph(self, tree_data):
        """构建有向图表示的语义树，只关注label和children
        Args:
            tree_data: JSON格式的语义树数据
        Returns:
            nx.DiGraph: 语义树的图表示
        """
        G = nx.DiGraph()
        
        def add_node_and_children(node_data, parent_label=None):
            """递归添加节点子节点"""
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
    
    def calculate_rank_metrics(self, audio_embeddings, text_embeddings ):
        """
        计算对比学习的排序准确度
        Args:
            audio_embeddings: 音频特征 [N, D]
            text_embeddings: 文本特征 [N, D]
        Returns:
            dict: 包含排序准确度的指标
        """

        print("caculate rank metrics")
        # 归一化特征
        audio_emb = F.normalize(audio_embeddings, p=2, dim=1)
        text_emb = F.normalize(text_embeddings, p=2, dim=1)
        
        # 计算相似度矩阵 [N, N]
        similarity = torch.matmul(audio_emb, text_emb.T)
        
        # 取每个样本的预测排序
        _, predictions = similarity.sort(dim=1, descending=True)
        
        # 正确的匹配应该在对角线上
        targets = torch.arange(similarity.size(0), device=similarity.device)
        
        # 计算Top-k准确率
        top1_correct = (predictions[:, 0] == targets).float().mean()
        top3_correct = torch.any(predictions[:, :3] == targets.unsqueeze(1), dim=1).float().mean()
        
        # 计算平均排名（Mean Rank）
        ranks = torch.where(predictions == targets.unsqueeze(1))[1].float() + 1
        mean_rank = ranks.mean()
        
        # 计算MRR (Mean Reciprocal Rank)
        mrr = (1.0 / ranks).mean()
        

        return {
            'top1_acc': top1_correct.item(),
            'top3_acc': top3_correct.item(),
            'mean_rank': mean_rank.item(),
            'mrr': mrr.item(),
        }

class MachineAnalysisLoss(nn.Module):
    def __init__(self, semantic_tree, device_weight=1.0, anomaly_weight=1.0, classification_weight=1.0):
        super().__init__()
        self.semantic_tree = semantic_tree
        self.device_weight = device_weight
        self.anomaly_weight = anomaly_weight
        self.classification_weight = classification_weight
        self.anomaly_loss = nn.BCEWithLogitsLoss()
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        # 确保所有张量都在同一个设备上
        device = predictions['device_embedding'].device
        
        # 1. 设备嵌入损失
        device_loss = 1 - F.cosine_similarity(
            F.normalize(predictions['device_embedding'], p=2, dim=1),
            F.normalize(targets['device_embedding'].to(device), p=2, dim=1)
        ).mean()
        
        # 2. 异常检测损失
        anomaly_loss = self.anomaly_loss(
            predictions['is_normal'].squeeze(1),
            targets['is_normal'].to(device)
        )
        
        # 3. 设备分类损失
        classification_loss = self.classification_loss(
            predictions['device_logits'],
            targets['device_type_idx'].to(device)
        )
        
        # 4. 总损失
        total_loss = (self.device_weight * device_loss + 
                     self.anomaly_weight * anomaly_loss +
                     self.classification_weight * classification_loss)
        
        # 5. 计算分类准确率
        with torch.no_grad():
            pred_classes = torch.argmax(predictions['device_logits'], dim=1)
            classification_acc = (pred_classes == targets['device_type_idx'].to(device)).float().mean()
        
        return {
            'total_loss': total_loss,
            'device_loss': device_loss.detach(),
            'anomaly_loss': anomaly_loss.detach(),
            'classification_loss': classification_loss.detach(),
            'classification_accuracy': classification_acc.item()
        }

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


def custom_collate_fn(batch):
    """自定义的collate函数，处理None值和不同长度的数据"""
    if len(batch) == 0:
        return {}
    
    # 过滤掉None值
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
        
    elem = batch[0]
    if isinstance(elem, dict):
        return {
            key: custom_collate_fn([d[key] for d in batch]) 
            for key in elem
        }
    elif isinstance(elem, torch.Tensor):
        try:
            return torch.stack(batch, 0)
        except:
            return batch
    elif isinstance(elem, (int, float)):
        try:
            return torch.tensor(batch)
        except:
            return batch
    elif isinstance(elem, (str, type(None))):
        return batch
    elif isinstance(elem, list):
        return [custom_collate_fn(samples) for samples in zip(*batch)]
    else:
        return batch




# 主函数
def main():
    print("use trainv3.py")
if __name__ == "__main__":
    main()
