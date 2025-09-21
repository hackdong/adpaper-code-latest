
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, average_precision_score, mean_squared_error
import pandas as pd
import librosa
import os
import csv
from datetime import datetime
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from tqdm import tqdm
import time

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        return x

class PANN(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        super(PANN, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        
    def forward(self, input):
        """Input: (batch_size, time_steps)"""
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        
        return embedding


class SegmentProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SegmentProcessor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, num_segments, input_dim)
        batch_size, num_segments, input_dim = x.size()
        
        # Reshape to (batch_size * num_segments, 1, input_dim)
        x = x.view(batch_size, num_segments, input_dim)
        
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x.squeeze(1)  # Remove the sequence dimension
        x = x.view(batch_size, num_segments, -1)
        return x.mean(dim=1)  # Average over segments

class AudioSegmenter:
    def __init__(self, sample_rate, min_segment_size, max_segment_size):
        self.sample_rate = sample_rate
        self.min_segment_size = min_segment_size
        self.max_segment_size = max_segment_size

    def __call__(self, audio, segment_size=None):
        if segment_size is None:
            segment_size = self.max_segment_size
        segment_size = max(min(segment_size, self.max_segment_size), self.min_segment_size)
        
        audio_length = audio.size(1)
        num_segments = audio_length // segment_size
        
        if num_segments == 0:
            padded_audio = F.pad(audio, (0, segment_size - audio_length))
            return padded_audio.unsqueeze(1)
        
        segments = audio[:, :num_segments*segment_size].view(-1, num_segments, segment_size)
        
        if audio_length % segment_size != 0:
            last_segment = audio[:, num_segments*segment_size:]
            last_segment = F.pad(last_segment, (0, segment_size - last_segment.size(1)))
            segments = torch.cat([segments, last_segment.unsqueeze(1)], dim=1)
           
        return segments
class AudioTextAlignmentModule(nn.Module):
    def __init__(self, audio_dim, text_dim, output_dim):
        super(AudioTextAlignmentModule, self).__init__()
        self.audio_proj = nn.Linear(audio_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)
        self.attention = nn.MultiheadAttention(output_dim, 8, batch_first=True)

    def forward(self, audio_features, text_features=None):
        audio_proj = self.audio_proj(audio_features)
        
        if text_features is not None:
            text_proj = self.text_proj(text_features)
            aligned_features, _ = self.attention(audio_proj, text_proj, text_proj)
        else:
            aligned_features, _ = self.attention(audio_proj, audio_proj, audio_proj)
        
        return aligned_features


class FlexibleAnomalyAudioClassifier(nn.Module):
    def __init__(self, pann, sample_rate, min_segment_size, max_segment_size, device, normal_categories,use_cls_token=True):
        super(FlexibleAnomalyAudioClassifier, self).__init__()
        self.device = device
        self.use_cls_token = use_cls_token
        self.segmenter = AudioSegmenter(sample_rate, min_segment_size, max_segment_size)
        self.pann = pann
        
        self.normal_categories = normal_categories

        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
        self.text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        text_feature_dim = self.text_encoder.config.hidden_size
        if not use_cls_token:
            text_feature_dim *= self.text_tokenizer.model_max_length
        
        self.alignment = AudioTextAlignmentModule(2048, text_feature_dim, 256).to(device)
        self.segment_processor = SegmentProcessor(256, 256, 128).to(device)
        
        self.anomaly_classifier = nn.Linear(128, 2).to(device)
        self.anomaly_regressor = nn.Linear(128, 1).to(device)
        self.audio_projector = nn.Linear(128, text_feature_dim).to(device)
        
        self.category_classifier = nn.Linear(128, len(normal_categories))
        
        # Encode normal categories
        self.category_embeddings = self.encode_categories(normal_categories)

    def forward(self, audio_input, text_input, segment_size=None):
        audio_input = audio_input.to(self.device, dtype=torch.float32)

        # Encode text input
        with torch.no_grad():  # Ensure no gradients are computed for text encoding
            encoded_text = self.text_tokenizer(text_input, return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_features = self.text_encoder(**encoded_text).last_hidden_state

        audio_segments = self.segmenter(audio_input, segment_size)
        batch_size, num_segments, segment_length = audio_segments.size()

        # Process each segment individually
        audio_features = []
        for i in range(num_segments):
            segment_feature = self.pann(audio_segments[:, i, :])
            audio_features.append(segment_feature)
        
        # Stack features along the time dimension (num_segments)
        audio_features = torch.stack(audio_features, dim=1)

        aligned_features = self.alignment(audio_features, text_features)
        segment_features = self.segment_processor(aligned_features)
        
        # Outputs
        anomaly_output = self.anomaly_classifier(segment_features)
        anomaly_severity = self.anomaly_regressor(segment_features).squeeze(-1)
        category_output = self.category_classifier(segment_features)
        
        # Project audio features to text embedding space
        audio_embedding = self.audio_projector(segment_features)
        
        return anomaly_output, anomaly_severity, category_output, audio_embedding

    def encode_categories(self, categories):
        # Encode text categories into embeddings
        inputs = self.text_tokenizer(categories, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding

    def classify_normal_audio(self, audio_embedding, category_embeddings):
        # Compute cosine similarity between audio embedding and category embeddings
        similarities = F.cosine_similarity(audio_embedding.unsqueeze(1), category_embeddings.unsqueeze(0), dim=2)
        return similarities

    def get_text_similarity(self, text_input, category_texts):
        with torch.no_grad():
            text_encoding = self.text_tokenizer(text_input, return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_features = self.text_encoder(**text_encoding).last_hidden_state[:, 0, :]  # 使用 [CLS] token
            
            category_encodings = self.text_tokenizer(category_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            category_features = self.text_encoder(**category_encodings).last_hidden_state[:, 0, :]
            
            similarities = F.cosine_similarity(text_features.unsqueeze(1), category_features.unsqueeze(0), dim=2)
        
        return similarities  # 返回二维张量，每行对应一个输入文本与所有类别的相似度

# 新增：合成数据集类
class SyntheticAudioDataset(Dataset):
    def __init__(self, metadata_file, audio_dir):
        self.metadata = pd.read_csv(metadata_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['file_name'])
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        if len(audio) < 160000:
            audio = np.pad(audio, (0, 160000 - len(audio)))
        else:
            audio = audio[:160000]

        label = 0 if row['is_normal'] else 1
        severity = row['overlay_ratio'] if not row['is_normal'] else 0.0

        text = f"This is a {row['device_type']} sound from product {row['product_id']}."
        if row['overlay_category']:
            text += f" It contains {row['overlay_category']} sound."

        true_category = f"{row['device_type']} and {row['overlay_category']}"

        # Convert to float32
        return torch.FloatTensor(audio).float(), text, torch.tensor(label).long(), torch.tensor(severity).float(), true_category

# 修改：训练函数
def calculate_metrics(labels, preds, severity_true, severity_pred):
    accuracy = accuracy_score(labels, np.round(preds))
    
    # Handle cases where only one class is present
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            auc = roc_auc_score(labels, preds)
            pauc = roc_auc_score(labels, preds, max_fpr=0.1)
    except ValueError:
        auc = np.nan
        pauc = np.nan
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, np.round(preds), average='binary', zero_division=0)
    mse = mean_squared_error(severity_true, severity_pred)
    return accuracy, auc, pauc, precision, recall, f1, mse

def calculate_severity_accuracy(labels, preds, severity):
    severity_levels = {
        'low': (0, 0.33),
        'medium': (0.33, 0.66),
        'high': (0.66, 1.01)
    }
    severity_accuracy = {}
    for level, (min_val, max_val) in severity_levels.items():
        mask = (severity >= min_val) & (severity < max_val)
        if np.sum(mask) > 0:
            severity_accuracy[level] = accuracy_score(labels[mask], np.round(preds[mask]))
    return severity_accuracy

def calculate_normal_category_accuracy(true_categories, pred_similarities, normal_categories):
    correct = 0
    total = 0
    for true_cat, similarities in zip(true_categories, pred_similarities):
        if true_cat is not None:  
            total += 1
            top3 = similarities.argsort()[-3:][::-1]
            if any(normal_categories[i] in true_cat for i in top3):
                correct += 1
    return correct / total if total > 0 else 0

def calculate_severity_level_accuracy(true_severity, pred_severity):
    true_levels = pd.cut(true_severity, bins=[0, 0.33, 0.66, 1], labels=['low', 'medium', 'high'])
    pred_levels = pd.cut(pred_severity, bins=[0, 0.33, 0.66, 1], labels=['low', 'medium', 'high'])
    return accuracy_score(true_levels, pred_levels)

def train(model, train_loader, optimizer, device, epoch, num_epochs, normal_categories):
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_severity_true = []
    all_severity_pred = []
    all_category_true = []
    all_category_similarities = []

    start_time = time.time()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in progress_bar:
        audio, text, labels, severity, true_categories = batch
        audio, labels, severity = audio.to(device).float(), labels.to(device).long(), severity.to(device).float()

        optimizer.zero_grad()
        anomaly_output, anomaly_severity, category_output, audio_embedding = model(audio, text)


        class_weights = torch.tensor([1.0, 5.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        anomaly_loss = criterion(anomaly_output, labels)
        

        severity_loss = F.mse_loss(anomaly_severity, severity)
        

        text_similarities = model.get_text_similarity(text, normal_categories)
        category_loss = -torch.log(text_similarities.max(dim=1)[0]).mean()  # 使用最大相似度的负对数作为损失

        loss = anomaly_loss + severity_loss + category_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(anomaly_output.softmax(dim=1)[:, 1].detach().cpu().numpy())
        all_severity_true.extend(severity.cpu().numpy())
        all_severity_pred.extend(anomaly_severity.detach().cpu().numpy())
        all_category_true.extend(true_categories)
        all_category_similarities.extend(text_similarities.detach().cpu().numpy())

        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    all_preds = np.array(all_preds)
    print(f"Predictions distribution: min={all_preds.min():.4f}, max={all_preds.max():.4f}, mean={all_preds.mean():.4f}")
    print(f"Positive predictions: {(all_preds > 0.5).sum()} out of {len(all_preds)}")

    metrics = calculate_metrics(all_labels, all_preds, all_severity_true, all_severity_pred)
    severity_accuracy = calculate_severity_accuracy(np.array(all_labels), np.array(all_preds), np.array(all_severity_true))
    

    normal_mask = np.array(all_labels) == 0
    if normal_mask.any():
        category_accuracy = calculate_normal_category_accuracy(
            np.array(all_category_true)[normal_mask],
            np.array(all_category_similarities)[normal_mask],
            normal_categories
        )
    else:
        category_accuracy = np.nan

    end_time = time.time()
    epoch_time = end_time - start_time

    return (total_loss / len(train_loader), *metrics, severity_accuracy, category_accuracy, epoch_time)


def evaluate(model, test_loader, device, normal_categories):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_severity_true = []
    all_severity_pred = []
    all_category_true = []
    all_category_similarities = []

    progress_bar = tqdm(test_loader, desc="Evaluating")

    with torch.no_grad():
        for batch in progress_bar:
            audio, text, labels, severity, true_categories = batch
            audio, labels, severity = audio.to(device), labels.to(device), severity.to(device)

            anomaly_output, anomaly_severity, category_output, audio_embedding = model(audio, text)


            text_similarities = model.get_text_similarity(text, normal_categories)

            anomaly_loss = F.cross_entropy(anomaly_output, labels)
            severity_loss = F.mse_loss(anomaly_severity, severity)
            loss = anomaly_loss + severity_loss

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(anomaly_output.softmax(dim=1)[:, 1].cpu().numpy())
            all_severity_true.extend(severity.cpu().numpy())
            all_severity_pred.extend(anomaly_severity.cpu().numpy())
            all_category_true.extend(true_categories)
            all_category_similarities.extend(text_similarities.cpu().numpy())

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_severity_true = np.array(all_severity_true)
    all_severity_pred = np.array(all_severity_pred)

    total_metrics = calculate_metrics(all_labels, all_preds, all_severity_true, all_severity_pred)
    
    normal_mask = all_labels == 0
    anomaly_mask = all_labels == 1
    
    normal_metrics = calculate_metrics(all_labels[normal_mask], all_preds[normal_mask], 
                                       all_severity_true[normal_mask], all_severity_pred[normal_mask]) if np.any(normal_mask) else (np.nan,) * 7
    anomaly_metrics = calculate_metrics(all_labels[anomaly_mask], all_preds[anomaly_mask], 
                                        all_severity_true[anomaly_mask], all_severity_pred[anomaly_mask]) if np.any(anomaly_mask) else (np.nan,) * 7

    severity_accuracy = calculate_severity_accuracy(all_labels, all_preds, all_severity_true)


    if normal_mask.any():
        category_accuracy = calculate_normal_category_accuracy(
            np.array(all_category_true)[normal_mask],
            np.array(all_category_similarities)[normal_mask],
            normal_categories
        )
    else:
        category_accuracy = np.nan


    severity_level_accuracy = calculate_severity_level_accuracy(
        all_severity_true[anomaly_mask],
        all_severity_pred[anomaly_mask]
    ) if np.any(anomaly_mask) else np.nan

    return {
        'total_loss': total_loss / len(test_loader),
        'total_metrics': total_metrics,
        'normal_metrics': normal_metrics,
        'anomaly_metrics': anomaly_metrics,
        'severity_accuracy': severity_accuracy,
        'category_accuracy': category_accuracy,
        'severity_level_accuracy': severity_level_accuracy
    }

def infer(model, audio, text, normal_categories, device):
    model.eval()
    with torch.no_grad():
        anomaly_output, anomaly_severity, audio_embedding = model(audio, text)
        
        is_anomaly = anomaly_output.argmax(dim=1).item()
        severity = anomaly_severity.item() * 100  # Convert to percentage

        if is_anomaly:
            severity_category = "{}%".format(round(severity / 20) * 20)
            return f"Anomaly detected. Severity: {severity_category}"
        else:
            # Encode normal categories
            category_embeddings = model.encode_categories(normal_categories)
            
            # Classify normal audio
            similarities = model.classify_normal_audio(audio_embedding, category_embeddings)
            best_match_idx = similarities.argmax().item()
            best_match_category = normal_categories[best_match_idx]
            confidence = similarities[0, best_match_idx].item()
            
            return f"Normal audio. Type: {best_match_category} (confidence: {confidence:.2f})"





class AudioAnomalyDataset(Dataset):
    def __init__(self, num_samples, audio_length, text_length, normal_categories):
        self.num_samples = num_samples
        self.audio_data = torch.randn(num_samples, audio_length)
        self.text_data = [f"Sample text {i}" for i in range(num_samples)]
        self.anomaly_labels = torch.randint(0, 2, (num_samples,))
        self.severity_labels = torch.rand(num_samples)
        self.normal_categories = normal_categories
        self.assigned_categories = None  # This will be set later

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        audio = self.audio_data[idx]
        text = self.text_data[idx]
        anomaly_label = self.anomaly_labels[idx]
        severity_label = self.severity_labels[idx]
        assigned_category = self.assigned_categories[idx] if self.assigned_categories is not None else None
        return audio, text, anomaly_label, severity_label, assigned_category

    def set_assigned_categories(self, assigned_categories):
        self.assigned_categories = assigned_categories


def getPANN(device):

    pann = PANN(sample_rate=16000, window_size=1024, hop_size=160, mel_bins=64, fmin=50, fmax=8000)

    state_dict = torch.load('Cnn14_mAP=0.431.pth')
    state_dict =  state_dict['model']

    model_dict = pann.state_dict()


    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}


    model_dict.update(pretrained_dict) 


    pann.load_state_dict(model_dict, strict=False)

    pann = pann.to(device)

    return pann

def get_text_embeddings(texts, model, tokenizer, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]


def assign_normal_categories(text_data, normal_categories, model, tokenizer, device):
    text_embeddings = get_text_embeddings(text_data, model, tokenizer, device)
    category_embeddings = get_text_embeddings(normal_categories, model, tokenizer, device)
    
    similarities = F.cosine_similarity(text_embeddings.unsqueeze(1), category_embeddings.unsqueeze(0), dim=2)
    assigned_categories = similarities.argmax(dim=1)
    
    return [normal_categories[i] for i in assigned_categories.tolist()]

def save_metrics(epoch, train_metrics, test_results, filename='training_metrics.csv'):
    fieldnames = [
        'epoch', 'train_loss', 'train_acc', 'train_auc', 'train_pauc', 'train_precision', 
        'train_recall', 'train_f1', 'train_mse', 'train_severity_low', 'train_severity_medium', 
        'train_severity_high', 'train_category_accuracy', 'test_loss', 'test_total_acc', 
        'test_total_auc', 'test_total_pauc', 'test_total_precision', 'test_total_recall', 
        'test_total_f1', 'test_total_mse', 'test_normal_acc', 'test_normal_auc', 'test_normal_pauc', 
        'test_normal_precision', 'test_normal_recall', 'test_normal_f1', 'test_normal_mse', 
        'test_anomaly_acc', 'test_anomaly_auc', 'test_anomaly_pauc', 'test_anomaly_precision', 
        'test_anomaly_recall', 'test_anomaly_f1', 'test_anomaly_mse', 'test_severity_low', 
        'test_severity_medium', 'test_severity_high', 'test_category_accuracy'
    ]

    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        row = {
            'epoch': epoch,
            'train_loss': train_metrics[0],
            'train_acc': train_metrics[1],
            'train_auc': train_metrics[2],
            'train_pauc': train_metrics[3],
            'train_precision': train_metrics[4],
            'train_recall': train_metrics[5],
            'train_f1': train_metrics[6],
            'train_mse': train_metrics[7],
            'train_severity_low': train_metrics[8].get('low', 0),
            'train_severity_medium': train_metrics[8].get('medium', 0),
            'train_severity_high': train_metrics[8].get('high', 0),
            'train_category_accuracy': train_metrics[9],
            'test_loss': test_results['total_loss'],
            'test_total_acc': test_results['total_metrics'][0],
            'test_total_auc': test_results['total_metrics'][1],
            'test_total_pauc': test_results['total_metrics'][2],
            'test_total_precision': test_results['total_metrics'][3],
            'test_total_recall': test_results['total_metrics'][4],
            'test_total_f1': test_results['total_metrics'][5],
            'test_total_mse': test_results['total_metrics'][6],
            'test_normal_acc': test_results['normal_metrics'][0],
            'test_normal_auc': test_results['normal_metrics'][1],
            'test_normal_pauc': test_results['normal_metrics'][2],
            'test_normal_precision': test_results['normal_metrics'][3],
            'test_normal_recall': test_results['normal_metrics'][4],
            'test_normal_f1': test_results['normal_metrics'][5],
            'test_normal_mse': test_results['normal_metrics'][6],
            'test_anomaly_acc': test_results['anomaly_metrics'][0],
            'test_anomaly_auc': test_results['anomaly_metrics'][1],
            'test_anomaly_pauc': test_results['anomaly_metrics'][2],
            'test_anomaly_precision': test_results['anomaly_metrics'][3],
            'test_anomaly_recall': test_results['anomaly_metrics'][4],
            'test_anomaly_f1': test_results['anomaly_metrics'][5],
            'test_anomaly_mse': test_results['anomaly_metrics'][6],
            'test_severity_low': test_results['severity_accuracy'].get('low', 0),
            'test_severity_medium': test_results['severity_accuracy'].get('medium', 0),
            'test_severity_high': test_results['severity_accuracy'].get('high', 0),
            'test_category_accuracy': test_results['category_accuracy']
        }
        
        writer.writerow(row)

def print_metrics(metrics, prefix=""):
    if isinstance(metrics, dict):
        # 处理评估结果字典
        print(f"{prefix}Loss: {metrics['total_loss']:.4f}")
        print(f"{prefix}Acc: {metrics['total_metrics'][0]:.4f}, AUC: {metrics['total_metrics'][1]:.4f}, pAUC: {metrics['total_metrics'][2]:.4f}")
        print(f"{prefix}Precision: {metrics['total_metrics'][3]:.4f}, Recall: {metrics['total_metrics'][4]:.4f}, F1: {metrics['total_metrics'][5]:.4f}, MSE: {metrics['total_metrics'][6]:.4f}")
        print(f"{prefix}Severity Accuracy: {metrics['severity_accuracy']}")
        print(f"{prefix}Category Accuracy: {metrics['category_accuracy']:.4f}")
        if 'severity_level_accuracy' in metrics:
            print(f"{prefix}Severity Level Accuracy: {metrics['severity_level_accuracy']:.4f}")
    else:

        loss, acc, auc, pauc, precision, recall, f1, mse, severity_accuracy, category_accuracy, epoch_time = metrics
        
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        pauc_str = f"{pauc:.4f}" if not np.isnan(pauc) else "N/A"
        
        print(f"{prefix}Loss: {loss:.4f}")
        print(f"{prefix}Acc: {acc:.4f}, AUC: {auc_str}, pAUC: {pauc_str}")
        print(f"{prefix}Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, MSE: {mse:.4f}")
        print(f"{prefix}Severity Accuracy: {severity_accuracy}")
        print(f"{prefix}Category Accuracy: {category_accuracy:.4f}")
        print(f"{prefix}Epoch Time: {epoch_time:.2f}s")

def save_final_metrics(results, filename='final_validation_metrics.csv'):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    writer.writerow([f"{key}_{sub_key}", sub_value])
            else:
                writer.writerow([key, value])

# 主程序
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    normal_categories = [
        "fan", "gearbox", "pump", "slider", "valve",  # MIMII categories
        "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
        "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",  # UrbanSound8K categories
        "airplane", "rain", "thunderstorm", "train", "vacuum_cleaner",
        "washing_machine", "clock_tick", "person_sneeze", "helicopter", "mouse_click"  # Additional common sounds
    ]
    

    pann = getPANN(device)
    
    model = FlexibleAnomalyAudioClassifier(pann, sample_rate=16000, min_segment_size=16000, 
                                           max_segment_size=48000, device=device, 
                                           normal_categories=normal_categories).to(device)


    train_val_dataset = SyntheticAudioDataset('dataset/synthetic_dataset/train_metadata.csv', 
                                              'dataset/synthetic_dataset')
    

    train_size = int(0.8 * len(train_val_dataset))
    test_size = len(train_val_dataset) - train_size
    train_dataset, test_dataset = random_split(train_val_dataset, [train_size, test_size])


    val_dataset = SyntheticAudioDataset('dataset/synthetic_validation_dataset/validation_metadata.csv', 
                                        'dataset/synthetic_validation_dataset')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f'training_metrics_{timestamp}.csv'


    total_start_time = time.time()
    for epoch in range(num_epochs):
        train_metrics = train(model, train_loader, optimizer, device, epoch, num_epochs, normal_categories)
        test_results = evaluate(model, test_loader, device, normal_categories)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Train Results:")
        print_metrics(train_metrics)
        print("\nTest Results:")
        print_metrics(test_results)  
        

        save_metrics(epoch, train_metrics, test_results, filename=metrics_filename)


    val_results = evaluate(model, val_loader, device, normal_categories)
    print("\nFinal Validation Results:")
    print_metrics(val_results)


    save_final_metrics(val_results, filename='final_validation_metrics.csv')

    print(f"\nTraining completed. Total time: {(time.time() - total_start_time)/3600:.2f} hours")
    print(f"Training metrics saved to {metrics_filename}")


    model_filename = f'anomaly_detection_model_{timestamp}.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()







