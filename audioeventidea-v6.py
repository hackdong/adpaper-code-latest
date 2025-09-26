# Import required libraries

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import torch.nn.functional as F
import whisper
from transformers import WhisperProcessor, WhisperModel

class AudioEventDataset(Dataset):
    """
    Custom Dataset class for audio event detection.
    Handles loading and preprocessing of audio files and their corresponding labels.
    """
    def __init__(self, csv_path, audio_dir, sample_rate=16000, duration=10):
        """
        Initialize the dataset.
        Args:
            csv_path (str): Path to the CSV file containing metadata
            audio_dir (str): Directory containing audio files
            sample_rate (int): Target sample rate for audio
            duration (int): Duration of audio clips in seconds
        """
        self.data = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Get unique device types and convert embeddings to tensors
        self.device_types = list(self.data['device_type'].dropna().unique())
        self.device_embeddings = {
            device: torch.tensor(self.embedding_model.encode(device), dtype=torch.float32)
            for device in self.device_types
        }
        
        # Get unique overlay categories and convert embeddings to tensors
        self.overlay_categories = list(self.data['overlay_category'].dropna().unique())
        self.overlay_embeddings = {
            cat: torch.tensor(self.embedding_model.encode(cat), dtype=torch.float32)
            for cat in self.overlay_categories
        }
        
        # Get embedding dimension
        self.embedding_dim = next(iter(self.device_embeddings.values())).shape[0]
        
        # Spectrogram transforms
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        
        # Initialize Whisper model and processor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.whisper_model = WhisperModel.from_pretrained("openai/whisper-small").to(device)
        
        # Freeze Whisper parameters
        for param in self.whisper_model.parameters():
            param.requires_grad = False
        
        # Get decoder start token id
        self.decoder_start_token_id = self.whisper_processor.tokenizer.get_decoder_prompt_ids()

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load and process audio
        audio_path = os.path.join(self.audio_dir, row['file_name'])
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
        if waveform.shape[1] < self.num_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.num_samples]

        # Convert to mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # Get device embedding
        device_embedding = torch.zeros(self.embedding_dim, dtype=torch.float32)
        if pd.notna(row['device_type']):
            device_embedding = self.device_embeddings[row['device_type']]
        
        # Get overlay embedding
        overlay_embedding = torch.zeros(self.embedding_dim, dtype=torch.float32)
        if pd.notna(row['overlay_category']):
            overlay_embedding = self.overlay_embeddings[row['overlay_category']]
        
        # Create time mask
        time_mask = torch.zeros(self.num_samples)
        if pd.notna(row['overlay_start_time']) and pd.notna(row['overlay_end_time']):
            start_idx = int(float(row['overlay_start_time']) * self.sample_rate)
            end_idx = int(float(row['overlay_end_time']) * self.sample_rate)
            time_mask[start_idx:end_idx] = 1.0
        
        # Get normal/abnormal state
        normal_state = torch.tensor([1.0, 0.0] if row['is_normal'] else [0.0, 1.0], 
                                  dtype=torch.float32)
        
        # Get Whisper features with no_grad to ensure parameters stay frozen
        with torch.no_grad():
            # Process audio input
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_features = self.whisper_processor(
                waveform.squeeze().numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_features.to(device)
            
            # Create decoder input ids
            decoder_input_ids = torch.tensor([[self.decoder_start_token_id]], device=device)
            
            # Get Whisper outputs
            whisper_outputs = self.whisper_model(
                input_features,
                decoder_input_ids=decoder_input_ids
            )
            
            # Extract encoder hidden states as features
            whisper_features = whisper_outputs.encoder_last_hidden_state

        return {
            'mel_spec': mel_spec.requires_grad_(True),
            'whisper_features': whisper_features.squeeze(0).detach(),  # Detach to ensure no gradients
            'overlay_embedding': overlay_embedding.requires_grad_(True),
            'device_embedding': device_embedding.requires_grad_(True),
            'normal_state': normal_state.requires_grad_(True),
            'time_mask': time_mask.requires_grad_(True),
            'overlay_ratio': torch.tensor(
                float(row['overlay_ratio']) if pd.notna(row['overlay_ratio']) else 0.0,
                dtype=torch.float32,
                requires_grad=True
            )
        }

    def __len__(self):
        return len(self.data)

class AudioEventDetector(nn.Module):
    """
    Neural network model for audio event detection.
    Performs both event classification and temporal localization.
    """
    def __init__(self, event_embedding_dim, device_embedding_dim):
        """
        Initialize the model architecture.
        Args:
            embedding_dim (int): Dimension of the event embeddings
            sample_rate (int): Audio sample rate (default: 16000)
            duration (int): Duration of audio in seconds (default: 10)
        """
        super().__init__()
        
        # Feature extraction blocks (from v4)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Dropout(0.2)
        )

        # Adaptive pooling and flatten size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 25))
        self.flatten_size = 512 * 4 * 25  # This is 51200
        
        # Add Whisper feature processing layers
        self.whisper_processor = nn.Sequential(
            nn.Linear(768, 512),  # Whisper hidden size is 768
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2)
        )
        
        # Modify feature fusion to handle the correct dimensions
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.flatten_size + 512, 1024),  # 51200 + 512 -> 1024
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2)
        )
        
        # Update TimeAwareFeatureExtractor input dimension
        self.normal_feature_extractor = TimeAwareFeatureExtractor(1024)  # Changed from self.flatten_size
        
        # Event classification branch with correct input dimension
        self.event_classifier = EventClassifier(
            input_dim=1024,  # Changed from self.flatten_size to 1024
            embedding_dim=event_embedding_dim
        )
        
        # Device classification branch with correct input dimension
        self.device_classifier = nn.Sequential(
            nn.Linear(1024, 512),  # Changed from self.flatten_size
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, device_embedding_dim)
        )
        
        self.normal_feature_extractor = TimeAwareFeatureExtractor(1024)
        self.normal_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        
        # Time detection branch (from v4)
        self.time_branch = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Conv1d(256, 1, kernel_size=1)
        )
        
        self.upsample = nn.Upsample(size=16000*10, mode='linear', align_corners=True)
        self.sigmoid = nn.Sigmoid()
        
        # Add Whisper feature processing layers
        self.whisper_processor = nn.Sequential(
            nn.Linear(768, 512),  # Whisper hidden size is 768
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2)
        )
        
        # Modify feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.flatten_size + 512, 1024),  # Combine CNN and Whisper features
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2)
        )

    def forward(self, x, whisper_features):
        # Add channel dimension if necessary
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Ensure input requires gradients
        x = x.requires_grad_()
        
        # Feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        features = x
        
        # Shared pooled features
        pooled_features = self.adaptive_pool(features)
        flat_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Time detection first
        time_output = self.sigmoid(self.upsample(
            self.time_branch(features.mean(dim=2))
        ))
        
        # Process Whisper features
        whisper_processed = self.whisper_processor(whisper_features.mean(dim=1))
        
        # Combine features
        combined_features = torch.cat([flat_features, whisper_processed], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # Use fused features for all branches
        normal_features = self.normal_feature_extractor(fused_features, time_output)
        normal_output = self.normal_classifier(normal_features)
        
        event_embeddings, event_features = self.event_classifier(fused_features, time_output)
        
        return {
            'overlay_output': event_embeddings,
            'overlay_features': event_features,
            'device_output': self.device_classifier(fused_features),
            'normal_output': normal_output,
            'time_output': time_output
        }

class TimeAwareFeatureExtractor(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512)
        )
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512)
        )

    def forward(self, x, time_mask):

        features = self.feature_net(x)
        
        attention_weights = self.attention(x)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        if time_mask.dim() == 4:  # (batch, channel, time, 1)
            time_mask = time_mask.squeeze(-1).squeeze(1)
        elif time_mask.dim() == 3:  # (batch, 1,time)
            time_mask = time_mask.squeeze(-2)
        elif time_mask.dim() == 1:
            time_mask = time_mask.unsqueeze(0)
            
        background_attention = 1 - F.adaptive_avg_pool1d(
            time_mask.unsqueeze(1),
            features.size(-1)
        ).squeeze(1)
        
        weighted_features = features * attention_weights * background_attention
        return weighted_features

class EventClassifier(nn.Module):

    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),  # Changed from input_dim to 512
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
        )
        
        self.projection = nn.Linear(512, embedding_dim)
        self.time_attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 512)
        )

    def forward(self, x, time_mask=None):
        features = self.encoder(x)  # Now expects input of size (batch, 1024)
        
        if time_mask is not None:

            attention_weights = self.time_attention(features)  # shape: (batch, 512)
            attention_weights = F.softmax(attention_weights, dim=1)
            

            if time_mask.dim() == 4:  # (batch, channel, time, 1)
                time_mask = time_mask.squeeze(-1).squeeze(1) 
            elif time_mask.dim() == 3:  # (batch, 1, time)
                time_mask = time_mask.squeeze(-2)
            elif time_mask.dim() == 1:  # (time,)
                time_mask = time_mask.unsqueeze(0) 
            

            if time_mask.dim() != 2:
                raise ValueError(f"Expected time_mask to be 2D after processing, but got {time_mask.dim()}D")
            

            time_attention = time_mask.float()  # shape: (batch, time)
            

            if time_attention.size(1) != features.size(1):
                time_attention = F.interpolate(
                    time_attention.unsqueeze(1),  
                    size=features.size(1),
                    mode='linear',
                    align_corners=False
                ).squeeze(1)  # shape: (batch, 512)
            

            features = features * attention_weights * time_attention
        
        embeddings = self.projection(features)
        return embeddings, features

def create_training_directory():
    """
    Create a new directory for this training run with timestamp and script name
    Returns:
        str: Path to the created directory
    """
    # Create base directory if it doesn't exist
    base_dir = 'training_runs'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Get the current script name without .py extension
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, f'{script_name}_{timestamp}')
    os.makedirs(run_dir)
    
    return run_dir

def save_training_history(history, save_dir):

    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    

    plt.figure(figsize=(15, 5))
    

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    

    plt.subplot(1, 2, 2)
    metrics = ['overlay_acc', 'device_acc', 'normal_acc', 'time_acc']
    for metric in metrics:
        plt.plot(history[f'val_{metric}'], label=f'Val {metric}')
    plt.title('Validation Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def evaluate_model(model, data_loader, device):

    model.eval()
    stats = {
        'total': {'correct': 0, 'total': 0},
        'overlay_categories': {},
        'device_types': {},
        'ratio_ranges': {
            'low': {'correct': 0, 'total': 0},
            'medium': {'correct': 0, 'total': 0},
            'high': {'correct': 0, 'total': 0},
        }
    }
    

    overlay_embeddings = torch.stack([
        torch.tensor(emb, device=device)
        for emb in data_loader.dataset.overlay_embeddings.values()
    ])  # [num_categories, embedding_dim]
    
    device_embeddings = torch.stack([
        torch.tensor(emb, device=device)
        for emb in data_loader.dataset.device_embeddings.values()
    ])  # [num_devices, embedding_dim]
    
    category_list = list(data_loader.dataset.overlay_embeddings.keys())
    device_list = list(data_loader.dataset.device_embeddings.keys())
    
    with torch.no_grad():
        for batch in data_loader:

            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            outputs = model(batch['mel_spec'], batch['whisper_features'])
            

            overlay_sim = F.cosine_similarity(
                outputs['overlay_output'].unsqueeze(1),  # [B, 1, D]
                overlay_embeddings.unsqueeze(0),  # [1, C, D]
                dim=2
            )  # [B, C]
            
            device_sim = F.cosine_similarity(
                outputs['device_output'].unsqueeze(1),  # [B, 1, D]
                device_embeddings.unsqueeze(0),  # [1, D, D]
                dim=2
            )  # [B, D]
            

            pred_overlay_idx = overlay_sim.argmax(dim=1)
            pred_device_idx = device_sim.argmax(dim=1)
            

            true_overlay_sim = F.cosine_similarity(
                batch['overlay_embedding'].unsqueeze(1),
                overlay_embeddings.unsqueeze(0),
                dim=2
            )
            true_overlay_idx = true_overlay_sim.argmax(dim=1)
            
            true_device_sim = F.cosine_similarity(
                batch['device_embedding'].unsqueeze(1),
                device_embeddings.unsqueeze(0),
                dim=2
            )
            true_device_idx = true_device_sim.argmax(dim=1)
            

            overlay_correct = (pred_overlay_idx == true_overlay_idx)
            device_correct = (pred_device_idx == true_device_idx)
            

            for i, (o_correct, d_correct) in enumerate(zip(overlay_correct, device_correct)):
                true_overlay = category_list[true_overlay_idx[i]]
                true_device = device_list[true_device_idx[i]]
                
                if o_correct and d_correct:
                    stats['total']['correct'] += 1
                    if true_overlay in stats['overlay_categories']:
                        stats['overlay_categories'][true_overlay]['correct'] += 1
                    if true_device in stats['device_types']:
                        stats['device_types'][true_device]['correct'] += 1
                
                stats['total']['total'] += 1
                if true_overlay in stats['overlay_categories']:
                    stats['overlay_categories'][true_overlay]['total'] += 1
                if true_device in stats['device_types']:
                    stats['device_types'][true_device]['total'] += 1
    

    results = {
        'total_accuracy': stats['total']['correct'] / max(stats['total']['total'], 1),
        'overlay_accuracies': {},
        'device_accuracies': {},
        'ratio_accuracies': {}
    }
    

    for cat, cat_stats in stats['overlay_categories'].items():
        results['overlay_accuracies'][cat] = cat_stats['correct'] / max(cat_stats['total'], 1)
    
    for dev_type, dev_stats in stats['device_types'].items():
        results['device_accuracies'][dev_type] = dev_stats['correct'] / max(dev_stats['total'], 1)
    
    for range_key, range_stats in stats['ratio_ranges'].items():
        results['ratio_accuracies'][range_key] = range_stats['correct'] / max(range_stats['total'], 1)
    
    return results

def print_evaluation_results(results):

    print(f"\nOverall Accuracy: {results['total_accuracy']:.4f}")
    
    print("\nAccuracy by Overlay Category:")
    for cat, acc in sorted(results['overlay_accuracies'].items()):
        if cat:  
            print(f"  {cat:<20}: {acc:.4f}")
    
    print("\nAccuracy by Device Type:")
    for dev_type, acc in sorted(results['device_accuracies'].items()):
        print(f"  {dev_type:<20}: {acc:.4f}")
    
    print("\nAccuracy by Ratio Range:")
    for range_key, acc in results['ratio_accuracies'].items():
        print(f"  {range_key:<20}: {acc:.4f}")

class NTXentLoss(nn.Module):

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8
        
    def forward(self, features, masks):

        device = features.device
        

        features = F.normalize(features, p=2, dim=1)
        

        if masks.dim() == 4:  # (batch, channel, time, 1)
            masks = masks.squeeze(-1).squeeze(1)
        elif masks.dim() == 3:  # (batch, 1, time)
            masks = masks.squeeze(1)
        elif masks.dim() == 1:  # (time,)
            masks = masks.unsqueeze(0)
            

        masks = masks.float()
        
        sim_matrix = torch.matmul(features, features.t()) / self.temperature
        
        mask_time = F.adaptive_avg_pool1d(
            masks.unsqueeze(1),
            1
        ).squeeze()
        

        positive_mask = (mask_time.unsqueeze(0) > 0.5) & (mask_time.unsqueeze(1) > 0.5)
        

        diag_mask = ~torch.eye(features.size(0), dtype=torch.bool, device=device)
        positive_mask = positive_mask & diag_mask

        exp_sim = torch.exp(sim_matrix - sim_matrix.max())
        

        pos_pairs = positive_mask.float() * exp_sim
        

        denominator = exp_sim.masked_fill(~diag_mask, 0).sum(dim=1) + self.eps
        

        losses = -torch.log((pos_pairs.sum(dim=1) + self.eps) / denominator)
        

        valid_samples = positive_mask.any(dim=1)
        
        if valid_samples.sum() > 0:
            return losses[valid_samples].mean()
        else:

            return torch.tensor(0.0, device=device, requires_grad=True)

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")  
    
    model = model.to(device)
    
    embedding_criterion = nn.CosineEmbeddingLoss(reduction='mean').to(device)
    normal_criterion = nn.BCEWithLogitsLoss(reduction='mean').to(device)
    time_criterion = nn.BCELoss(reduction='mean').to(device)
    contrastive_criterion = NTXentLoss(temperature=0.5).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    

    history = {
        'train_loss': [], 'train_overlay_acc': [], 'train_device_acc': [], 
        'train_normal_acc': [], 'train_time_acc': [],
        'val_loss': [], 'val_overlay_acc': [], 'val_device_acc': [], 
        'val_normal_acc': [], 'val_time_acc': []
    }
    
    save_dir = create_training_directory()
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics = {'overlay': 0, 'device': 0, 'normal': 0, 'time': 0}
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_pbar:
            # Move to device and ensure gradients
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch['mel_spec'], batch['whisper_features'])
            
            # Calculate losses
            target = torch.ones(batch['mel_spec'].size(0), device=device)
            
            overlay_loss = embedding_criterion(
                outputs['overlay_output'], 
                batch['overlay_embedding'],
                target
            )
            device_loss = embedding_criterion(
                outputs['device_output'],
                batch['device_embedding'],
                target
            )
            normal_loss = normal_criterion(
                outputs['normal_output'],
                batch['normal_state']
            )
            time_loss = time_criterion(
                outputs['time_output'].squeeze(),
                batch['time_mask']
            )
            

            contrastive_loss = contrastive_criterion(
                outputs['overlay_features'],  
                batch['time_mask']
            )
            

            loss = overlay_loss + device_loss + normal_loss + time_loss + 0.1 * torch.clamp(contrastive_loss, min=-10, max=10)  
            

            if torch.isnan(loss):
                print("Warning: Loss is NaN!")
                print(f"overlay_loss: {overlay_loss}")
                print(f"device_loss: {device_loss}")
                print(f"normal_loss: {normal_loss}")
                print(f"time_loss: {time_loss}")
                print(f"contrastive_loss: {contrastive_loss}")
                continue  
                
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracies
            train_loss += loss.item()
            batch_size = batch['mel_spec'].size(0)
            train_total += batch_size
            
            # Update metrics
            with torch.no_grad():
                overlay_sim = F.cosine_similarity(
                    outputs['overlay_output'].unsqueeze(1),  # [B, 1, D]
                    batch['overlay_embedding'].unsqueeze(0),  # [1, B, D]
                    dim=2
                )  # [B, B]
                device_sim = F.cosine_similarity(
                    outputs['device_output'].unsqueeze(1),  # [B, 1, D]
                    batch['device_embedding'].unsqueeze(0),  # [1, B, D]
                    dim=2
                )  # [B, B]
                

                overlay_correct = (overlay_sim.argmax(dim=1) == torch.arange(batch_size, device=device))
                device_correct = (device_sim.argmax(dim=1) == torch.arange(batch_size, device=device))
                
                train_metrics['overlay'] += overlay_correct.sum().item()
                train_metrics['device'] += device_correct.sum().item()
                normal_pred = torch.argmax(outputs['normal_output'], dim=1)
                normal_true = torch.argmax(batch['normal_state'], dim=1)
                time_acc = (torch.abs(outputs['time_output'].squeeze() - 
                                    batch['time_mask']) < 0.5).float().mean()
                
                train_metrics['normal'] += (normal_pred == normal_true).sum().item()
                train_metrics['time'] += time_acc.item() * batch_size
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'overlay_acc': f'{100 * train_metrics["overlay"] / train_total:.2f}%',
                'device_acc': f'{100 * train_metrics["device"] / train_total:.2f}%'
            })
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_accuracies = {k: v / train_total for k, v in train_metrics.items()}
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {'overlay': 0, 'device': 0, 'normal': 0, 'time': 0}
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Valid]')
        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch['mel_spec'], batch['whisper_features'])
                
                # Calculate validation losses
                target = torch.ones(batch['mel_spec'].size(0), device=device)
                overlay_loss = embedding_criterion(outputs['overlay_output'], 
                                                batch['overlay_embedding'], target)
                device_loss = embedding_criterion(outputs['device_output'], 
                                               batch['device_embedding'], target)
                normal_loss = normal_criterion(outputs['normal_output'], 
                                            batch['normal_state'])
                time_loss = time_criterion(outputs['time_output'].squeeze(), 
                                         batch['time_mask'])
                
                loss = overlay_loss + device_loss + normal_loss + time_loss
                val_loss += loss.item()
                
                # Calculate validation metrics
                batch_size = batch['mel_spec'].size(0)
                val_total += batch_size
                
                overlay_sim = F.cosine_similarity(
                    outputs['overlay_output'].unsqueeze(1),  # [B, 1, D]
                    batch['overlay_embedding'].unsqueeze(0),  # [1, B, D]
                    dim=2
                )  # [B, B]
                
                device_sim = F.cosine_similarity(
                    outputs['device_output'].unsqueeze(1),  # [B, 1, D]
                    batch['device_embedding'].unsqueeze(0),  # [1, B, D]
                    dim=2
                )  # [B, B]
                

                overlay_correct = (overlay_sim.argmax(dim=1) == torch.arange(batch_size, device=device))
                device_correct = (device_sim.argmax(dim=1) == torch.arange(batch_size, device=device))
                
                val_metrics['overlay'] += overlay_correct.sum().item()
                val_metrics['device'] += device_correct.sum().item()
                normal_pred = torch.argmax(outputs['normal_output'], dim=1)
                normal_true = torch.argmax(batch['normal_state'], dim=1)
                time_acc = (torch.abs(outputs['time_output'].squeeze() - 
                                    batch['time_mask']) < 0.5).float().mean()
                
                val_metrics['normal'] += (normal_pred == normal_true).sum().item()
                val_metrics['time'] += time_acc.item() * batch_size
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'overlay_acc': f'{100 * val_metrics["overlay"] / val_total:.2f}%',
                    'device_acc': f'{100 * val_metrics["device"] / val_total:.2f}%'
                })
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_accuracies = {k: v / val_total for k, v in val_metrics.items()}
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_overlay_acc'].append(train_accuracies['overlay'])
        history['train_device_acc'].append(train_accuracies['device'])
        history['train_normal_acc'].append(train_accuracies['normal'])
        history['train_time_acc'].append(train_accuracies['time'])
        
        history['val_loss'].append(val_loss)
        history['val_overlay_acc'].append(val_accuracies['overlay'])
        history['val_device_acc'].append(val_accuracies['device'])
        history['val_normal_acc'].append(val_accuracies['normal'])
        history['val_time_acc'].append(val_accuracies['time'])
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
            }, os.path.join(save_dir, 'best_model.pth'))
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
        print(f'Training Loss: {train_loss:.4f}')
        print('Training Accuracies:')
        for k, v in train_accuracies.items():
            print(f'  {k}: {100*v:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}')
        print('Validation Accuracies:')
        for k, v in val_accuracies.items():
            print(f'  {k}: {100*v:.2f}%')
        print('-' * 50)
    
    # Save final model and training history
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
    }, os.path.join(save_dir, 'final_model.pth'))
    
    # Save training history
    save_training_history(history, save_dir)
    
    return history, save_dir

def main():
    """
    Main function to set up and run the training process.
    """
    # Define paths
    train_csv = 'dataset/synthetic_dataset/train_metadata.csv'
    train_audio_dir = 'dataset/synthetic_dataset'
    val_csv = 'dataset/synthetic_validation_dataset/validation_metadata.csv'
    val_audio_dir = 'dataset/synthetic_validation_dataset'
    
    # Initialize datasets
    train_dataset = AudioEventDataset(train_csv, train_audio_dir)
    val_dataset = AudioEventDataset(val_csv, val_audio_dir)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model and move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}") 
    
    model = AudioEventDetector(
        event_embedding_dim=train_dataset.embedding_dim,
        device_embedding_dim=train_dataset.embedding_dim
    ).to(device)
    

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model device: {next(model.parameters()).device}")

    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        lr=0.001
    )

if __name__ == "__main__":
    main()