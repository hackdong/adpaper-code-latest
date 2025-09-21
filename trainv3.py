
import os
import json
import logging
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from semantictreemodelv3 import AudioDataset, EventDetectionModel, MachineAnalysisModel, EventDetectionLoss, MachineAnalysisLoss
import torch.nn.functional as F
from sklearn.metrics import f1_score

def custom_collate_fn(batch):

    if len(batch) == 0:
        return {}
    

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

def setup_logging(run_dir):

    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    

    fh = logging.FileHandler(os.path.join(run_dir, 'training.log'))
    fh.setLevel(logging.INFO)
    

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    

    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def create_run_directory():

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', timestamp)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
    return run_dir

def plot_metrics(metrics_dict, save_path):

    key_metrics = [
        'event_detection_accuracy',
        'event_classification_accuracy',
        'device_accuracy',
        'anomaly_accuracy'
    ]
    
    for metric_name in key_metrics:
        if metric_name in metrics_dict:
            plt.figure(figsize=(10, 6))
            plt.plot(metrics_dict[metric_name]['train'], label='Train')
            plt.plot(metrics_dict[metric_name]['val'], label='Validation')
            plt.title(f'{metric_name} over epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(os.path.join(save_path, f'{metric_name}.png'))
            plt.close()

def evaluate_model(model, dataloader, criterion, device):

    model.eval()
    total_metrics = {
        'total_loss': 0.0,
        'device_accuracy': 0.0,  
        'anomaly_accuracy': 0.0, 
        'device_total': 0,
        'device_correct': 0,
        'anomaly_total': 0,
        'anomaly_correct': 0
    }
    
    predictions_data = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            inputs = batch['audio'].to(device)
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch['labels'].items()}
            
            outputs = model(inputs)
            loss_results = criterion(outputs, targets)
            

            for key in ['total_loss', 'device_loss', 'anomaly_loss', 
                       'classification_loss', 'classification_accuracy']:
                if key in loss_results:
                    total_metrics[key] += loss_results[key]
            

            device_preds = torch.argmax(outputs['device_logits'], dim=1)
            total_metrics['device_correct'] += (device_preds == targets['device_type_idx']).sum().item()
            total_metrics['device_total'] += targets['device_type_idx'].size(0)
            

            anomaly_preds = (outputs['is_normal'].squeeze(1) > 0.5).float()
            total_metrics['anomaly_correct'] += (anomaly_preds == targets['is_normal']).sum().item()
            total_metrics['anomaly_total'] += targets['is_normal'].size(0)
            

            pred_devices = torch.argmax(outputs['device_logits'], dim=1)
            predictions_data.append({
                'file_names': batch['labels'].get('file_name', []),
                'predicted_device': pred_devices.cpu().tolist(),
                'true_device': targets['device_type_idx'].cpu().tolist(),
                'is_normal_pred': anomaly_preds.cpu().tolist(),
                'is_normal_true': targets['is_normal'].cpu().tolist(),
                'overlay_ratio': batch['labels'].get('overlay_ratio', [0] * len(pred_devices))
            })
    

    num_batches = len(dataloader)
    for key in ['total_loss', 'device_loss', 'anomaly_loss', 
                'classification_loss', 'classification_accuracy']:
        if key in total_metrics:
            total_metrics[key] /= num_batches
    

    if total_metrics['device_total'] > 0:
        total_metrics['device_accuracy'] = total_metrics['device_correct'] / total_metrics['device_total']
    if total_metrics['anomaly_total'] > 0:
        total_metrics['anomaly_accuracy'] = total_metrics['anomaly_correct'] / total_metrics['anomaly_total']
    
    return total_metrics, predictions_data

def validate_event_detection(model, val_loader, criterion, device, epoch):

    model.eval()
    total_loss = 0
    total_mask_loss = 0
    total_embedding_loss = 0
    total_classification_loss = 0
    

    metrics_sum = {
        'precision': 0, 
        'recall': 0, 
        'f1': 0, 
        'accuracy': 0,
        'event_total': 0,
        'event_correct': 0,
        'classification_total': 0,
        'classification_correct': 0
    }
    batch_count = 0
    
    def move_to_device(batch, device):

        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {key: move_to_device(value, device) for key, value in batch.items()}
        elif isinstance(batch, list):
            return [move_to_device(item, device) for item in batch]
        else:
            return batch

    progress_bar = tqdm(val_loader, desc='Validation')
    
    try:
        with torch.no_grad():
            for batch in progress_bar:
                try:

                    inputs = batch['audio'].to(device)
                    labels = move_to_device(batch['labels'], device)
                    

                    outputs = model(inputs)
                    

                    loss_dict = criterion(outputs, labels)
                    

                    total_loss += loss_dict['total_loss'].item()
                    total_mask_loss += loss_dict['mask_loss'].item()
                    total_embedding_loss += loss_dict['embedding_loss'].item()
                    total_classification_loss += loss_dict['classification_loss'].item()
                    

                    for key, value in loss_dict['mask_metrics'].items():
                        if key in metrics_sum:
                            metrics_sum[key] += value
                        else:
                            logging.debug(f"Skipping unknown metric: {key}")
                    
                    batch_count += 1
                    

                    progress_bar.set_description(
                        f'Validation - Loss: {total_loss/batch_count:.4f}'
                    )
                    
                except Exception as e:
                    logging.error(f"Error in validation batch: {str(e)}")
                    continue
                
    except Exception as e:
        logging.error(f"Error during validation: {str(e)}")

    

    avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    avg_mask_loss = total_mask_loss / batch_count if batch_count > 0 else float('inf')
    avg_embedding_loss = total_embedding_loss / batch_count if batch_count > 0 else float('inf')
    avg_classification_loss = total_classification_loss / batch_count if batch_count > 0 else float('inf')
    

    avg_metrics = {}
    for key, value in metrics_sum.items():
        if key in ['event_correct', 'classification_correct']:

            total_key = f"{key.split('_')[0]}_total"
            if metrics_sum[total_key] > 0:
                avg_metrics[f"{key.split('_')[0]}_accuracy"] = value / metrics_sum[total_key]
            else:
                avg_metrics[f"{key.split('_')[0]}_accuracy"] = 0
        elif not key.endswith('_total'):  
            avg_metrics[key] = value / batch_count if batch_count > 0 else 0
    

    print(f"\nEpoch {epoch} Validation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Mask Loss: {avg_mask_loss:.4f}")
    print(f"Average Embedding Loss: {avg_embedding_loss:.4f}")
    print(f"Average Classification Loss: {avg_classification_loss:.4f}")
    print("Metrics:", {k: f"{v:.4f}" for k, v in avg_metrics.items()})
    
    return {
        'val_loss': avg_loss,
        'val_mask_loss': avg_mask_loss,
        'val_embedding_loss': avg_embedding_loss,
        'val_classification_loss': avg_classification_loss,
        'val_metrics': avg_metrics
    }

def train_event_detection(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device, 
    run_dir, 
    num_epochs=25,
    scheduler=None,
    early_stopping_patience=None
):

    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_losses': [], 'val_losses': [], 'val_metrics': [], 'train_cc_acc': []}


    os.makedirs(run_dir, exist_ok=True)
    best_model_path = os.path.join(run_dir, 'best_event_model.pth')
    
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        running_mask_loss = 0.0
        running_embedding_loss = 0.0
        running_classification_loss = 0.0
        running_classification_acc = 0.0
        
        for batch_idx, batch in enumerate(train_loader):

            inputs = batch['audio'].to(device)
            labels = batch['labels']
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_dict = criterion(outputs, labels)
            

            loss_dict['total_loss'].backward()
            optimizer.step()
            

            running_loss += loss_dict['total_loss'].item()
            running_mask_loss += loss_dict['mask_loss'].item()
            running_embedding_loss += loss_dict['embedding_loss'].item()
            running_classification_loss += loss_dict['classification_loss'].item()
            running_classification_acc += loss_dict['mask_metrics']['classification_accuracy']
            

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Total Loss: {loss_dict["total_loss"].item():.4f}, '
                      f'Mask Loss: {loss_dict["mask_loss"].item():.4f}, '
                      f'Embedding Loss: {loss_dict["embedding_loss"].item():.4f}, '
                      f'Class Loss: {loss_dict["classification_loss"].item():.4f}, '
                      f'Class Acc: {loss_dict["mask_metrics"]["classification_accuracy"]:.4f}')


        train_loss = running_loss / len(train_loader)
        train_cc_acc = running_classification_acc / len(train_loader)


        model.eval()
        val_running_loss = 0.0
        val_running_acc = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['audio'].to(device)
                labels = batch['labels']
                
                outputs = model(inputs)
                loss_dict = criterion(outputs, labels)
                
                val_running_loss += loss_dict['total_loss'].item()
                val_running_acc += loss_dict["mask_metrics"]['classification_accuracy']
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = val_running_acc / len(val_loader)
        print(f'Val Accuracy: {val_acc:.4f}')

        val_results = validate_event_detection(model, val_loader, criterion, device, epoch)
        val_loss = val_results['val_loss']
        

        if scheduler is not None:
            scheduler.step(val_loss)
        

        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['val_metrics'].append(val_results['val_metrics'])
        history['train_cc_acc'].append(train_cc_acc)
        

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_results['val_metrics']
            }, best_model_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        

        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    return history

def train_machine_analysis(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device,
    run_dir,  
    num_epochs=25,
    scheduler=None,
    early_stopping_patience=None
):

    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}


    os.makedirs(run_dir, exist_ok=True)
    best_model_path = os.path.join(run_dir, 'best_machine_model.pth')
    
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        running_device_loss = 0.0
        running_anomaly_loss = 0.0
        running_classification_loss = 0.0
        running_classification_acc = 0.0
        
        for batch_idx, batch in enumerate(train_loader):

            inputs = batch['audio'].to(device)
            labels = batch['labels']
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_dict = criterion(outputs, labels)
            

            loss_dict['total_loss'].backward()
            optimizer.step()
            

            running_loss += loss_dict['total_loss'].item()
            running_device_loss += loss_dict['device_loss'].item()
            running_anomaly_loss += loss_dict['anomaly_loss'].item()
            running_classification_loss += loss_dict['classification_loss'].item()
            running_classification_acc += loss_dict['classification_accuracy']
            

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Total Loss: {loss_dict["total_loss"].item():.4f}, '
                      f'Device Loss: {loss_dict["device_loss"].item():.4f}, '
                      f'Anomaly Loss: {loss_dict["anomaly_loss"].item():.4f}, '
                      f'Class Loss: {loss_dict["classification_loss"].item():.4f}, '
                      f'Class Acc: {loss_dict["classification_accuracy"]:.4f}')


        train_loss = running_loss / len(train_loader)
        train_acc = running_classification_acc / len(train_loader)
        

        model.eval()
        val_running_loss = 0.0
        val_running_acc = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['audio'].to(device)
                labels = batch['labels']
                
                outputs = model(inputs)
                loss_dict = criterion(outputs, labels)
                
                val_running_loss += loss_dict['total_loss'].item()
                val_running_acc += loss_dict['classification_accuracy']
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = val_running_acc / len(val_loader)
        

        if scheduler is not None:
            scheduler.step(val_loss)
        

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, best_model_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        

        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    return history

def save_training_history(history, run_dir):

    history_path = os.path.join(run_dir, 'training_history.json')
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj


    serializable_history = convert_to_serializable(history)
    
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f, indent=4)
    
    print(f"Training history saved to {history_path}")

def save_predictions(predictions_data, run_dir, filename='test_predictions.json'):

    with open(os.path.join(run_dir, filename), 'w') as f:
        json.dump(predictions_data, f, indent=4)

def main():

    run_dir = create_run_directory()
    

    logger = setup_logging(run_dir)
    

    config = {

        'data_paths': {
            'train_metadata': './dataset/synthetic_dataset_v3/train_metadata.csv',

            'semantic_tree': './semantic_treev3.json',
        },

        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 30,
        'temperature': 0.07,
    }
    
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    

    try:
        print(f"Creating datasets from {config['data_paths']['train_metadata']}")
        

        train_dataset = AudioDataset(
            metadata_path=config['data_paths']['train_metadata'],
            semantic_tree_path=config['data_paths']['semantic_tree'],
            split='train',
            train_ratio=0.8,  
            random_seed=42    
        )
        

        val_dataset = AudioDataset(
            metadata_path=config['data_paths']['train_metadata'],
            semantic_tree_path=config['data_paths']['semantic_tree'],
            split='val',
            train_ratio=0.8,  
            random_seed=42    
        )
        

        

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        

        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'],
            shuffle=True, 
            num_workers=5,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'],
            shuffle=False, 
            num_workers=5,
            persistent_workers=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
        

        
    except Exception as e:
        logger.error(f"Error creating datasets: {str(e)}")
        raise
    

    logger.info("Starting Event Detection Training...")
    event_model = EventDetectionModel(
        num_event_categories=len(train_dataset.event_categories)
    ).to(device)  
    
    event_criterion = EventDetectionLoss(train_dataset.semantic_tree)
    event_optimizer = torch.optim.Adam(event_model.parameters(), 
                                     lr=config['learning_rate'])
    event_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        event_optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    event_history = train_event_detection(
        event_model, 
        train_loader, 
        val_loader, 
        event_criterion, 
        event_optimizer,
        device,
        run_dir, 
        num_epochs=config['num_epochs'],
        scheduler=event_scheduler,

    )

    save_training_history(event_history, run_dir)


    logger.info("Starting Machine Analysis Training...")
    machine_model = MachineAnalysisModel(
        num_device_types=len(train_dataset.device_types),  # Add this argument
        pretrained_event_model=event_model
    )
    machine_model.to(device)
    machine_criterion = MachineAnalysisLoss(train_dataset.semantic_tree)
    machine_optimizer = torch.optim.Adam(machine_model.parameters(), 
                                       lr=config['learning_rate'])
    machine_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        machine_optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    machine_history = train_machine_analysis(
        machine_model,
        train_loader,
        val_loader,
        machine_criterion,
        machine_optimizer,
        device,
        run_dir,  
        num_epochs=config['num_epochs'],
        scheduler=machine_scheduler,
    )
    


    save_training_history(machine_history, run_dir) 

    logger.info("Training completed!")



if __name__ == "__main__":
    main() 