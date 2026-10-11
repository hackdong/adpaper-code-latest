import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from models import AudioDataset, EventDetectionModel, MachineAnalysisModel, EventDetectionLoss, MachineAnalysisLoss
from semantic_tree_embedding import compress_embeddings

# disable tokenizers parallelism to avoid deadlocks
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def setup_logging(run_dir):
    logger = logging.getLogger('train_crossval')
    logger.setLevel(logging.DEBUG)  # switch to DEBUG level
    # detailed format with timestamps
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    fh = logging.FileHandler(os.path.join(run_dir, 'train_crossval.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # also configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    return logger


def create_run_directory(base_dir='runs/train_crossval'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def custom_collate_fn(batch):
    if len(batch) == 0:
        return {}

    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    elem = batch[0]
    if isinstance(elem, dict):
        return {key: custom_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, torch.Tensor):
        try:
            return torch.stack(batch, 0)
        except Exception:
            return batch
    elif isinstance(elem, (int, float)):
        try:
            return torch.tensor(batch)
        except Exception:
            return batch
    elif isinstance(elem, (str, type(None))):
        return batch
    elif isinstance(elem, list):
        return [custom_collate_fn(samples) for samples in zip(*batch)]
    else:
        return batch


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def prepare_metadata_with_subfolder(fold_dir, split):
    metadata_path = os.path.join(fold_dir, f'{split}_metadata.csv')
    print(f"[VERBOSE] Loading metadata from {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    print(f"[VERBOSE] Loaded {len(metadata)} samples from {split} split")
    metadata['file_name'] = metadata['file_name'].apply(lambda x: os.path.join(split, x))
    prepared_path = os.path.join(fold_dir, f'{split}_metadata_prefixed.csv')
    metadata.to_csv(prepared_path, index=False)
    print(f"[VERBOSE] Saved prefixed metadata to {prepared_path}")
    return prepared_path


def validate_event_detection(model, dataloader, criterion, device):
    print('[VERBOSE] Starting validation...')
    model.eval()
    total = 0
    sum_loss = 0.0
    sum_mask_loss = 0.0
    sum_embedding_loss = 0.0
    sum_classification_loss = 0.0
    metrics_accum = {
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'accuracy': 0.0,
        'classification_accuracy': 0.0
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validating', leave=False)):
            if batch_idx == 0:
                print(f'[VERBOSE] Validation batch 0 - audio shape: {batch["audio"].shape}')
            inputs = batch['audio'].to(device)
            labels = batch['labels']
            outputs = model(inputs)
            loss_dict = criterion(outputs, labels)

            batch_size = inputs.size(0)
            total += 1
            sum_loss += loss_dict['total_loss'].item()
            sum_mask_loss += loss_dict['mask_loss'].item()
            sum_embedding_loss += loss_dict['embedding_loss'].item()
            sum_classification_loss += loss_dict['classification_loss'].item()

            for key in metrics_accum:
                metrics_accum[key] += loss_dict['mask_metrics'].get(key, 0.0) if key != 'classification_accuracy' else loss_dict['mask_metrics'].get(key, 0.0)

    if total == 0:
        return None

    return {
        'val_loss': sum_loss / total,
        'val_mask_loss': sum_mask_loss / total,
        'val_embedding_loss': sum_embedding_loss / total,
        'val_classification_loss': sum_classification_loss / total,
        'val_precision': metrics_accum['precision'] / total,
        'val_recall': metrics_accum['recall'] / total,
        'val_f1': metrics_accum['f1'] / total,
        'val_accuracy': metrics_accum['accuracy'] / total,
        'val_classification_accuracy': metrics_accum['classification_accuracy'] / total
    }


def evaluate_machine_analysis(model, dataloader, criterion, device):
    model.eval()
    total = 0
    sum_loss = 0.0
    sum_device_loss = 0.0
    sum_anomaly_loss = 0.0
    sum_classification_loss = 0.0
    sum_classification_acc = 0.0
    sum_device_acc = 0.0
    sum_anomaly_acc = 0.0
    device_correct = 0
    device_total = 0
    anomaly_correct = 0
    anomaly_total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating machine', leave=False):
            inputs = batch['audio'].to(device)
            labels = batch['labels']
            outputs = model(inputs)
            loss_dict = criterion(outputs, labels)

            total += 1
            sum_loss += loss_dict['total_loss'].item()
            sum_device_loss += loss_dict['device_loss'].item()
            sum_anomaly_loss += loss_dict['anomaly_loss'].item()
            sum_classification_loss += loss_dict['classification_loss'].item()
            sum_classification_acc += loss_dict['classification_accuracy']
            
            # compute device classification accuracy
            device_preds = torch.argmax(outputs['device_logits'], dim=1)
            device_correct += (device_preds == labels['device_type_idx'].to(device)).sum().item()
            device_total += labels['device_type_idx'].size(0)
            
            # compute anomaly detection accuracy
            anomaly_preds = (outputs['is_normal'].squeeze(1) > 0.5).float()
            is_normal_target = labels['is_normal'].to(device)
            anomaly_correct += (anomaly_preds == is_normal_target).sum().item()
            anomaly_total += is_normal_target.size(0)

    if total == 0:
        return None

    return {
        'val_loss': sum_loss / total,
        'val_device_loss': sum_device_loss / total,
        'val_anomaly_loss': sum_anomaly_loss / total,
        'val_classification_loss': sum_classification_loss / total,
        'val_classification_accuracy': sum_classification_acc / total,
        'val_device_accuracy': device_correct / device_total if device_total > 0 else 0.0,
        'val_anomaly_accuracy': anomaly_correct / anomaly_total if anomaly_total > 0 else 0.0
    }


def train_event_stage(train_loader, val_loader, device, run_dir, config):
    print(f'[VERBOSE] Creating EventDetectionModel with {len(train_loader.dataset.category_to_idx)} categories...')
    model = EventDetectionModel(num_event_categories=len(train_loader.dataset.category_to_idx))
    print(f'[VERBOSE] Moving model to {device}')
    model.to(device)
    print(f'[VERBOSE] Model moved to {device}')
    print(f'[VERBOSE] Creating loss function...')
    criterion = EventDetectionLoss(train_loader.dataset.semantic_tree)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print(f'[VERBOSE] Training setup complete')

    history = []
    best_val_loss = float('inf')
    best_path = os.path.join(run_dir, 'best_event_model.pth')

    for epoch in range(1, config['event_epochs'] + 1):
        print(f'[VERBOSE] Event Stage - Starting epoch {epoch}/{config["event_epochs"]}')
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Fold Event Train E{epoch}', leave=False)):
            if batch_idx == 0:
                print(f'[VERBOSE] First batch - audio shape: {batch["audio"].shape}')
            inputs = batch['audio'].to(device)
            labels = batch['labels']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_dict = criterion(outputs, labels)
            loss_dict['total_loss'].backward()
            optimizer.step()
            epoch_loss += loss_dict['total_loss'].item()
            batch_count += 1

        epoch_loss /= max(batch_count, 1)
        print(f'[VERBOSE] Epoch {epoch} train loss: {epoch_loss:.6f}')
        print(f'[VERBOSE] Starting validation...')
        val_results = validate_event_detection(model, val_loader, criterion, device)
        if val_results:
            print(f'[VERBOSE] Epoch {epoch} val loss: {val_results["val_loss"]:.6f}')
        scheduler.step(val_results['val_loss'] if val_results else epoch_loss)

        epoch_state = {
            'epoch': epoch,
            'train_loss': epoch_loss,
            **(val_results or {})
        }
        history.append(epoch_state)
        save_json(history, os.path.join(run_dir, 'event_history.json'))

        if val_results and val_results['val_loss'] < best_val_loss:
            best_val_loss = val_results['val_loss']
            torch.save(model.state_dict(), best_path)

    return model, history


def train_machine_stage(train_loader, val_loader, device, run_dir, event_model, config):
    print(f'[VERBOSE] Creating MachineAnalysisModel with {len(train_loader.dataset.device_types)} device types...')
    machine_model = MachineAnalysisModel(num_device_types=len(train_loader.dataset.device_types), pretrained_event_model=event_model)
    print(f'[VERBOSE] Moving machine model to {device}')
    machine_model.to(device)
    print(f'[VERBOSE] Creating machine loss function...')
    criterion = MachineAnalysisLoss(train_loader.dataset.semantic_tree)
    optimizer = torch.optim.Adam(machine_model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print(f'[VERBOSE] Machine training setup complete')

    history = []
    best_val_loss = float('inf')
    best_path = os.path.join(run_dir, 'best_machine_model.pth')

    for epoch in range(1, config['machine_epochs'] + 1):
        print(f'[VERBOSE] Machine Stage - Starting epoch {epoch}/{config["machine_epochs"]}')
        machine_model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Fold Machine Train E{epoch}', leave=False)):
            if batch_idx == 0:
                print(f'[VERBOSE] First batch - audio shape: {batch["audio"].shape}')
            inputs = batch['audio'].to(device)
            labels = batch['labels']
            optimizer.zero_grad()
            outputs = machine_model(inputs)
            loss_dict = criterion(outputs, labels)
            loss_dict['total_loss'].backward()
            optimizer.step()
            epoch_loss += loss_dict['total_loss'].item()
            batch_count += 1

        epoch_loss /= max(batch_count, 1)
        print(f'[VERBOSE] Epoch {epoch} train loss: {epoch_loss:.6f}')
        print(f'[VERBOSE] Starting machine validation...')
        val_results = evaluate_machine_analysis(machine_model, val_loader, criterion, device)
        if val_results:
            print(f'[VERBOSE] Epoch {epoch} val loss: {val_results["val_loss"]:.6f}')
            print(f'[VERBOSE]   Device Accuracy: {val_results["val_device_accuracy"]:.4f}')
            print(f'[VERBOSE]   Anomaly Accuracy: {val_results["val_anomaly_accuracy"]:.4f}')
            print(f'[VERBOSE]   Classification Accuracy: {val_results["val_classification_accuracy"]:.4f}')
        scheduler.step(val_results['val_loss'] if val_results else epoch_loss)

        epoch_state = {
            'epoch': epoch,
            'train_loss': epoch_loss,
            **(val_results or {})
        }
        history.append(epoch_state)
        save_json(history, os.path.join(run_dir, 'machine_history.json'))

        if val_results and val_results['val_loss'] < best_val_loss:
            best_val_loss = val_results['val_loss']
            torch.save(machine_model.state_dict(), best_path)

    return machine_model, history


def train_validate_fold(fold_idx, fold_dir, run_root, config, device, logger):
    logger.info(f'Starting fold {fold_idx}')
    fold_run_dir = os.path.join(run_root, f'fold_{fold_idx}')
    os.makedirs(fold_run_dir, exist_ok=True)

    train_meta = prepare_metadata_with_subfolder(fold_dir, 'train')
    test_meta = prepare_metadata_with_subfolder(fold_dir, 'test')

    logger.info(f'Loading datasets for fold {fold_idx}')
    # create the training dataset first to build the category mapping
    print(f'[VERBOSE] Creating train dataset from {train_meta}...')
    start_time = datetime.now()
    train_dataset = AudioDataset(metadata_path=train_meta, semantic_tree_path=config['semantic_tree_path'], split='all')
    print(f'[VERBOSE] Train dataset created in {(datetime.now() - start_time).total_seconds():.2f}s')
    logger.info(f'Train dataset categories: {len(train_dataset.category_to_idx)}')
    
    # create the validation dataset
    print(f'[VERBOSE] Creating val dataset from {test_meta}...')
    start_time = datetime.now()
    val_dataset = AudioDataset(metadata_path=test_meta, semantic_tree_path=config['semantic_tree_path'], split='all')
    print(f'[VERBOSE] Val dataset created in {(datetime.now() - start_time).total_seconds():.2f}s')
    logger.info(f'Val dataset categories before merge: {len(val_dataset.category_to_idx)}')
    
    # merge the category mappings of both datasets (covering all categories in train and validation)
    # this handles zero-shot categories
    print(f'[VERBOSE] Merging categories...')
    print(f'[VERBOSE] Train categories: {sorted(train_dataset.category_to_idx.keys())}')
    print(f'[VERBOSE] Val unique categories: {sorted(set(val_dataset.category_to_idx.keys()) - set(train_dataset.category_to_idx.keys()))}')
    all_categories = sorted(set(list(train_dataset.category_to_idx.keys()) + list(val_dataset.category_to_idx.keys())))
    merged_category_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}
    print(f'[VERBOSE] Merged {len(all_categories)} categories total')
    
    # generate embeddings for the newly added categories
    print(f'[VERBOSE] Generating embeddings for all categories...')
    missing_in_train = set(all_categories) - set(train_dataset.category_embeddings.keys())
    print(f'[VERBOSE] Missing embeddings in train set: {missing_in_train}')
    for cat in all_categories:
        if cat not in train_dataset.category_embeddings:
            # try to fetch from the validation set; if absent, generate it dynamically
            if cat in val_dataset.category_embeddings:
                print(f'[VERBOSE] Using embedding from val set for: {cat}')
                train_dataset.category_embeddings[cat] = val_dataset.category_embeddings[cat]
            else:
                # dynamically generate the missing category embedding
                print(f'[VERBOSE] Generating embedding for zero-shot category: {cat}')
                try:
                    encoded = train_dataset.text_encoder.encode(f" {cat}").reshape(1, -1)
                    compressed = compress_embeddings(encoded)[0]
                    train_dataset.category_embeddings[cat] = torch.tensor(compressed, dtype=torch.float32)
                    print(f'[VERBOSE] Successfully generated embedding for {cat}')
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for {cat}: {e}, using zero tensor")
                    train_dataset.category_embeddings[cat] = torch.zeros(32)
    
    # update both datasets with the merged mapping and embeddings
    print(f'[VERBOSE] Updating datasets with merged category mappings...')
    train_dataset.category_to_idx = merged_category_to_idx
    val_dataset.category_to_idx = merged_category_to_idx
    val_dataset.category_embeddings = train_dataset.category_embeddings
    print(f'[VERBOSE] Datasets updated. Total categories: {len(merged_category_to_idx)}')

    logger.info(f'Train size: {len(train_dataset)}, Val size: {len(val_dataset)}')

    print(f'[VERBOSE] Creating DataLoaders...')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # disable multiprocessing to avoid CUDA/SentenceTransformer subprocess issues
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    print(f'[VERBOSE] Train DataLoader created')
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,  # disable multiprocessing to avoid CUDA/SentenceTransformer subprocess issues
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    print(f'[VERBOSE] Val DataLoader created')

    event_dir = os.path.join(fold_run_dir, 'event_stage')
    os.makedirs(event_dir, exist_ok=True)
    print(f'[VERBOSE] Starting Event Detection stage training...')
    event_model, event_history = train_event_stage(train_loader, val_loader, device, event_dir, config)
    print(f'[VERBOSE] Event Detection stage completed')

    machine_dir = os.path.join(fold_run_dir, 'machine_stage')
    os.makedirs(machine_dir, exist_ok=True)
    print(f'[VERBOSE] Starting Machine Analysis stage training...')
    machine_model, machine_history = train_machine_stage(train_loader, val_loader, device, machine_dir, event_model, config)
    print(f'[VERBOSE] Machine Analysis stage completed')

    fold_result = {
        'fold': fold_idx,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'event_history': event_history,
        'machine_history': machine_history
    }
    save_json(fold_result, os.path.join(fold_run_dir, 'fold_summary.json'))
    logger.info(f'Fold {fold_idx} finished')
    return fold_result


def main():
    config = {
        'semantic_tree_path': './semantic_tree.json',
        'batch_size': 64,
        'learning_rate': 1e-3,
        'event_epochs': 25,
        'machine_epochs': 25,
        'num_workers': 0,  # disable multiprocessing to avoid CUDA/SentenceTransformer subprocess issues
        'folds': 10
    }

    run_root = create_run_directory()
    logger = setup_logging(run_root)
    logger.info(f'Run root: {run_root}')
    
    # determine the device and log it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    if torch.cuda.is_available():
        logger.info(f'CUDA device: {torch.cuda.get_device_name(0)}')
        logger.info(f'CUDA version: {torch.version.cuda}')

    fold_results = []
    for fold_idx in range(config['folds']):
        fold_dir = os.path.join('dataset', 'synthetic_dataset', f'fold_{fold_idx}')
        if not os.path.exists(fold_dir):
            logger.warning(f'Fold directory not found: {fold_dir}, skipping')
            continue

        fold_result = train_validate_fold(fold_idx, fold_dir, run_root, config, device, logger)
        fold_results.append(fold_result)

    save_json(fold_results, os.path.join(run_root, 'all_folds_summary.json'))
    logger.info('10-fold training and validation complete')


if __name__ == '__main__':
    main()
