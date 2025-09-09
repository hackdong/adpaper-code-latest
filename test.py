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
from sklearn.metrics import roc_auc_score


def setup_logging(run_dir):
    """设置日志记录器"""
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    fh = logging.FileHandler(os.path.join(run_dir, 'training.log'))
    fh.setLevel(logging.INFO)
    
    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

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

def calculate_metrics(y_true, scores, pos_ratio=0.1):
    """计算AUC和pAUC指标"""
    try:
        # 计算AUC
        auc_score = roc_auc_score(y_true, scores)
        
        # 计算pAUC
        n_samples = len(y_true)
        thresh_idx = int(n_samples * (1 - pos_ratio))
        sorted_pred_idx = np.argsort(scores)
        y_true_filtered = y_true[sorted_pred_idx[thresh_idx:]]
        scores_filtered = scores[sorted_pred_idx[thresh_idx:]]
        pauc_score = roc_auc_score(y_true_filtered, scores_filtered)
        
        return auc_score, pauc_score
    except:
        return None, None

def main():
    # 创建运行目录
    run_dir = "./runs/20241212_221047/"
    
    # 设置日志记录器
    logger = setup_logging(run_dir)
        # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    # 保存配置
    config = {
        # 数据路径配置
        'data_paths': {
            'train_metadata': './dataset/synthetic_dataset_v3/train_metadata.csv',

            'semantic_tree': './semantic_treev3.json',
        },
        # 训练参数
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 30,
        'temperature': 0.07,
    }

            # 创建验证集
    test_dataset = AudioDataset(
        metadata_path=config['data_paths']['train_metadata'],
        semantic_tree_path=config['data_paths']['semantic_tree'],
        split='val',
        train_ratio=0.8,  # 使用相同的比例
        random_seed=42    # 使用相同的随机种子确保划分一致
    )
        
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    # 加载最佳事件检测模型并评估
    logger.info("Evaluating best event detection model on test set...")
    best_event_model = EventDetectionModel(num_event_categories=len(test_dataset.event_categories))
    best_event_model.load_state_dict(torch.load(os.path.join(run_dir, 'best_event_model.pth'))['model_state_dict'])
    best_event_model.to(device)
    best_event_model.eval()
    
    event_predictions = []
    for i, batch in enumerate(test_loader):
        inputs = batch['audio'].to(device)
        labels = batch['labels']
        file_paths = batch['file_path']
        with torch.no_grad():
            predictions = best_event_model(inputs)
            for j in range(len(inputs)):
                event_mask_pred = predictions['event_mask_logits'][j]>0.5
                event_category_pred_idx = torch.argmax(predictions['event_logits'][j])
                event_category_true = labels['overlay_info']['category'][j]
                event_category_pred = test_dataset.event_categories[event_category_pred_idx]
                sample_rate = 16000
                hop_length = 512
                event_start_time_pred = None
                event_end_time_pred = None
                event_start_time_true = None
                event_end_time_true = None
                for k, mask in enumerate(event_mask_pred):
                    if mask == 1:
                        if event_start_time_pred is None:
                            event_start_time_pred = k * hop_length / sample_rate
                        event_end_time_pred = k * hop_length / sample_rate
                for k, mask in enumerate(labels['event_mask'][j]):
                    if mask == 1:
                        if event_start_time_true is None:
                            event_start_time_true = k * hop_length / sample_rate
                        event_end_time_true = k * hop_length / sample_rate
                event_predictions.append({
                    'id': file_paths[j],
                    'Event Start Time Predicted': event_start_time_pred,
                    'Event End Time Predicted': event_end_time_pred,
                    'Event Start Time True': event_start_time_true,
                    'Event End Time True': event_end_time_true,
                    'Event Category Predicted': event_category_pred,
                    'Event Category True': event_category_true,
                    'event_embedding_pred': predictions['audio_embeddings'][j].cpu().numpy().flatten(),
                })
    
    # 保存预测结果为CSV文件
    print("len(event_predictions):",len(event_predictions))
    predictions_df = pd.DataFrame(event_predictions)
    predictions_df.to_csv(os.path.join(run_dir, 'event_detection_test_predictions.csv'), index=False)
    logger.info("Event detection test predictions saved to CSV.")
    
    
    # 加载最佳机器分析模型并评估
    logger.info("Evaluating best machine analysis model on test set...")
    best_machine_model = MachineAnalysisModel(
        num_device_types=len(test_dataset.device_types),
        pretrained_event_model=best_event_model
    )
    best_machine_model.load_state_dict(torch.load(os.path.join(run_dir, 'best_machine_model.pth'))['model_state_dict'])
    best_machine_model.to(device)
    best_machine_model.eval()
    
    # 评估机器分析模型
    machine_predictions = []
    for i, batch in enumerate(test_loader):
        inputs = batch['audio'].to(device)
        labels = batch['labels']
        file_paths = batch['file_path']
        with torch.no_grad():
            predictions = best_machine_model(inputs)
            for j in range(len(inputs)):
                device_type_pred_idx = torch.argmax(predictions['device_logits'][j])
                device_type_pred = test_dataset.device_types[device_type_pred_idx]
                anomaly_pred = predictions['is_normal'][j] > 0.5
                device_type_true = labels['device_type_idx'][j]
                anomaly_true = labels['is_normal'][j]
                machine_predictions.append({
                    'id': file_paths[j],
                    'Anomaly Predicted': anomaly_pred.item(),
                    'Anomaly True': anomaly_true.item()==1, 
                    'Device Type Predicted': device_type_pred,
                    'Device Type True': test_dataset.device_types[device_type_true],
                    'device_embedding_pred': predictions['device_embedding'][j].cpu().numpy().flatten(),
                    'anomaly_score': 1 - predictions['is_normal'][j].cpu().item(),
                })
    
    # 保存机器分析模型的测试结果为CSV文件
    print("len(machine_predictions):",len(machine_predictions))
    machine_predictions_df = pd.DataFrame(machine_predictions)
    machine_predictions_df.to_csv(os.path.join(run_dir, 'machine_analysis_test_predictions.csv'), index=False)
    logger.info("Machine analysis test predictions saved to CSV.")
    
    # 在保存CSV之前添加指标计算
    metrics_results = {
        'overall': {},
        'by_device': {}
    }
    
    # 准备数据用于计算指标
    y_true = np.array([not pred['Anomaly True'] for pred in machine_predictions])
    anomaly_scores = np.array([pred['anomaly_score'] for pred in machine_predictions])
    
    # 计算总体指标
    overall_auc, overall_pauc = calculate_metrics(y_true, anomaly_scores, pos_ratio=0.1)
    metrics_results['overall'] = {
        'AUC': overall_auc,
        'pAUC': overall_pauc
    }
    
    # 按设备类型计算指标
    unique_devices = set(pred['Device Type True'] for pred in machine_predictions)
    for device in unique_devices:
        device_indices = [i for i, pred in enumerate(machine_predictions) 
                         if pred['Device Type True'] == device]
        device_y_true = y_true[device_indices]
        device_scores = anomaly_scores[device_indices]
        
        device_auc, device_pauc = calculate_metrics(device_y_true, device_scores, pos_ratio=0.1)
        metrics_results['by_device'][device] = {
            'AUC': device_auc,
            'pAUC': device_pauc
        }
    
    # 保存指标结果
    with open(os.path.join(run_dir, 'anomaly_detection_metrics.json'), 'w') as f:
        json.dump(metrics_results, f, indent=4)
    
    # 记录指标结果
    logger.info("Anomaly Detection Metrics:")
    logger.info(f"Overall AUC: {metrics_results['overall']['AUC']:.4f}")
    logger.info(f"Overall pAUC: {metrics_results['overall']['pAUC']:.4f}")
    
    logger.info("\nMetrics by device type:")
    for device, metrics in metrics_results['by_device'].items():
        logger.info(f"\n{device}:")
        logger.info(f"AUC: {metrics['AUC']:.4f}")
        logger.info(f"pAUC: {metrics['pAUC']:.4f}")

    logger.info("Testing completed! All results have been saved.")


    
if __name__ == "__main__":
    main() 