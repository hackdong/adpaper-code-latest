import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(training_metrics_file, validation_metrics_file):
    # 加载训练指标
    train_df = pd.read_csv(training_metrics_file)
    
    # 加载验证指标
    val_df = pd.read_csv(validation_metrics_file)
    
    return train_df, val_df

def plot_training_progress(train_df):
    plt.figure(figsize=(15, 10))
    
    # 绘制损失
    plt.subplot(2, 2, 1)
    plt.plot(train_df['epoch'], train_df['train_loss'], label='Train Loss')
    plt.plot(train_df['epoch'], train_df['test_loss'], label='Test Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率
    plt.subplot(2, 2, 2)
    plt.plot(train_df['epoch'], train_df['train_acc'], label='Train Accuracy')
    plt.plot(train_df['epoch'], train_df['test_total_acc'], label='Test Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 绘制AUC
    plt.subplot(2, 2, 3)
    plt.plot(train_df['epoch'], train_df['train_auc'], label='Train AUC')
    plt.plot(train_df['epoch'], train_df['test_total_auc'], label='Test AUC')
    plt.title('AUC over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    # 绘制F1分数
    plt.subplot(2, 2, 4)
    plt.plot(train_df['epoch'], train_df['train_f1'], label='Train F1')
    plt.plot(train_df['epoch'], train_df['test_total_f1'], label='Test F1')
    plt.title('F1 Score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

def plot_final_metrics(val_df):
    metrics = val_df['Metric'].tolist()
    values = val_df['Value'].tolist()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=values, y=metrics)
    plt.title('Final Validation Metrics')
    plt.xlabel('Value')
    plt.tight_layout()
    plt.savefig('final_validation_metrics.png')
    plt.close()

def plot_confusion_matrix(val_df):
    # 检查是否存在混淆矩阵数据
    cm_data = val_df[val_df['Metric'].str.contains('confusion_matrix', case=False, na=False)]
    
    if cm_data.empty:
        print("No confusion matrix data found. Skipping confusion matrix plot.")
        return
    
    cm = cm_data['Value'].values.reshape(2, 2)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    training_metrics_file = 'training_metrics_20241021_081959.csv'
    validation_metrics_file = 'final_validation_metrics.csv'
    
    train_df, val_df = load_data(training_metrics_file, validation_metrics_file)
    
    plot_training_progress(train_df)
    plot_final_metrics(val_df)
    plot_confusion_matrix(val_df)
    
    print("Visualization completed. Check the output images.")

if __name__ == "__main__":
    main()
