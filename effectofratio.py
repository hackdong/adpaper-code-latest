import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_pauc(y_true, y_score, max_fpr=0.1):
    """计算pAUC分数"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # 找到最接近max_fpr的fpr索引
    last_index = np.searchsorted(fpr, max_fpr, 'right')
    # 计算部分AUC
    area = np.trapz(tpr[:last_index], fpr[:last_index])
    # 归一化
    return area / max_fpr

def main():
    # 读取数据
    metadata_df = pd.read_csv('dataset/synthetic_dataset_v3/train_metadata.csv')
    predictions_df = pd.read_csv('runs/20241212_221047/machine_analysis_test_predictions.csv')
    
    # 从文件路径中提取文件名
    predictions_df['file_name'] = predictions_df['id'].apply(lambda x: x.split('\\')[-1])
    
    # 合并两个数据框
    merged_df = pd.merge(predictions_df, metadata_df, on='file_name', how='inner')
    
    # 修正overlay_ratio: 当overlay_category为空时设置为0
    merged_df.loc[merged_df['overlay_category'].isna(), 'overlay_ratio'] = 0
    
    # 将overlay_ratio分组
    ratio_groups = {
        0.00: (merged_df['overlay_ratio'] == 0.00),
        0.75: (merged_df['overlay_ratio'] == 0.75),
        0.80: (merged_df['overlay_ratio'] == 0.80),
        0.85: (merged_df['overlay_ratio'] == 0.85),
        0.90: (merged_df['overlay_ratio'] == 0.90),
        0.95: (merged_df['overlay_ratio'] == 0.95),
        1.00: (merged_df['overlay_ratio'] == 1.00)
    }
    
    # 计算每个组的AUC和pAUC
    results = []
    for ratio, mask in ratio_groups.items():
        group_df = merged_df[mask]
        if len(group_df) > 0:  # 确保组内有数据
            auc = roc_auc_score(~group_df['is_normal'], group_df['anomaly_score'])
            pauc = calculate_pauc(~group_df['is_normal'], group_df['anomaly_score'])
            sample_count = len(group_df)
            results.append({
                'overlay_ratio': ratio,
                'AUC': auc,
                'pAUC': pauc,
                'sample_count': sample_count
            })
    
    # 转换结果为DataFrame
    results_df = pd.DataFrame(results)
    
    # 打印结果
    print("\nResults by overlay ratio:")
    print(results_df.to_string(index=False))
    
    # 可视化
    plt.figure(figsize=(12, 6))
    
    # 创建双轴图
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # 绘制AUC和pAUC曲线
    ax1.plot(results_df['overlay_ratio'], results_df['AUC'], 'b-', label='AUC', marker='o')
    ax1.plot(results_df['overlay_ratio'], results_df['pAUC'], 'r-', label='pAUC', marker='s')
    
    # 绘制样本数量条形图
    ax2.bar(results_df['overlay_ratio'], results_df['sample_count'], alpha=0.2, color='gray', label='Sample Count')
    
    # 设置标签和标题
    ax1.set_xlabel('Overlay Ratio')
    ax1.set_ylabel('Score')
    ax2.set_ylabel('Sample Count')
    plt.title('AUC and pAUC Scores by Overlay Ratio')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 保存图表
    plt.savefig('overlay_ratio_analysis.png')
    plt.close()

if __name__ == "__main__":
    main()