import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Use a built-in style similar to seaborn
plt.style.use('seaborn-v0_8-darkgrid')  # For newer versions of matplotlib
# plt.style.use('seaborn-darkgrid')  # For older versions of matplotlib

sns.set_palette("deep")

def load_metadata(file_path):
    return pd.read_csv(file_path)

def plot_device_distribution(df, dataset_type):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='device_type', data=df)
    plt.title(f'Distribution of Device Types in {dataset_type.capitalize()} Dataset')
    plt.xlabel('Device Type')
    plt.ylabel('Count')
    plt.savefig(f'{dataset_type}_device_distribution.png')
    plt.close()

def plot_normal_anomaly_distribution(df, dataset_type):
    plt.figure(figsize=(8, 6))
    df['is_normal'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title(f'Normal vs Anomaly Distribution in {dataset_type.capitalize()} Dataset')
    plt.ylabel('')
    plt.savefig(f'{dataset_type}_normal_anomaly_distribution.png')
    plt.close()

def plot_overlay_category_distribution(df, dataset_type):
    plt.figure(figsize=(12, 6))
    sns.countplot(y='overlay_category', data=df)
    plt.title(f'Distribution of Overlay Categories in {dataset_type.capitalize()} Dataset')
    plt.xlabel('Count')
    plt.ylabel('Overlay Category')
    plt.savefig(f'{dataset_type}_overlay_category_distribution.png')
    plt.close()

def plot_overlay_ratio_distribution(df, dataset_type):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='overlay_ratio', bins=20, kde=True)
    plt.title(f'Distribution of Overlay Ratios in {dataset_type.capitalize()} Dataset')
    plt.xlabel('Overlay Ratio')
    plt.ylabel('Count')
    plt.savefig(f'{dataset_type}_overlay_ratio_distribution.png')
    plt.close()

def plot_device_normal_anomaly_distribution(df, dataset_type):
    plt.figure(figsize=(12, 6))
    sns.countplot(x='device_type', hue='is_normal', data=df)
    plt.title(f'Normal vs Anomaly Distribution by Device Type in {dataset_type.capitalize()} Dataset')
    plt.xlabel('Device Type')
    plt.ylabel('Count')
    plt.legend(title='Is Normal', labels=['Anomaly', 'Normal'])
    plt.savefig(f'{dataset_type}_device_normal_anomaly_distribution.png')
    plt.close()

def visualize_dataset_stats(metadata_file, dataset_type):
    # 创建输出文件夹
    output_dir = 'dataset_visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_metadata(metadata_file)
    
    # 修改所有的 plot 函数调用，将输出路径改为新文件夹
    plot_device_distribution(df, f'{output_dir}/{dataset_type}')
    plot_normal_anomaly_distribution(df, f'{output_dir}/{dataset_type}')
    plot_overlay_category_distribution(df, f'{output_dir}/{dataset_type}')
    plot_overlay_ratio_distribution(df, f'{output_dir}/{dataset_type}')
    plot_device_normal_anomaly_distribution(df, f'{output_dir}/{dataset_type}')

if __name__ == "__main__":
    train_metadata_file = 'dataset/synthetic_dataset_v3/train_metadata.csv'
    validation_metadata_file = 'dataset/synthetic_validation_dataset_v3/validation_metadata.csv'

    if os.path.exists(train_metadata_file):
        visualize_dataset_stats(train_metadata_file, 'train')
    else:
        print(f"Train metadata file not found: {train_metadata_file}")

    if os.path.exists(validation_metadata_file):
        visualize_dataset_stats(validation_metadata_file, 'validation')
    else:
        print(f"Validation metadata file not found: {validation_metadata_file}")

    print("Visualization complete. Check the 'visualization_results' directory for the generated plots.")
