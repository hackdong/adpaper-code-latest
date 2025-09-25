import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
import ast
import re

class TestResultVisualizer:
    def __init__(self, result_dir, metadata_path):

        self.result_dir = Path(result_dir)
        self.metadata_df = pd.read_csv(metadata_path)
        

        self.event_results = pd.read_csv(self.result_dir / 'event_detection_test_predictions.csv')
        self.machine_results = pd.read_csv(self.result_dir / 'machine_analysis_test_predictions.csv')
        

        self._add_ratio_info()
        
    def _add_ratio_info(self):

        def extract_filename(path):
            return Path(path).name
            
        self.event_results['filename'] = self.event_results['id'].apply(extract_filename)
        self.machine_results['filename'] = self.machine_results['id'].apply(extract_filename)
        

        filename_to_ratio = dict(zip(
            self.metadata_df['file_name'], 
            self.metadata_df['overlay_ratio']
        ))
        
        self.event_results['overlay_ratio'] = self.event_results['filename'].map(filename_to_ratio)
        self.machine_results['overlay_ratio'] = self.machine_results['filename'].map(filename_to_ratio)
        
    def analyze_event_detection(self):

        time_accuracy = []
        for _, row in self.event_results.iterrows():
            if pd.notna(row['Event Start Time True']):  
                pred_start = row['Event Start Time Predicted']
                pred_end = row['Event End Time Predicted']
                true_start = row['Event Start Time True']
                true_end = row['Event End Time True']
                
                if pd.notna(pred_start) and pd.notna(pred_end):
                    intersection_start = max(pred_start, true_start)
                    intersection_end = min(pred_end, true_end)
                    intersection_length = max(0, intersection_end - intersection_start)
                    union_length = (pred_end - pred_start) + (true_end - true_start) - intersection_length
                    if union_length == 0 or pd.isna(true_start) or pd.isna(true_end): 
                        time_accuracy.append(1)
                    else:
                        time_accuracy.append(intersection_length / union_length)
        

        category_correct = (
            self.event_results['Event Category Predicted'] == 
            self.event_results['Event Category True']
        )
        

        ratio_bins = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        ratio_labels = ['rt=0.75', 'rt=0.8', 'rt=0.85', 'rt=0.9', 'rt=0.95', 'rt=1.0']
        self.event_results['ratio_group'] = pd.cut(
            self.event_results['overlay_ratio'],
            bins=ratio_bins,
            labels=ratio_labels
        )
        
        ratio_accuracy = self.event_results.groupby('ratio_group').apply(
            lambda x: (x['Event Category Predicted'] == x['Event Category True']).mean()
        )
        

        plt.figure(figsize=(15, 5))
        

        plt.subplot(131)
        plt.hist(time_accuracy, bins=30)
        plt.title('Time Prediction Accuracy Distribution')
        plt.xlabel('Average Time Accuracy')
        plt.ylabel('Count')
        

        plt.subplot(132)
        plt.bar(['Overall'], [category_correct.mean()])
        plt.title('Event Category Prediction Accuracy')
        plt.ylabel('Accuracy')
        

        plt.subplot(133)
        ratio_accuracy.plot(kind='bar')
        plt.title('Accuracy by Overlay Ratio')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.result_dir / 'event_detection_analysis.png')
        plt.close()
        
    def analyze_machine_analysis(self):

        anomaly_accuracy = (
            self.machine_results['Anomaly Predicted'] == 
            self.machine_results['Anomaly True']
        ).mean()
        

        device_accuracy = (
            self.machine_results['Device Type Predicted'] == 
            self.machine_results['Device Type True']
        ).mean()
        

        ratio_bins = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        ratio_labels = ['rt=0.75', 'rt=0.8', 'rt=0.85', 'rt=0.9', 'rt=0.95', 'rt=1.0']
        self.machine_results['ratio_group'] = pd.cut(
            self.machine_results['overlay_ratio'],
            bins=ratio_bins,
            labels=ratio_labels
        )
        
        ratio_analysis = self.machine_results.groupby('ratio_group').apply(
            lambda x: pd.Series({
                'anomaly_accuracy': (x['Anomaly Predicted'] == x['Anomaly True']).mean(),
                'device_accuracy': (x['Device Type Predicted'] == x['Device Type True']).mean()
            })
        )
        
 
        plt.figure(figsize=(15, 5))
        

        plt.subplot(131)
        plt.bar(['Anomaly Detection', 'Device Classification'], 
                [anomaly_accuracy, device_accuracy])
        plt.title('Overall Accuracy')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        

        plt.subplot(132)
        ratio_analysis['anomaly_accuracy'].plot(kind='bar')
        plt.title('Anomaly Detection Accuracy by Ratio')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        

        plt.subplot(133)
        ratio_analysis['device_accuracy'].plot(kind='bar')
        plt.title('Device Classification Accuracy by Ratio')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.result_dir / 'machine_analysis.png')
        plt.close()
        
    def visualize_embeddings(self):

        def process_embedding(embedding_str):

            try:

                numbers = embedding_str.strip('[]').split()

                numbers = [float(n) for n in numbers]
                return np.array(numbers)
            except Exception as e:
                print(f"Error processing embedding: {e}")
                print(f"Problematic string: {embedding_str}")
                return None
        

        event_embeddings = []
        for emb in self.event_results['event_embedding_pred']:
            processed = process_embedding(emb)
            if processed is not None:
                event_embeddings.append(processed)
        event_embeddings = np.stack(event_embeddings)
        event_labels = self.event_results['Event Category True'][:len(event_embeddings)]
        

        machine_embeddings = []
        for emb in self.machine_results['device_embedding_pred']:
            processed = process_embedding(emb)
            if processed is not None:
                machine_embeddings.append(processed)
        machine_embeddings = np.stack(machine_embeddings)
        machine_labels = self.machine_results['Device Type True'][:len(machine_embeddings)]
        

        tsne = TSNE(n_components=2, random_state=42)
        event_tsne = tsne.fit_transform(event_embeddings)
        machine_tsne = tsne.fit_transform(machine_embeddings)
        

        plt.figure(figsize=(12, 5))
        
        plt.subplot(121)
        unique_events = event_labels.unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_events)))
        for event, color in zip(unique_events, colors):
            mask = event_labels == event
            plt.scatter(
                event_tsne[mask, 0], 
                event_tsne[mask, 1],
                c=[color],
                label=event,
                alpha=0.6
            )
        plt.title('Event Embeddings')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        

        plt.subplot(122)
        unique_devices = machine_labels.unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_devices)))
        for device, color in zip(unique_devices, colors):
            mask = machine_labels == device
            plt.scatter(
                machine_tsne[mask, 0],
                machine_tsne[mask, 1],
                c=[color],
                label=device,
                alpha=0.6
            )
        plt.title('Device Embeddings')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        plt.tight_layout()
        plt.savefig(self.result_dir / 'embeddings_visualization.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
    def save_results_to_csv(self):

        event_results_df = pd.DataFrame()
        event_results_df['audio_name'] = self.event_results['filename']
        

        time_accuracies = []
        for _, row in self.event_results.iterrows():
            if pd.notna(row['Event Start Time True']):
                pred_start = row['Event Start Time Predicted']
                pred_end = row['Event End Time Predicted']
                true_start = row['Event Start Time True']
                true_end = row['Event End Time True']
                
                if pd.notna(pred_start) and pd.notna(pred_end):
                    intersection_start = max(pred_start, true_start)
                    intersection_end = min(pred_end, true_end)
                    intersection_length = max(0, intersection_end - intersection_start)
                    union_length = (pred_end - pred_start) + (true_end - true_start) - intersection_length
                    if union_length == 0 or pd.isna(true_start) or pd.isna(true_end):
                        time_accuracies.append(1.0)
                    else:
                        time_accuracies.append(intersection_length / union_length)
                else:
                    time_accuracies.append(0.0)
            else:
                time_accuracies.append(1.0 if pd.isna(row['Event Start Time Predicted']) else 0.0)
        
        event_results_df['time_accuracy'] = time_accuracies
        event_results_df['category_accuracy'] = (
            self.event_results['Event Category Predicted'] == 
            self.event_results['Event Category True']
        ).astype(int)
        event_results_df['event_category'] = self.event_results['Event Category True']
        event_results_df['overlay_ratio'] = self.event_results['overlay_ratio']
        

        event_results_df.to_csv(self.result_dir / 'eventresult.csv', index=False)

        machine_results_df = pd.DataFrame()
        machine_results_df['audio_name'] = self.machine_results['filename']
        machine_results_df['anomaly_accuracy'] = (
            self.machine_results['Anomaly Predicted'] == 
            self.machine_results['Anomaly True']
        ).astype(int)
        machine_results_df['device_accuracy'] = (
            self.machine_results['Device Type Predicted'] == 
            self.machine_results['Device Type True']
        ).astype(int)
        machine_results_df['device_type'] = self.machine_results['Device Type True']
        machine_results_df['overlay_ratio'] = self.machine_results['overlay_ratio']
        

        machine_results_df.to_csv(self.result_dir / 'machineresult.csv', index=False)

def main():

    result_dir = "./runs/20241212_221047"
    metadata_path = "./dataset/synthetic_dataset_v3/train_metadata.csv"
    

    visualizer = TestResultVisualizer(result_dir, metadata_path)
    

    visualizer.analyze_event_detection()
    visualizer.analyze_machine_analysis()
    visualizer.visualize_embeddings()
    

    visualizer.save_results_to_csv()
    
    print("Analysis completed! Results saved to:", result_dir)

if __name__ == "__main__":
    main() 