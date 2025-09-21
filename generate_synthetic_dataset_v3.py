import os
import random
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import time


random.seed(42)
np.random.seed(42)


MIMII_PATH = 'dataset/MIMII'
URBAN_SOUND_PATH = 'dataset/UrbanSound8K'
ESC_50_PATH = 'dataset/ESC-50-master'
OUTPUT_PATH = 'dataset/synthetic_dataset_v3'
VALIDATION_OUTPUT_PATH = 'dataset/synthetic_validation_dataset_v3'


os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(VALIDATION_OUTPUT_PATH, exist_ok=True)


urban_sound_metadata = pd.read_csv(os.path.join(URBAN_SOUND_PATH, 'UrbanSound8K.csv'))
esc_50_metadata = pd.read_csv(os.path.join(ESC_50_PATH, 'meta', 'esc50.csv'))

def load_mimii_audio_files(device_type, dataset_type='train'):
    normal_audio_files = []
    anomaly_audio_files = []
    if dataset_type == 'train':
        paths = [os.path.join(MIMII_PATH, device_type, 'train'),
                 os.path.join(MIMII_PATH, device_type, 'source_test')]
    elif dataset_type == 'validation':
        paths = [os.path.join(MIMII_PATH, device_type, 'target_test')]
    
    for path in paths:
        for file in os.listdir(path):
            if file.endswith('.wav'):
                full_path = os.path.join(path, file)
                is_normal = 'normal' in file
                product_id = file.split('_')[1]
                if is_normal:
                    normal_audio_files.append((full_path, is_normal, device_type, product_id))
                else:
                    anomaly_audio_files.append((full_path, is_normal, device_type, product_id))
    
    return normal_audio_files, anomaly_audio_files

def load_urban_sound_file():
    file = urban_sound_metadata.sample(1).iloc[0]
    fold = file['fold']
    filename = file['slice_file_name']
    category = file['class']
    full_path = os.path.join(URBAN_SOUND_PATH, f'fold{fold}', filename)
    return full_path, category

def load_esc_50_file():
    file = esc_50_metadata.sample(1).iloc[0]
    filename = file['filename']
    category = file['category']
    full_path = os.path.join(ESC_50_PATH, 'audio', filename)
    return full_path, category

def mix_audio(base_audio, overlay_audio, start_time, overlay_ratio):

    max_start = len(base_audio) - len(overlay_audio)
    start_sample = min(int(start_time * 16000), max_start)
    

    overlay_audio = overlay_audio * overlay_ratio
    

    mixed_audio = base_audio.copy()
    end_sample = start_sample + len(overlay_audio)
    mixed_audio[start_sample:end_sample] += overlay_audio
    
    return mixed_audio

def generate_synthetic_sample(mimii_file, is_normal, device_type, product_id, overlay_source):

    base_audio, _ = librosa.load(mimii_file, sr=16000, mono=True)
    

    if random.random() < 0.5:  
        if overlay_source == 'urban':
            overlay_file, overlay_category = load_urban_sound_file()
        else:  # ESC-50
            overlay_file, overlay_category = load_esc_50_file()
        
        overlay_audio, _ = librosa.load(overlay_file, sr=16000, mono=True)
        

        start_time = random.uniform(0, len(base_audio) / 16000 - len(overlay_audio) / 16000)
        

        overlay_ratio = random.choice([0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
        

        mixed_audio = mix_audio(base_audio, overlay_audio, start_time, overlay_ratio)
        
        end_time = start_time + len(overlay_audio) / 16000
    else:
        mixed_audio = base_audio
        overlay_category = None
        start_time = None
        end_time = None
        overlay_ratio = 1.0  
    
    return mixed_audio, is_normal, device_type, product_id, overlay_category, start_time, end_time, overlay_ratio

def generate_dataset(num_samples, dataset_type='train'):
    metadata = []
    device_types = ['fan', 'gearbox', 'pump', 'slider', 'valve']
    

    all_normal_files = []
    all_anomaly_files = []
    for device in device_types:
        normal, anomaly = load_mimii_audio_files(device, dataset_type)
        all_normal_files.extend(normal)
        all_anomaly_files.extend(anomaly)
    
    output_path = OUTPUT_PATH if dataset_type == 'train' else VALIDATION_OUTPUT_PATH
    
    start_timeT = time.time()
    

    num_anomaly_samples = len(all_anomaly_files)
    num_normal_samples = num_samples - num_anomaly_samples
    
    with tqdm(total=num_samples, desc=f"Generating {dataset_type} samples") as pbar:

        for i, (mimii_file, is_normal, device_type, product_id) in enumerate(all_anomaly_files):
            overlay_source = random.choice(['urban', 'esc'])
            mixed_audio, is_normal, device_type, product_id, overlay_category, start_time, end_time, overlay_ratio = generate_synthetic_sample(
                mimii_file, is_normal, device_type, product_id, overlay_source)
            
            file_name = f"synthetic_{dataset_type}_{i:04d}_{device_type}_{product_id}_anomaly.wav"
            output_file = os.path.join(output_path, file_name)
            sf.write(output_file, mixed_audio, 16000)
            
            metadata.append({
                'file_name': file_name,
                'is_normal': is_normal,
                'device_type': device_type,
                'product_id': product_id,
                'overlay_category': overlay_category,
                'overlay_start_time': start_time,
                'overlay_end_time': end_time,
                'overlay_ratio': overlay_ratio
            })
            
            pbar.update(1)
        

        for i in range(num_normal_samples):
            mimii_file, is_normal, device_type, product_id = random.choice(all_normal_files)
            overlay_source = random.choice(['urban', 'esc'])
            mixed_audio, is_normal, device_type, product_id, overlay_category, start_time, end_time, overlay_ratio = generate_synthetic_sample(
                mimii_file, is_normal, device_type, product_id, overlay_source)
            
            file_name = f"synthetic_{dataset_type}_{i+num_anomaly_samples:04d}_{device_type}_{product_id}_normal.wav"
            output_file = os.path.join(output_path, file_name)
            sf.write(output_file, mixed_audio, 16000)
            
            metadata.append({
                'file_name': file_name,
                'is_normal': is_normal,
                'device_type': device_type,
                'product_id': product_id,
                'overlay_category': overlay_category,
                'overlay_start_time': start_time,
                'overlay_end_time': end_time,
                'overlay_ratio': overlay_ratio
            })
            
            pbar.update(1)
            

            elapsed_time = time.time() - start_timeT
            samples_per_second = (i + 1 + num_anomaly_samples) / elapsed_time
            remaining_samples = num_samples - (i + 1 + num_anomaly_samples)
            estimated_time_remaining = remaining_samples / samples_per_second
            

            pbar.set_description(f"Generating {dataset_type} samples - ETA: {estimated_time_remaining:.2f}s")
    

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_path, f'{dataset_type}_metadata.csv'), index=False)

if __name__ == "__main__":
    generate_dataset(30000, 'train')
    generate_dataset(3000, 'validation')
    print("Dataset generation complete.")
