import os
import random
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import time
from collections import defaultdict

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# dataset paths
MIMII_PATH = 'dataset/MIMII'
URBAN_SOUND_PATH = 'dataset/UrbanSound8K'
ESC_50_PATH = 'dataset/ESC-50-master'
OUTPUT_BASE_PATH = 'dataset/synthetic_dataset'

# make sure the output directory exists
os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

# read UrbanSound8K and ESC-50 metadata
urban_sound_metadata = pd.read_csv(os.path.join(URBAN_SOUND_PATH, 'UrbanSound8K.csv'))
esc_50_metadata = pd.read_csv(os.path.join(ESC_50_PATH, 'meta', 'esc50.csv'))

def load_all_mimii_audio_files():
    """Load all MIMII audio files, organized by device type and product ID"""
    all_files = defaultdict(lambda: defaultdict(list))

    device_types = ['fan', 'gearbox', 'pump', 'slider', 'valve']

    for device_type in device_types:
        # load all paths
        paths = [
            os.path.join(MIMII_PATH, device_type, 'train'),
            os.path.join(MIMII_PATH, device_type, 'source_test'),
            os.path.join(MIMII_PATH, device_type, 'target_test')
        ]

        for path in paths:
            if not os.path.exists(path):
                continue

            for file in os.listdir(path):
                if file.endswith('.wav'):
                    full_path = os.path.join(path, file)
                    is_normal = 'normal' in file
                    product_id = file.split('_')[1]

                    all_files[device_type][product_id].append({
                        'file_path': full_path,
                        'is_normal': is_normal,
                        'device_type': device_type,
                        'product_id': product_id
                    })

    return all_files

def get_overlay_categories():
    """Get all available interference categories"""
    urban_categories = set(urban_sound_metadata['class'].unique())
    esc_categories = set(esc_50_metadata['category'].unique())
    return {
        'urban': list(urban_categories),
        'esc': list(esc_categories)
    }

def create_10fold_splits(all_mimii_files):
    """Create 10-fold cross-validation splits, ensuring that the same product ID does not span train/test"""
    folds = []

    for fold_idx in range(10):
        train_files = []
        test_files = []

        for device_type, product_ids in all_mimii_files.items():
            # convert product IDs to a list and shuffle
            product_id_list = list(product_ids.keys())
            random.shuffle(product_id_list)

            # 90% for training, 10% for testing
            split_idx = int(len(product_id_list) * 0.9)
            train_product_ids = product_id_list[:split_idx]
            test_product_ids = product_id_list[split_idx:]

            # collect training files
            for pid in train_product_ids:
                train_files.extend(product_ids[pid])

            # collect test files
            for pid in test_product_ids:
                test_files.extend(product_ids[pid])

        folds.append({
            'fold_idx': fold_idx,
            'train_files': train_files,
            'test_files': test_files
        })

    return folds

def load_overlay_file(source, category=None):
    """Load interference audio files of the specified category"""
    if source == 'urban':
        if category:
            candidates = urban_sound_metadata[urban_sound_metadata['class'] == category]
        else:
            candidates = urban_sound_metadata

        if len(candidates) == 0:
            return None, None

        file = candidates.sample(1).iloc[0]
        fold = file['fold']
        filename = file['slice_file_name']
        category = file['class']
        full_path = os.path.join(URBAN_SOUND_PATH, f'fold{fold}', filename)
        return full_path, category

    elif source == 'esc':
        if category:
            candidates = esc_50_metadata[esc_50_metadata['category'] == category]
        else:
            candidates = esc_50_metadata

        if len(candidates) == 0:
            return None, None

        file = candidates.sample(1).iloc[0]
        filename = file['filename']
        category = file['category']
        full_path = os.path.join(ESC_50_PATH, 'audio', filename)
        return full_path, category

    return None, None

def mix_audio(base_audio, overlay_audio, start_time, overlay_ratio):
    """Mix the base audio and the interference audio"""
    # ensure the start time does not make the overlay exceed the base audio length
    max_start = len(base_audio) - len(overlay_audio)
    start_sample = min(int(start_time * 16000), max_start) if max_start > 0 else 0

    # scale the overlay volume to a given ratio of the base audio
    overlay_audio = overlay_audio * overlay_ratio

    # create the mixed audio
    mixed_audio = base_audio.copy()
    end_sample = start_sample + len(overlay_audio)
    if end_sample <= len(mixed_audio):
        mixed_audio[start_sample:end_sample] += overlay_audio

    return mixed_audio

def generate_synthetic_sample(mimii_file_info, overlay_source, overlay_ratio, overlay_category=None):
    """Generate one synthetic sample"""
    # load MIMII audio
    base_audio, _ = librosa.load(mimii_file_info['file_path'], sr=16000, mono=True)

    # decide whether to overlay interference
    do_overlay = random.random() < 0.5  # 50% probability of overlaying

    if do_overlay:
        overlay_file, actual_category = load_overlay_file(overlay_source, overlay_category)

        if overlay_file and os.path.exists(overlay_file):
            overlay_audio, _ = librosa.load(overlay_file, sr=16000, mono=True)

            # randomly choose the overlay start time
            max_start_time = max(0, len(base_audio) / 16000 - len(overlay_audio) / 16000)
            start_time = random.uniform(0, max_start_time)

            # mix the audio
            mixed_audio = mix_audio(base_audio, overlay_audio, start_time, overlay_ratio)

            end_time = start_time + len(overlay_audio) / 16000
            overlay_category = actual_category
        else:
            # if the interference file cannot be loaded, skip overlaying
            mixed_audio = base_audio
            start_time = None
            end_time = None
            overlay_category = None
            overlay_ratio = 1.0
    else:
        mixed_audio = base_audio
        start_time = None
        end_time = None
        overlay_category = None
        overlay_ratio = 1.0

    return (mixed_audio, mimii_file_info['is_normal'], mimii_file_info['device_type'],
            mimii_file_info['product_id'], overlay_category, start_time, end_time, overlay_ratio)

def generate_fold_dataset(fold_idx, fold_data, train_samples=2970, test_samples=330):
    """Generate the dataset for one fold"""
    print(f"\nGenerating fold {fold_idx + 1}/10...")

    fold_base_path = os.path.join(OUTPUT_BASE_PATH, f'fold_{fold_idx}')
    train_path = os.path.join(fold_base_path, 'train')
    test_path = os.path.join(fold_base_path, 'test')

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # pre-sample the training set first to determine the seen interference categories
    print("Pre-sampling training set to determine seen overlay categories...")
    train_overlay_categories = set()

    # get all available categories
    all_categories = get_overlay_categories()

    # randomly select half of the urban/esc categories as seen, ensuring unseen ones remain
    seen_urban = set(random.sample(all_categories['urban'], len(all_categories['urban']) // 2))
    seen_esc = set(random.sample(all_categories['esc'], len(all_categories['esc']) // 2))

    # sample some training files to fix the seen categories, using only seen categories
    sample_size = min(5000, len(fold_data['train_files']))
    sampled_train_files = random.sample(fold_data['train_files'], sample_size)

    for file_info in sampled_train_files:
        if random.random() < 0.5:  # 50% probability of overlaying
            overlay_source = random.choice(['urban', 'esc'])
            if overlay_source == 'urban' and seen_urban:
                overlay_category = random.choice(list(seen_urban))
                overlay_file, _ = load_overlay_file(overlay_source, overlay_category)
                if overlay_file:
                    train_overlay_categories.add((overlay_source, overlay_category))
            elif overlay_source == 'esc' and seen_esc:
                overlay_category = random.choice(list(seen_esc))
                overlay_file, _ = load_overlay_file(overlay_source, overlay_category)
                if overlay_file:
                    train_overlay_categories.add((overlay_source, overlay_category))

    # compute the unseen categories
    unseen_urban = [cat for cat in all_categories['urban'] if cat not in seen_urban]
    unseen_esc = [cat for cat in all_categories['esc'] if cat not in seen_esc]

    print(f"Fold {fold_idx + 1}: {len(unseen_urban)} unseen urban categories, {len(unseen_esc)} unseen ESC categories")

    # generate training samples
    train_metadata = []
    print("Generating training samples...")

    with tqdm(total=train_samples, desc=f"Fold {fold_idx + 1} Training") as pbar:
        for i in range(train_samples):
            mimii_file = random.choice(fold_data['train_files'])
            overlay_source = random.choice(['urban', 'esc'])
            overlay_ratio = random.choice([0.5, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])  # 0.5 included

            # restrict overlay categories to the seen ones
            if overlay_source == 'urban' and seen_urban:
                overlay_category = random.choice(list(seen_urban))
            elif overlay_source == 'esc' and seen_esc:
                overlay_category = random.choice(list(seen_esc))
            else:
                overlay_category = None

            mixed_audio, is_normal, device_type, product_id, overlay_category, start_time, end_time, overlay_ratio = generate_synthetic_sample(
                mimii_file, overlay_source, overlay_ratio, overlay_category)

            file_name = f"synthetic_fold{fold_idx}_train_{i:05d}_{device_type}_{product_id}_{'normal' if is_normal else 'anomaly'}.wav"
            output_file = os.path.join(train_path, file_name)
            sf.write(output_file, mixed_audio, 16000)

            # determine whether this is a zero-shot interference category
            is_zero_shot_overlay = (overlay_category is not None and
                                  ((overlay_source == 'urban' and overlay_category in unseen_urban) or
                                   (overlay_source == 'esc' and overlay_category in unseen_esc)))

            train_metadata.append({
                'file_name': file_name,
                'is_normal': is_normal,
                'device_type': device_type,
                'product_id': product_id,
                'overlay_category': overlay_category,
                'overlay_source': overlay_source if overlay_category else None,
                'overlay_start_time': start_time,
                'overlay_end_time': end_time,
                'overlay_ratio': overlay_ratio,
                'is_zero_shot_overlay': is_zero_shot_overlay
            })

            pbar.update(1)

    # generate test samples - must contain a fixed proportion of clean and unseen-category samples
    test_metadata = []
    print("Generating test samples...")

    # compute the test sample distribution
    num_mimii_only = test_samples // 10  # 1/10 clean MIMII without interference
    num_unseen_urban = test_samples // 10  # 1/10 unseen Urban categories
    num_unseen_esc = test_samples // 10  # 1/10 unseen ESC categories
    num_regular = test_samples - num_mimii_only - num_unseen_urban - num_unseen_esc  # remaining regular samples

    print(f"Test distribution: {num_mimii_only} MIMII-only, {num_unseen_urban} unseen urban, {num_unseen_esc} unseen ESC, {num_regular} regular")

    with tqdm(total=test_samples, desc=f"Fold {fold_idx + 1} Test") as pbar:
        sample_idx = 0

        # generate clean MIMII samples without interference
        for i in range(num_mimii_only):
            mimii_file = random.choice(fold_data['test_files'])
            mixed_audio, _ = librosa.load(mimii_file['file_path'], sr=16000, mono=True)

            file_name = f"synthetic_fold{fold_idx}_test_{sample_idx:05d}_{mimii_file['device_type']}_{mimii_file['product_id']}_{'normal' if mimii_file['is_normal'] else 'anomaly'}_mimii_only.wav"
            output_file = os.path.join(test_path, file_name)
            sf.write(output_file, mixed_audio, 16000)

            test_metadata.append({
                'file_name': file_name,
                'is_normal': mimii_file['is_normal'],
                'device_type': mimii_file['device_type'],
                'product_id': mimii_file['product_id'],
                'overlay_category': None,
                'overlay_source': None,
                'overlay_start_time': None,
                'overlay_end_time': None,
                'overlay_ratio': 1.0,
                'is_zero_shot_overlay': False
            })

            sample_idx += 1
            pbar.update(1)

        # generate samples with unseen Urban categories
        for i in range(num_unseen_urban):
            mimii_file = random.choice(fold_data['test_files'])
            if unseen_urban:
                overlay_category = random.choice(unseen_urban)
                overlay_ratio = random.choice([0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

                mixed_audio, is_normal, device_type, product_id, _, start_time, end_time, overlay_ratio = generate_synthetic_sample(
                    mimii_file, 'urban', overlay_ratio, overlay_category)

                file_name = f"synthetic_fold{fold_idx}_test_{sample_idx:05d}_{device_type}_{product_id}_{'normal' if is_normal else 'anomaly'}_unseen_urban.wav"
                output_file = os.path.join(test_path, file_name)
                sf.write(output_file, mixed_audio, 16000)

                test_metadata.append({
                    'file_name': file_name,
                    'is_normal': is_normal,
                    'device_type': device_type,
                    'product_id': product_id,
                    'overlay_category': overlay_category,
                    'overlay_source': 'urban',
                    'overlay_start_time': start_time,
                    'overlay_end_time': end_time,
                    'overlay_ratio': overlay_ratio,
                    'is_zero_shot_overlay': True
                })
            else:
                # if no unseen urban category is available, fall back to a regular sample
                overlay_source = random.choice(['urban', 'esc'])
                overlay_ratio = random.choice([0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

                mixed_audio, is_normal, device_type, product_id, overlay_category, start_time, end_time, overlay_ratio = generate_synthetic_sample(
                    mimii_file, overlay_source, overlay_ratio)

                file_name = f"synthetic_fold{fold_idx}_test_{sample_idx:05d}_{device_type}_{product_id}_{'normal' if is_normal else 'anomaly'}_regular.wav"
                output_file = os.path.join(test_path, file_name)
                sf.write(output_file, mixed_audio, 16000)

                test_metadata.append({
                    'file_name': file_name,
                    'is_normal': is_normal,
                    'device_type': device_type,
                    'product_id': product_id,
                    'overlay_category': overlay_category,
                    'overlay_source': overlay_source if overlay_category else None,
                    'overlay_start_time': start_time,
                    'overlay_end_time': end_time,
                    'overlay_ratio': overlay_ratio,
                    'is_zero_shot_overlay': False
                })

            sample_idx += 1
            pbar.update(1)

        # generate samples with unseen ESC categories
        for i in range(num_unseen_esc):
            mimii_file = random.choice(fold_data['test_files'])
            if unseen_esc:
                overlay_category = random.choice(unseen_esc)
                overlay_ratio = random.choice([0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

                mixed_audio, is_normal, device_type, product_id, _, start_time, end_time, overlay_ratio = generate_synthetic_sample(
                    mimii_file, 'esc', overlay_ratio, overlay_category)

                file_name = f"synthetic_fold{fold_idx}_test_{sample_idx:05d}_{device_type}_{product_id}_{'normal' if is_normal else 'anomaly'}_unseen_esc.wav"
                output_file = os.path.join(test_path, file_name)
                sf.write(output_file, mixed_audio, 16000)

                test_metadata.append({
                    'file_name': file_name,
                    'is_normal': is_normal,
                    'device_type': device_type,
                    'product_id': product_id,
                    'overlay_category': overlay_category,
                    'overlay_source': 'esc',
                    'overlay_start_time': start_time,
                    'overlay_end_time': end_time,
                    'overlay_ratio': overlay_ratio,
                    'is_zero_shot_overlay': True
                })
            else:
                # if no unseen esc category is available, fall back to a regular sample
                overlay_source = random.choice(['urban', 'esc'])
                overlay_ratio = random.choice([0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

                mixed_audio, is_normal, device_type, product_id, overlay_category, start_time, end_time, overlay_ratio = generate_synthetic_sample(
                    mimii_file, overlay_source, overlay_ratio)

                file_name = f"synthetic_fold{fold_idx}_test_{sample_idx:05d}_{device_type}_{product_id}_{'normal' if is_normal else 'anomaly'}_regular.wav"
                output_file = os.path.join(test_path, file_name)
                sf.write(output_file, mixed_audio, 16000)

                test_metadata.append({
                    'file_name': file_name,
                    'is_normal': is_normal,
                    'device_type': device_type,
                    'product_id': product_id,
                    'overlay_category': overlay_category,
                    'overlay_source': overlay_source if overlay_category else None,
                    'overlay_start_time': start_time,
                    'overlay_end_time': end_time,
                    'overlay_ratio': overlay_ratio,
                    'is_zero_shot_overlay': False
                })

            sample_idx += 1
            pbar.update(1)

        # generate regular test samples
        for i in range(num_regular):
            mimii_file = random.choice(fold_data['test_files'])
            overlay_source = random.choice(['urban', 'esc'])
            overlay_ratio = random.choice([0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

            mixed_audio, is_normal, device_type, product_id, overlay_category, start_time, end_time, overlay_ratio = generate_synthetic_sample(
                mimii_file, overlay_source, overlay_ratio)

            file_name = f"synthetic_fold{fold_idx}_test_{sample_idx:05d}_{device_type}_{product_id}_{'normal' if is_normal else 'anomaly'}_regular.wav"
            output_file = os.path.join(test_path, file_name)
            sf.write(output_file, mixed_audio, 16000)

            # check whether it is zero-shot
            is_zero_shot = (overlay_category is not None and
                          ((overlay_source == 'urban' and overlay_category in unseen_urban) or
                           (overlay_source == 'esc' and overlay_category in unseen_esc)))

            test_metadata.append({
                'file_name': file_name,
                'is_normal': is_normal,
                'device_type': device_type,
                'product_id': product_id,
                'overlay_category': overlay_category,
                'overlay_source': overlay_source if overlay_category else None,
                'overlay_start_time': start_time,
                'overlay_end_time': end_time,
                'overlay_ratio': overlay_ratio,
                'is_zero_shot_overlay': is_zero_shot
            })

            sample_idx += 1
            pbar.update(1)

    # save metadata
    train_df = pd.DataFrame(train_metadata)
    test_df = pd.DataFrame(test_metadata)

    train_df.to_csv(os.path.join(fold_base_path, 'train_metadata.csv'), index=False)
    test_df.to_csv(os.path.join(fold_base_path, 'test_metadata.csv'), index=False)

    print(f"Fold {fold_idx + 1} completed: {len(train_metadata)} train, {len(test_metadata)} test samples")

def main():
    """Main function"""
    print("Loading MIMII audio files...")
    all_mimii_files = load_all_mimii_audio_files()

    print("Creating 10-fold splits...")
    folds = create_10fold_splits(all_mimii_files)

    print(f"Starting generation of 10-fold cross-validation dataset...")

    for fold_data in folds:
        generate_fold_dataset(fold_data['fold_idx'], fold_data)

    print("\n10-fold dataset generation complete!")
    print(f"Dataset saved to: {OUTPUT_BASE_PATH}")

if __name__ == "__main__":
    main()