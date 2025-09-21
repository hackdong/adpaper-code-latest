import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import torchaudio
import torch

class FrameworkVisualizer:
    def __init__(self, audio_dir):

        self.audio_dir = Path(audio_dir)
        

        self.sample_rate = 16000
        self.n_fft = 400
        self.hop_length = 160
        self.n_mels = 64
        self.f_min = 50
        self.f_max = 8000
        

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max
        )

    def load_audio(self, file_path):

        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        return waveform.numpy()[0]  

    def plot_waveform_and_melspec(self, audio_files, output_path):

        n_files = len(audio_files)
        fig, axes = plt.subplots(n_files, 2, figsize=(15, 5*n_files))
        
        if n_files == 1:
            axes = axes.reshape(1, -1)
        
        for i, audio_file in enumerate(audio_files):

            waveform = self.load_audio(audio_file)
            

            mel_spec = self.mel_spectrogram(torch.from_numpy(waveform))
            mel_spec_db = librosa.power_to_db(mel_spec.numpy(), ref=np.max)
            

            axes[i, 0].plot(np.linspace(0, len(waveform)/self.sample_rate, len(waveform)), 
                          waveform)
            axes[i, 0].set_title(f'Waveform - {Path(audio_file).stem}')
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('Amplitude')
            

            img = librosa.display.specshow(mel_spec_db, 
                                         sr=self.sample_rate,
                                         hop_length=self.hop_length,
                                         x_axis='time',
                                         y_axis='mel',
                                         fmin=self.f_min,
                                         fmax=self.f_max,
                                         ax=axes[i, 1])
            axes[i, 1].set_title(f'Mel Spectrogram - {Path(audio_file).stem}')
            plt.colorbar(img, ax=axes[i, 1], format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=1200, bbox_inches='tight')
        plt.close()

def main():

    audio_dir = "./dataset/synthetic_validation_dataset_v3"
    output_dir = "./framework_visualization"
    os.makedirs(output_dir, exist_ok=True)
    

    audio_files = [

        "./dataset/synthetic_validation_dataset_v3/synthetic_validation_2372_valve_01_normal.wav",

        "./dataset/synthetic_validation_dataset_v3/synthetic_validation_2381_valve_01_normal.wav",

        "./dataset/synthetic_validation_dataset_v3/synthetic_validation_2380_fan_01_normal.wav"
    ]
    

    visualizer = FrameworkVisualizer(audio_dir)
    

    visualizer.plot_waveform_and_melspec(
        audio_files,
        os.path.join(output_dir, 'framework_audio_visualization.png')
    )
    
    print("Visualization completed! Results saved to:", output_dir)

if __name__ == "__main__":
    main() 