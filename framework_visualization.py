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
        """
        初始化可视化器
        Args:
            audio_dir: 音频文件目录
        """
        self.audio_dir = Path(audio_dir)
        
        # 设置音频参数
        self.sample_rate = 16000
        self.n_fft = 400
        self.hop_length = 160
        self.n_mels = 64
        self.f_min = 50
        self.f_max = 8000
        
        # 初始化梅尔频谱转换器
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max
        )

    def load_audio(self, file_path):
        """加载音频文件"""
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        return waveform.numpy()[0]  # 转换为numpy数组并获取第一个通道

    def plot_waveform_and_melspec(self, audio_files, output_path):
        """为每个音频文件绘制波形图和梅尔频谱图"""
        n_files = len(audio_files)
        fig, axes = plt.subplots(n_files, 2, figsize=(15, 5*n_files))
        
        if n_files == 1:
            axes = axes.reshape(1, -1)
        
        for i, audio_file in enumerate(audio_files):
            # 加载音频
            waveform = self.load_audio(audio_file)
            
            # 计算梅尔频谱图
            mel_spec = self.mel_spectrogram(torch.from_numpy(waveform))
            mel_spec_db = librosa.power_to_db(mel_spec.numpy(), ref=np.max)
            
            # 绘制波形图
            axes[i, 0].plot(np.linspace(0, len(waveform)/self.sample_rate, len(waveform)), 
                          waveform)
            axes[i, 0].set_title(f'Waveform - {Path(audio_file).stem}')
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('Amplitude')
            
            # 绘制梅尔频谱图
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
    # 配置路径
    audio_dir = "./dataset/synthetic_validation_dataset_v3"
    output_dir = "./framework_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择示例音频文件
    audio_files = [
        # 选择一个正常的音频文件
        "./dataset/synthetic_validation_dataset_v3/synthetic_validation_2372_valve_01_normal.wav",
        # 选择一个有事件的音频文件
        "./dataset/synthetic_validation_dataset_v3/synthetic_validation_2381_valve_01_normal.wav",
        # 选择一个异常的音频文件
        "./dataset/synthetic_validation_dataset_v3/synthetic_validation_2380_fan_01_normal.wav"
    ]
    
    # 创建可视化器
    visualizer = FrameworkVisualizer(audio_dir)
    
    # 生成波���图和梅尔频谱图
    visualizer.plot_waveform_and_melspec(
        audio_files,
        os.path.join(output_dir, 'framework_audio_visualization.png')
    )
    
    print("Visualization completed! Results saved to:", output_dir)

if __name__ == "__main__":
    main() 