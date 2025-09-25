import pandas as pd
import yt_dlp
import os
from tqdm import tqdm
import time
import json
import subprocess

class VGGAudioDownloader:
    def __init__(self, csv_path, output_dir, segment_length=10, max_downloads=500):

        self.csv_path = csv_path
        self.output_dir = output_dir
        self.segment_length = segment_length
        self.max_downloads = max_downloads
        

        os.makedirs(output_dir, exist_ok=True)
        

        self.download_record_path = os.path.join(output_dir, 'download_record.json')
        self.download_record = self._load_download_record()
        

        self.ffmpeg_path = r'C:/Program Files/ffmpeg-7.1-full_build/bin/ffmpeg.exe'  # 使用原始字符串
        
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'quiet': True,
            'no_warnings': True,
            'extract_audio': True,
            'outtmpl': os.path.join(output_dir, 'temp', '%(id)s.%(ext)s'),
            'ffmpeg_location': self.ffmpeg_path,
        }


        self.df = pd.read_csv(csv_path, header=None, 
                            names=['youtube_id', 'start_seconds', 'label', 'split'])

    def _load_download_record(self):

        if os.path.exists(self.download_record_path):
            with open(self.download_record_path, 'r') as f:
                return json.load(f)
        return {'completed': [], 'failed': []}

    def _save_download_record(self):

        with open(self.download_record_path, 'w') as f:
            json.dump(self.download_record, f)

    def download_audio(self):


        segments_dir = os.path.join(self.output_dir, 'segments')
        temp_dir = os.path.join(self.output_dir, 'temp')
        os.makedirs(segments_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        download_count = 0
        processed_ids = set()  
        

        for _, row in tqdm(self.df.iterrows(), desc="Downloading segments", total=len(self.df)):
            if download_count >= self.max_downloads:
                print(f"\nReached maximum download limit ({self.max_downloads})")
                break
                
            youtube_id = row['youtube_id']
            start_time = row['start_seconds']
            label = row['label']
            split = row['split']
            

            if youtube_id in processed_ids or youtube_id in self.download_record['failed']:
                continue
                
            output_filename = f"{youtube_id}_{start_time}_{label}_{split}.wav"
            output_path = os.path.join(segments_dir, output_filename)
            
            if os.path.exists(output_path):
                continue
                
            try:
                url = f"https://www.youtube.com/watch?v={youtube_id}"
                temp_file = os.path.join(temp_dir, f"{youtube_id}.wav")
                

                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    ydl.download([url])
                

                subprocess.run([
                    self.ffmpeg_path,
                    '-i', temp_file,
                    '-ss', str(start_time),
                    '-t', str(self.segment_length),
                    '-c', 'copy',
                    output_path,
                    '-y'
                ], check=True)
                

                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                if youtube_id not in self.download_record['completed']:
                    self.download_record['completed'].append(youtube_id)
                    
                processed_ids.add(youtube_id)
                download_count += 1
                
            except Exception as e:
                print(f"Failed to download segment {youtube_id} at {start_time}: {str(e)}")
                if youtube_id not in self.download_record['failed']:
                    self.download_record['failed'].append(youtube_id)
                
            self._save_download_record()

def main():

    csv_path = "./dataset/vggdata/vggsound.csv"  
    output_dir = "./dataset/vggdata/vgg_audio"  
    

    downloader = VGGAudioDownloader(csv_path, output_dir, max_downloads=500)
    

    downloader.download_audio()
    
    print("\nDownload completed!")
    print(f"Successfully downloaded: {len(downloader.download_record['completed'])} files")
    print(f"Failed downloads: {len(downloader.download_record['failed'])} files")

if __name__ == "__main__":
    main() 