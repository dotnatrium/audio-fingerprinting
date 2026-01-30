import os
import sys
import json
import glob
import logging
import hashlib
import warnings
import numpy as np
import librosa
import h5py
import soundfile as sf
from scipy import signal
from datetime import datetime
from multiprocessing import Pool, cpu_count
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.theme import Theme

custom_theme = Theme({"info": "dim cyan", "warning": "magenta", "danger": "bold red", "success": "bold green"})
console = Console(theme=custom_theme)

CONFIG = {
    'INPUT_DIR': 'audio',
    'OUTPUT_DIR': 'data',
    'DB_FILE': 'fingerprints.h5',
    'REGISTRY_FILE': 'metadata.json',
    'SAMPLE_RATE': 22050,
    'DURATION': None,
    'HOP_LENGTH': 512,
    'N_BINS': 84,
    'BINS_PER_OCTAVE': 12,
    'FMIN': 30,
    'FMAX': 10000,
    'NOISE_FACTOR': 0.005,
    'NUM_WORKERS': max(1, cpu_count() - 1)
}

logging.basicConfig(filename='progress.log', level=logging.ERROR)
warnings.filterwarnings('ignore')

def generate_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def clean_audio(y, sr):
    try:
        sos = signal.butter(10, [CONFIG['FMIN'], CONFIG['FMAX']], 'bandpass', fs=sr, output='sos')
        cleaned_y = signal.sosfilt(sos, y)
        max_peak = np.max(np.abs(cleaned_y))
        if max_peak > 0:
            cleaned_y = cleaned_y / max_peak * 0.95
        return cleaned_y
    except:
        return None

def augment_audio(y):
    try:
        noise = np.random.randn(len(y))
        augmented = y + (CONFIG['NOISE_FACTOR'] * noise)
        if np.max(np.abs(augmented)) < 0.001:
            return None
        return augmented
    except:
        return None

def compute_cqt(y, sr):
    cqt = librosa.cqt(
        y, sr=sr,
        hop_length=CONFIG['HOP_LENGTH'],
        n_bins=CONFIG['N_BINS'],
        bins_per_octave=CONFIG['BINS_PER_OCTAVE']
    )
    return librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

def worker_process(args):
    file_path, existing_hash = args
    filename = os.path.basename(file_path)
    
    try:
        current_hash = generate_file_hash(file_path)
        if existing_hash == current_hash:
            return {'status': 'SKIPPED', 'filename': filename}

        y, sr = librosa.load(file_path, sr=CONFIG['SAMPLE_RATE'], duration=CONFIG['DURATION'], res_type='kaiser_fast')
        
        y_clean = clean_audio(y, sr)
        if y_clean is None: raise ValueError("Signal cleaning failed")

        y_distorted = augment_audio(y_clean)
        if y_distorted is None: return {'status': 'ERROR', 'filename': filename, 'msg': 'Distortion NULL'}

        feat_clean = compute_cqt(y_clean, sr)
        feat_distorted = compute_cqt(y_distorted, sr)

        return {
            'status': 'SUCCESS',
            'filename': filename,
            'hash': current_hash,
            'duration': float(librosa.get_duration(y=y_clean, sr=sr)),
            'feat_clean': feat_clean,
            'feat_distorted': feat_distorted
        }

    except Exception as e:
        return {'status': 'ERROR', 'filename': filename, 'msg': str(e)}

class DatasetBuilder:
    def __init__(self):
        os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
        self.registry_path = os.path.join(CONFIG['OUTPUT_DIR'], CONFIG['REGISTRY_FILE'])
        self.h5_path = os.path.join(CONFIG['OUTPUT_DIR'], CONFIG['DB_FILE'])
        self.registry = self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"files": {}}

    def save_registry(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4)

    def run(self):
        extensions = ('*.mp3', '*.wav', '*.flac', '*.m4a')
        audio_files = []
        for ext in extensions:
            audio_files.extend(glob.glob(os.path.join(CONFIG['INPUT_DIR'], ext)))
        
        if not audio_files:
            console.print(f"[danger]No audio files found in {CONFIG['INPUT_DIR']}[/]")
            return

        console.print(f"[info]Starting Pipeline[/] | [bold]{len(audio_files)}[/] Files | [bold]{CONFIG['NUM_WORKERS']}[/] Workers")

        tasks = [(f, self.registry['files'].get(os.path.basename(f), {}).get('hash')) for f in audio_files]
        stats = {'SUCCESS': 0, 'SKIPPED': 0, 'ERROR': 0}

        with h5py.File(self.h5_path, 'a') as hf:
            grp_clean = hf.require_group('clean')
            grp_dist = hf.require_group('distorted')

            with Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=None, complete_style="green", finished_style="green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                
                task_id = progress.add_task("Processing...", total=len(tasks))
                
                with Pool(processes=CONFIG['NUM_WORKERS']) as pool:
                    for result in pool.imap_unordered(worker_process, tasks):
                        status = result['status']
                        fname = result['filename']

                        if status == 'SUCCESS':
                            h = result['hash']
                            
                            if h in grp_clean: del grp_clean[h]
                            if h in grp_dist: del grp_dist[h]

                            grp_clean.create_dataset(h, data=result['feat_clean'], compression="gzip")
                            grp_dist.create_dataset(h, data=result['feat_distorted'], compression="gzip")

                            self.registry['files'][fname] = {
                                'hash': h,
                                'duration': result['duration'],
                                'ts': str(datetime.now())
                            }
                            progress.console.print(f"[success]✔ {fname}[/]")
                            stats['SUCCESS'] += 1

                        elif status == 'SKIPPED':
                            progress.console.print(f"[dim]⏭ {fname}[/]")
                            stats['SKIPPED'] += 1

                        elif status == 'ERROR':
                            progress.console.print(f"[danger]✘ {fname}: {result['msg']}[/]")
                            logging.error(f"{fname}: {result['msg']}")
                            stats['ERROR'] += 1
                        
                        progress.advance(task_id)

        try:
            self.save_registry()
            console.print(f"\n[bold green]Job Finished.[/] [white]New: {stats['SUCCESS']} | Cached: {stats['SKIPPED']} | Failed: {stats['ERROR']}[/]")
        except Exception as e:
            console.print(f"[danger]Failed to save registry: {e}[/]")

if __name__ == '__main__':
    try:
        app = DatasetBuilder()
        app.run()
    except KeyboardInterrupt:
        console.print("\n[bold red]Aborted by user.[/]")
        sys.exit()
