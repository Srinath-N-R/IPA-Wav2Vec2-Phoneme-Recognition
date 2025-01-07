from pathlib import Path
from os.path import exists
from typing import Dict
from tqdm import tqdm

from pathlib import Path
from typing import Dict
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_file(audio_file: Path, base_folder: Path) -> Dict:
    """Process a single audio file to generate its corresponding data entry."""
    txt_folder = base_folder / 'txt'
    phonemized_folder = base_folder / 'phonemized'

    relative_path = audio_file.relative_to(base_folder / 'wav')
    word_file = txt_folder / relative_path.parent / relative_path.name.replace('.wav', '.txt')
    phonetic_file = phonemized_folder / relative_path.parent / relative_path.name.replace('.wav', '.txt')

    # Check file existence
    if not (os.path.exists(audio_file) and os.path.exists(word_file) and os.path.exists(phonetic_file)):
        return None

    # Return data entry
    return {
        'audio_file': str(audio_file),
        'word_file': str(word_file),
        'phonetic_file': str(phonetic_file)
    }

def return_data(dataset_folder: str, max_files: int = 1000, num_workers: int = 4) -> Dict:
    wav_folder = Path(dataset_folder) / 'wav'

    # Collect audio files and limit to max_files if needed
    audio_files = [
        path for path in wav_folder.rglob('*.wav') if not path.name.startswith('._')
    ]
    if max_files != -1:
        audio_files = audio_files[:max_files]

    base_folder = Path(dataset_folder)
    data = {}

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(process_file, audio_file, base_folder): idx for idx, audio_file in enumerate(audio_files)}
        
        for future in tqdm(as_completed(future_to_file), total=len(audio_files), desc="Processing files"):
            idx = future_to_file[future]
            result = future.result()
            if result is not None:
                data[idx] = result

    return data
