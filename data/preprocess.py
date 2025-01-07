import librosa
from typing import Dict

def get_durations(data_dict: Dict, sampling_rate: int = 16000) -> int:
    total_durations = 0
    for entry in data_dict.values():
        audio_data, _ = librosa.load(entry['audio_file'], sr=sampling_rate)
        duration = len(audio_data) / sampling_rate
        total_durations += duration
    return int(total_durations)