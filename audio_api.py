from flask import Flask, request, jsonify
import requests
import numpy as np
import soundfile as sf
import time
from contextlib import contextmanager
import os

app = Flask(__name__)

# Define your existing functions here

@contextmanager
def timer():
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

def downsample_array(arr, factor):
    new_length = len(arr) // factor
    downsampled_array = arr[:new_length * factor:factor]
    return downsampled_array

def average_amplitude_excluding_silence(sound_wave, silence_threshold=0.005):
    non_silent_samples = sound_wave[np.abs(sound_wave) > silence_threshold]
    mean = np.mean(np.abs(non_silent_samples))
    std = np.std(np.abs(non_silent_samples))
    return mean, std

def find_epochs(data, samplerate, silence_duration=5, min_epoch_duration=5, threshold=0.005):
    silence_samples = silence_duration * samplerate
    min_epoch_samples = min_epoch_duration * samplerate
    sound_indices = np.where(np.abs(data) > threshold)[0]
    if len(sound_indices) == 0:
        return np.array([]), np.array([])
    breaks = np.where(np.diff(sound_indices) > silence_samples)[0]
    segment_starts = np.concatenate(([0], breaks + 1))
    segment_ends = np.concatenate((breaks, [len(sound_indices) - 1]))
    onsets = sound_indices[segment_starts]
    offsets = sound_indices[segment_ends]
    durations = offsets - onsets
    valid_epochs = durations >= min_epoch_samples
    onsets = onsets[valid_epochs] / samplerate
    offsets = offsets[valid_epochs] / samplerate
    durations = durations[valid_epochs] / samplerate
    return onsets, offsets, durations

def flatten_to_1d(data):
    return np.mean(data, axis=tuple(range(1, data.ndim))) if data.ndim > 1 else data

def norm_data(data):
    non_zero_mask = data != 0
    non_zero_data = data[non_zero_mask]
    mean_non_zero = np.mean(non_zero_data)
    std_non_zero = np.std(non_zero_data)
    normalized_data = (data - mean_non_zero) / std_non_zero
    min_val, max_val = -3, 3
    normalized_data = np.clip(normalized_data, min_val, max_val)
    return normalized_data

def overwrite_below_threshold(data, threshold=0.5, min_duration=3, samplerate=1):
    min_samples = int(min_duration * samplerate)
    below_threshold = np.abs(data) <= threshold
    starts = np.where(np.diff(np.concatenate(([False], below_threshold, [False]))))[0]
    for start, end in zip(starts[0::2], starts[1::2]):
        if end - start >= min_samples:
            data[start:end] = 0
    return data

@app.route('/process_audio', methods=['POST'])
def process_audio():
    data = request.json
    file_url = data.get('file_url')
    if not file_url:
        return jsonify({'error': 'No file URL provided'}), 400
    
    response = requests.get(file_url)
    if response.status_code != 200:
        return jsonify({'error': 'Failed to download file'}), 400

    temp_file = 'temp_audio.wav'
    with open(temp_file, 'wb') as f:
        f.write(response.content)

    try:
        downfac = 1000
        silamp_thresh = .005

        audio_data, sample_rate = sf.read(temp_file)
        aud_data = flatten_to_1d(downsample_array(audio_data, downfac))
        aud_data[np.abs(aud_data) <= silamp_thresh] = 0
        aud_data = norm_data(aud_data)
        aud_data = overwrite_below_threshold(aud_data, threshold=0.5, min_duration=5, samplerate=(sample_rate / downfac))
        aud_time = np.linspace(0, len(aud_data) / (sample_rate / downfac), num=len(aud_data))

        onsets, offsets, durations = find_epochs(aud_data, (sample_rate / downfac), silence_duration=5, min_epoch_duration=5, threshold=0.005)
        
        result = {
            'num_epochs': len(onsets),
            'epochs': [
                {'onset': float(onset), 'offset': float(offset), 'duration': float(duration)}
                for onset, offset, duration in zip(onsets, offsets, durations)
            ]
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
