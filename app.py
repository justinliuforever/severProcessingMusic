from flask import Flask, request, jsonify
import requests
import os
from audio_processing import process_audio_file

app = Flask(__name__)

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
        result = process_audio_file(temp_file)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return jsonify(result)

if __name__ == '__main__':
    from config import SERVER_HOST, SERVER_PORT
    app.run(debug=True, host=SERVER_HOST, port=SERVER_PORT)
