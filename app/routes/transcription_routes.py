from flask import Blueprint, request, jsonify
from app.services.audio_transcriber import transcribe_audio
import os

transcription_bp = Blueprint('transcription', __name__)
UPLOAD_FOLDER = 'uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@transcription_bp.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    result, status_code = transcribe_audio(file, UPLOAD_FOLDER)
    return jsonify(result), status_code
