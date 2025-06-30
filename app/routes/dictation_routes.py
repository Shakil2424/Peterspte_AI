from flask import Blueprint, request, jsonify
from app.services.audio_transcriber import transcribe_audio
from app.services.dictation_service import dictation_ai
import os

dictation_bp = Blueprint('dictation', __name__)
UPLOAD_FOLDER = 'uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@dictation_bp.route('/dictation', methods=['POST'])
def dictation():
    if 'file' not in request.files or 'reference' not in request.form:
        return jsonify({'error': 'No file or reference text provided'}), 400
    file = request.files['file']
    reference = request.form['reference']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    result, status_code = transcribe_audio(file, UPLOAD_FOLDER)
    if status_code != 200:
        return jsonify(result), status_code
    user_text = result.get('transcript', '')
    ai_result = dictation_ai(user_text, reference)
    return jsonify(ai_result), 200 