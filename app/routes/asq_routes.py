from flask import Blueprint, request, jsonify
from app.services.asq_service import assess_audio_answer
import os

asq_bp = Blueprint('asq', __name__)

@asq_bp.route('/asq', methods=['POST'])
def assess_speaking():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    result = assess_audio_answer(file)
    return jsonify(result), 200 