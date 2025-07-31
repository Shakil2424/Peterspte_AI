from flask import Blueprint, request, jsonify
from app.services.read_aloud_service import evaluate_read_aloud
import os

read_aloud_bp = Blueprint('read_aloud', __name__)

@read_aloud_bp.route('/read_aloud', methods=['POST'])
def read_aloud():
    """
    Read Aloud endpoint
    Expects: audio file + reference text
    Returns: content, pronunciation, and fluency scores (10-90 range)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['file']
    reference = request.form.get('reference')
    
    if not reference:
        return jsonify({'error': 'Missing reference text'}), 400
    
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    
    # Save audio file temporarily
    filename = os.path.join('uploads', audio_file.filename)
    audio_file.save(filename)
    
    try:
        result = evaluate_read_aloud(filename, reference)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': f'Evaluation failed: {str(e)}'}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(filename):
            os.remove(filename) 