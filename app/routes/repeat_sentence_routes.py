from flask import Blueprint, request, jsonify, current_app
from app.services.repeat_sentence_service import evaluate_repeat_sentence

repeat_sentence_bp = Blueprint('repeat_sentence', __name__)

@repeat_sentence_bp.route('/repeat_sentence', methods=['POST'])
def repeat_sentence():
    reference = request.form.get('reference')
    file = request.files.get('file')
    if not reference or not file:
        return jsonify({'error': 'Missing reference or audio file'}), 400
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    result, status_code = evaluate_repeat_sentence(reference, file, upload_folder)
    return jsonify(result), status_code 