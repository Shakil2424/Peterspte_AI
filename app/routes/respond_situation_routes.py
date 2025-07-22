from flask import Blueprint, request, jsonify, current_app
import tempfile
import os
from app.services.respond_situation_service import evaluate_respond_situation

respond_situation_bp = Blueprint('respond_situation', __name__)

@respond_situation_bp.route('/respond_situation', methods=['POST'])
def respond_situation():
    reference = request.form.get('reference')
    file = request.files.get('file')
    if not reference or not file:
        return jsonify({'error': 'Missing reference or audio file'}), 400
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    result, status_code = evaluate_respond_situation(reference, file, upload_folder)
    return jsonify(result), status_code 