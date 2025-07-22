from flask import Blueprint, request, jsonify, current_app
from app.services.summarize_group_service import evaluate_summarize_group

summarize_group_bp = Blueprint('summarize_group', __name__)

@summarize_group_bp.route('/summarize_group', methods=['POST'])
def summarize_group():
    reference = request.form.get('reference')
    file = request.files.get('file')
    if not reference or not file:
        return jsonify({'error': 'Missing reference or audio file'}), 400
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    result, status_code = evaluate_summarize_group(reference, file, upload_folder)
    return jsonify(result), status_code 