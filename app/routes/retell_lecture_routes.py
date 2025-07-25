from flask import Blueprint, request, jsonify, current_app
from app.services.retell_lecture_service import evaluate_retell_lecture

retell_lecture_bp = Blueprint('retell_lecture', __name__)

@retell_lecture_bp.route('/retell_lecture', methods=['POST'])
def retell_lecture():
    reference = request.form.get('reference')
    file = request.files.get('file')
    if not reference or not file:
        return jsonify({'error': 'Missing reference or audio file'}), 400
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    result, status_code = evaluate_retell_lecture(reference, file, upload_folder)
    return jsonify(result), status_code 