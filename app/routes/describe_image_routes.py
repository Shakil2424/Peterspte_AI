from flask import Blueprint, request, jsonify, current_app
import tempfile
import os
from app.services.describe_image_service import evaluate_describe_image

describe_image_bp = Blueprint('describe_image', __name__)

@describe_image_bp.route('/describe_image', methods=['POST'])
def describe_image():
    reference = request.form.get('reference')
    file = request.files.get('file')
    if not reference or not file:
        return jsonify({'error': 'Missing reference or audio file'}), 400
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    result, status_code = evaluate_describe_image(reference, file, upload_folder)
    return jsonify(result), status_code 