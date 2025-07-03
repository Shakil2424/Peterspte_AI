from flask import Blueprint, request, jsonify
from app.services.dictation_service import dictation_ai

dictation_bp = Blueprint('dictation', __name__)

@dictation_bp.route('/dictation', methods=['POST'])
def dictation():
    if request.is_json:
        data = request.get_json()
        reference = data.get('reference')
        response = data.get('response')
    else:
        reference = request.form.get('reference')
        response = request.form.get('response')
    if not reference or not response:
        return jsonify({'error': 'Missing reference or response'}), 400
    ai_result = dictation_ai(response, reference)
    return jsonify(ai_result), 200 