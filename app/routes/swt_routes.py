from flask import Blueprint, request, jsonify, Response
from app.services.swt_service import evaluate_summary_service
from app.services.chatbot_service import stream_chatbot_response

swt_bp = Blueprint('swt', __name__)

@swt_bp.route('/swt', methods=['POST'])
def swt():
    data = request.get_json()
    summary = data.get('summary')
    reference = data.get('reference')
    if not summary or not reference:
        return jsonify({'error': 'Missing summary or reference'}), 400
    result = evaluate_summary_service(summary, reference)
    return jsonify(result), 200

