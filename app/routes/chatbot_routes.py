from flask import Blueprint, request, jsonify, Response
from app.services.chatbot_service import stream_chatbot_response
import json

chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    prompt = data.get('prompt')
    model = data.get('model', 'llama3')
    if not prompt:
        return jsonify({'error': 'Missing prompt'}), 400
    messages = [{"role": "user", "content": prompt}]
    return Response(stream_chatbot_response(messages, model), mimetype='text/plain')

@chatbot_bp.route('/swt_chatbot', methods=['POST'])
def swt_chatbot():
    data = request.get_json()
    reference = data.get('reference')
    summary = data.get('summary')
    model = data.get('model', 'llama3')
    if not reference or not summary:
        return jsonify({'error': 'Missing reference or summary'}), 400
    prompt = (
        f"Reference: {reference}\n"
        f"Summary: {summary}\n"
        "Please provide feedback on my writing (summarize), suggestions to improve, and an improved version of my summary as a single, complete sentence between 5 and 75 words.\n"
        "Ensure your improved version:\n"
        "- Covers all main points clearly\n"
        "- Is grammatically correct\n"
        "- Uses appropriate, varied vocabulary\n"
        "- Is exactly one complete sentence, not a list or fragmented\n"
        "Do not mention any scores, marks, or rubric points in your response."
    )
    messages = [{"role": "user", "content": prompt}]
    response_text = ""
    for chunk in stream_chatbot_response(messages, model):
        response_text += chunk
    return jsonify({'response': response_text}), 200
