

from flask import Blueprint, request, jsonify, Response, stream_with_context
from app.services.chatbot_service import stream_chatbot_response
import json

chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    prompt = data.get('prompt')
    model = data.get('model', 'llama2')
    if not prompt:
        return jsonify({'error': 'Missing prompt'}), 400
    messages = [{"role": "user", "content": prompt}]

    @stream_with_context
    def generate():
        try:
            for chunk in stream_chatbot_response(messages, model):
                yield chunk
        except Exception as e:
            yield f"\n[Stream Error: {str(e)}]\n"

    return Response(generate(), mimetype='text/plain', headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # disable nginx buffering if using nginx proxy
        "Content-Encoding": "none"
    })


@chatbot_bp.route('/swt_chatbot', methods=['POST'])
def swt_chatbot():
    data = request.get_json()
    reference = data.get('reference')
    summary = data.get('summary')
    model = data.get('model', 'llama2')

    if not reference or not summary:
        return jsonify({'error': 'Missing reference or summary'}), 400
    
    prompt = (
        f"Reference: {reference}\n"
        f"Summary: {summary}\n"
        "Please evaluate how well this summary captures the reference material, then provide:\n"
        "1. Feedback on the summary's accuracy and writing quality\n"
        "2. Specific suggestions for improvement\n"
        "3. An improved version as a single, complete sentence (5-75 words)\n\n"
        "Your improved version should:\n"
        "- Cover all main points from the reference clearly\n"
        "- Be grammatically correct\n"
        "- Use appropriate, varied vocabulary\n"
        "- Form exactly one complete sentence\n\n"
        "Do not mention scores, marks, or rubric points in your response."
    )

    messages = [{"role": "user", "content": prompt}]

    @stream_with_context
    def generate():
        try:
            for chunk in stream_chatbot_response(messages, model):
                yield chunk
        except Exception as e:
            yield f"\n[Stream Error: {str(e)}]\n"

    return Response(generate(), content_type="text/plain", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Content-Encoding": "none"
    })


@chatbot_bp.route('/sst_chatbot', methods=['POST'])
def sst_chatbot():
    data = request.get_json()
    reference = data.get('reference')
    summary = data.get('summary')
    model = data.get('model', 'llama2')

    if not reference or not summary:
        return jsonify({'error': 'Missing reference or summary'}), 400

    prompt_sst = (
        f"Reference: {reference}\n"
        f"Summary: {summary}\n"
        "Please evaluate how well this summary captures the reference material, then provide:\n"
        "1. Feedback on the summary's accuracy and writing quality\n"
        "2. Specific suggestions for improvement\n"
        "3. An improved version as a single, complete sentence (5-75 words)\n\n"
        "Your improved version should:\n"
        "- Cover all main points from the reference clearly\n"
        "- Be grammatically correct\n"
        "- Use appropriate, varied vocabulary\n"
        "- Form exactly one complete sentence\n\n"
        "Do not mention scores, marks, or rubric points in your response."
    )

    messages = [{"role": "user", "content": prompt_sst}]

    @stream_with_context
    def generate():
        try:
            for chunk in stream_chatbot_response(messages, model):
                yield chunk
        except Exception as e:
            yield f"\n[Stream Error: {str(e)}]\n"

    return Response(generate(), content_type="text/plain", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Content-Encoding": "none"
    })

