

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


@chatbot_bp.route('/write_essay_chatbot', methods=['POST'])
def write_essay_chatbot():
    data = request.get_json()
    reference = data.get('reference')
    essay = data.get('essay')
    model = data.get('model', 'llama3')

    if not reference or not essay:
        return jsonify({'error': 'Missing reference or essay'}), 400

    prompt_essay = (
        f"Essay Topic: {reference}\n"
        f"Student Essay: {essay}\n"
        "Please provide comprehensive feedback and improvement suggestions for this essay. Focus on the following areas:\n\n"
        "1. CONTENT:\n"
        "- How well does the essay address the given topic?\n"
        "- Does it provide relevant arguments and examples?\n"
        "- Is the content complete and well-developed?\n\n"
        "2. FORM:\n"
        "- Is the essay length appropriate (200-300 words)?\n"
        "- Is it properly formatted (no ALL CAPS, proper punctuation)?\n"
        "- Does it avoid bullet points or fragmented sentences?\n\n"
        "3. DEVELOPMENT, STRUCTURE & COHERENCE:\n"
        "- Is the essay well-structured with clear paragraphs?\n"
        "- Does it use logical connectors and transitions?\n"
        "- Is there a logical flow of ideas?\n\n"
        "4. GRAMMAR:\n"
        "- Are there grammar errors?\n"
        "- Is the grammar consistently accurate?\n"
        "- Are complex sentence structures used correctly?\n\n"
        "5. GENERAL LINGUISTIC RANGE:\n"
        "- Does the essay show variety in sentence structures?\n"
        "- Are complex sentences and passive voice used appropriately?\n"
        "- Is there linguistic sophistication?\n\n"
        "6. VOCABULARY RANGE:\n"
        "- Is there a good variety of vocabulary?\n"
        "- Are advanced words used appropriately?\n"
        "- Is the vocabulary precise and academic?\n\n"
        "7. SPELLING:\n"
        "- Are there spelling mistakes?\n"
        "- Is the spelling consistently accurate?\n\n"
        "Provide specific, constructive feedback for each area with examples from the essay. Include detailed suggestions for improvement and highlight both strengths and weaknesses. Be encouraging but honest about areas that need work. Do not mention any scores, points, or marks - focus purely on feedback and guidance for improvement."
    )

    messages = [{"role": "user", "content": prompt_essay}]

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


@chatbot_bp.route('/write_essay_test', methods=['POST'])
def write_essay_test():
    """
    Test endpoint that doesn't require Ollama - returns basic feedback
    """
    data = request.get_json()
    reference = data.get('reference')
    essay = data.get('essay')

    if not reference or not essay:
        return jsonify({'error': 'Missing reference or essay'}), 400

    # Simple feedback without AI
    word_count = len(essay.split())
    feedback = f"""
**BASIC FEEDBACK (No AI Required):**

**ESSAY LENGTH:**
Your essay has {word_count} words. 
{'✅ Good length (200-300 words recommended)' if 200 <= word_count <= 300 else '⚠️ Too short (aim for 200-300 words)' if word_count < 200 else '⚠️ Too long (aim for 200-300 words)'}

**FORMATTING:**
{'✅ No ALL CAPS detected' if not essay.isupper() else '❌ Avoid using ALL CAPS'}
{'✅ Proper punctuation detected' if any(char in essay for char in '.!?') else '❌ Add proper punctuation'}

**BASIC STRUCTURE:**
{'✅ Multiple sentences detected' if essay.count('.') + essay.count('!') + essay.count('?') > 1 else '❌ Add more sentences'}
{'✅ Paragraphs detected' if '\n\n' in essay else '⚠️ Consider adding paragraph breaks'}

**VOCABULARY:**
{'✅ Good variety of words' if len(set(essay.lower().split())) / word_count > 0.6 else '⚠️ Consider using more varied vocabulary'}

**TOPIC RELEVANCE:**
The essay appears to address the topic: "{reference[:50]}..."

**NEXT STEPS:**
1. Install Ollama: https://ollama.ai/
2. Run: ollama serve
3. Install model: ollama pull llama2 (smaller model, works with less memory)
4. Try the /write_essay_chatbot endpoint for detailed AI feedback
    """

    return Response(feedback, content_type="text/plain", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Content-Encoding": "none"
    })
