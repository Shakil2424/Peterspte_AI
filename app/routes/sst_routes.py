from flask import Blueprint, request, jsonify
from app.services.sst_service import evaluate_sst_service

sst_bp = Blueprint('sst', __name__)

@sst_bp.route('/sst', methods=['POST'])
def sst():
    """
    SST (Summarize Spoken Text) endpoint
    Expects: reference text + summary text
    Returns: detailed scoring across 5 criteria
    """
    if request.is_json:
        data = request.get_json()
        reference = data.get('reference')
        summary = data.get('summary')
    else:
        reference = request.form.get('reference')
        summary = request.form.get('summary')
    
    if not reference or not summary:
        return jsonify({'error': 'Missing reference or summary'}), 400

    # Evaluate the summary against the reference
    result = evaluate_sst_service(summary, reference)
    
    return jsonify(result), 200 