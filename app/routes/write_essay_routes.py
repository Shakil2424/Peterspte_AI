from flask import Blueprint, request, jsonify
from app.services.write_essay_service import evaluate_write_essay

write_essay_bp = Blueprint('write_essay', __name__)

@write_essay_bp.route('/write_essay', methods=['POST'])
def write_essay():
    """
    Write Essay endpoint
    Expects: reference text + essay text
    Returns: detailed scoring across 7 criteria
    """
    if request.is_json:
        data = request.get_json()
        reference = data.get('reference')
        essay = data.get('essay')
    else:
        reference = request.form.get('reference')
        essay = request.form.get('essay')
    
    if not reference or not essay:
        return jsonify({'error': 'Missing reference or essay'}), 400

    # Evaluate the essay against the reference
    result = evaluate_write_essay(essay, reference)
    
    return jsonify(result), 200 