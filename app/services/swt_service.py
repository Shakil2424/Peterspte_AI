from sentence_transformers import SentenceTransformer, util
from lexicalrichness import LexicalRichness
import language_tool_python
import re

# Load models/tools once
sbert_model = SentenceTransformer('all-mpnet-base-v2')
lang_tool = language_tool_python.LanguageTool('en-US')

def evaluate_summary_service(summary, reference):
    scores = {}
    # === 1. Form (1 point) ===
    words = summary.strip().split()
    word_count = len(words)
    sentence_count = len(re.findall(r'[.!?]', summary.strip()))
    if 5 <= word_count <= 75 and sentence_count == 1 and not summary.isupper():
        scores['form'] = 1
    else:
        scores['form'] = 0
    # === 2. Content (2 points) ===
    emb_ref = sbert_model.encode(reference, convert_to_tensor=True)
    emb_sum = sbert_model.encode(summary, convert_to_tensor=True)
    similarity = util.cos_sim(emb_ref, emb_sum).item()
    if similarity >= 0.85:
        scores['content'] = 2
    elif similarity >= 0.70:
        scores['content'] = 1
    else:
        scores['content'] = 0
    # === 3. Grammar (2 points) ===
    matches = lang_tool.check(summary)
    grammar_errors = [m for m in matches if m.ruleIssueType in ("grammar", "typographical")]
    spelling_errors = [m for m in matches if m.ruleIssueType in ("spelling", "misspelling")]
    num_errors = len(grammar_errors)
    if num_errors == 0:
        scores['grammar'] = 2
    elif num_errors <= 2:
        scores['grammar'] = 1
    else:
        scores['grammar'] = 0
    # === 4. Vocabulary (2 points) ===
    lex = LexicalRichness(summary)
    ttr = lex.ttr
    mtld = lex.mtld()
    if ttr > 0.7 and mtld > 20:
        scores['vocabulary'] = 2
    elif ttr > 0.5 and mtld > 15:
        scores['vocabulary'] = 1
    else:
        scores['vocabulary'] = 0
    # === Word Highlights for Grammar/Typo/Spelling Errors ===
    word_highlights = []
    for word in words:
        word_highlights.append({"word": word, "status": "correct", "replacement": None})
    
    # Process grammar errors
    for error in grammar_errors:
        error_text = summary[error.offset:error.offset+error.errorLength]
        error_words = error_text.split()
        for i, highlight in enumerate(word_highlights):
            if highlight['word'] in error_words:
                word_highlights[i] = {
                    "word": highlight['word'],
                    "status": "grammar",
                    "replacement": error.replacements[0] if error.replacements else None,
                }
                break
    
    # Process spelling errors (show in word highlights but don't affect score)
    for error in spelling_errors:
        error_text = summary[error.offset:error.offset+error.errorLength]
        error_words = error_text.split()
        suggestions = error.replacements[:3] if error.replacements else []
        for i, highlight in enumerate(word_highlights):
            if highlight['word'] in error_words:
                word_highlights[i] = {
                    "word": highlight['word'],
                    "status": "spelling",
                    "suggestions": suggestions,
                }
                break
    # === Final Total ===
    total = sum(scores.values())
    scores['total'] = total
    # Add details for API response
    scores['details'] = {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'similarity': similarity,
        'grammar_errors': [err.message for err in grammar_errors],
        'spelling_errors': [err.message for err in spelling_errors],
        'ttr': ttr,
        'mtld': mtld
    }
    scores['word_highlights'] = word_highlights
    return scores 