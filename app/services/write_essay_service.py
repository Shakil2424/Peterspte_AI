from sentence_transformers import SentenceTransformer, util
from lexicalrichness import LexicalRichness
import language_tool_python
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Load models/tools once
sbert_model = SentenceTransformer('all-mpnet-base-v2')
lang_tool = language_tool_python.LanguageTool('en-US')

def evaluate_write_essay(essay, reference):
    """
    Evaluate Write Essay across 7 criteria
    """
    scores = {}
    
    # === 1. Content (0-3 points) ===
    emb_reference = sbert_model.encode(reference, convert_to_tensor=True)
    emb_essay = sbert_model.encode(essay, convert_to_tensor=True)
    similarity = util.cos_sim(emb_reference, emb_essay).item()
    
    # Enhanced content scoring with multiple criteria
    content_score = 0
    
    # Basic topic relevance (0-1 point)
    if similarity >= 0.60:
        content_score += 1
    
    # Check for argument structure (0-1 point)
    argument_indicators = ['firstly', 'secondly', 'thirdly', 'on the one hand', 'on the other hand', 
                          'however', 'nevertheless', 'in contrast', 'similarly', 'likewise',
                          'for example', 'for instance', 'such as', 'specifically']
    argument_count = sum(1 for indicator in argument_indicators if indicator.lower() in essay.lower())
    
    if argument_count >= 3:
        content_score += 1
    
    # Check for conclusion (0-1 point)
    conclusion_indicators = ['in conclusion', 'to conclude', 'therefore', 'thus', 'hence', 
                           'as a result', 'consequently', 'overall', 'in summary']
    has_conclusion = any(indicator.lower() in essay.lower() for indicator in conclusion_indicators)
    
    if has_conclusion:
        content_score += 1
    
    scores['content'] = content_score
    
    # === 2. Form (0-2 points) ===
    words = essay.strip().split()
    word_count = len(words)
    is_all_caps = essay.isupper()
    has_punctuation = bool(re.search(r'[.!?]', essay))
    has_bullets = bool(re.search(r'^[\s]*[•\-\*]', essay, re.MULTILINE))
    
    if 200 <= word_count <= 300 and not is_all_caps and has_punctuation and not has_bullets:
        scores['form'] = 2
    elif (120 <= word_count <= 199 or 301 <= word_count <= 380) and not is_all_caps and has_punctuation:
        scores['form'] = 1
    else:
        scores['form'] = 0
    
    # === 3. Development, Structure & Coherence (0-2 points) ===
    sentences = sent_tokenize(essay)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Check for paragraph structure (basic)
    paragraphs = essay.split('\n\n')
    paragraph_count = len([p for p in paragraphs if p.strip()])
    
    # Check for logical connectors
    connectors = ['however', 'therefore', 'furthermore', 'moreover', 'in addition', 
                 'consequently', 'as a result', 'on the other hand', 'nevertheless',
                 'firstly', 'secondly', 'finally', 'in conclusion', 'to summarize']
    connector_count = sum(1 for connector in connectors if connector.lower() in essay.lower())
    
    # Coherence scoring
    if sentence_count >= 8 and avg_sentence_length >= 15 and paragraph_count >= 2 and connector_count >= 2:
        scores['development_structure_coherence'] = 2
    elif sentence_count >= 5 and avg_sentence_length >= 12 and paragraph_count >= 1 and connector_count >= 1:
        scores['development_structure_coherence'] = 1
    else:
        scores['development_structure_coherence'] = 0
    
    # === 4. Grammar (0-2 points) ===
    try:
        matches = lang_tool.check(essay)
        grammar_errors = [m for m in matches if m.ruleIssueType in ("grammar", "typographical")]
        num_grammar_errors = len(grammar_errors)
        
        # Base grammar score
        if num_grammar_errors == 0:
            base_grammar_score = 2
        elif num_grammar_errors == 1:
            base_grammar_score = 1
        else:
            base_grammar_score = 0
    except Exception as e:
        print(f"Grammar check error: {e}")
        base_grammar_score = 2  # Default to perfect if tool fails
        num_grammar_errors = 0
    
    # === 5. General Linguistic Range (0-2 points) ===
    # Analyze sentence complexity and variety
    complex_sentences = 0
    for sentence in sentences:
        # Count clauses (basic complexity measure)
        clause_indicators = ['because', 'although', 'while', 'when', 'if', 'unless', 'since', 'as']
        if any(indicator in sentence.lower() for indicator in clause_indicators):
            complex_sentences += 1
    
    # Check for passive voice usage
    passive_patterns = [r'\b(am|is|are|was|were|be|been|being)\s+\w+ed\b', 
                       r'\b(has|have|had)\s+been\s+\w+ing\b']
    passive_count = sum(len(re.findall(pattern, essay, re.IGNORECASE)) for pattern in passive_patterns)
    
    # Linguistic range scoring
    complexity_ratio = complex_sentences / sentence_count if sentence_count > 0 else 0
    passive_ratio = passive_count / sentence_count if sentence_count > 0 else 0
    
    if complexity_ratio >= 0.4 and passive_ratio >= 0.1 and sentence_count >= 8:
        scores['general_linguistic_range'] = 2
    elif complexity_ratio >= 0.2 and sentence_count >= 5:
        scores['general_linguistic_range'] = 1
    else:
        scores['general_linguistic_range'] = 0
    
    # === 6. Vocabulary Range (0-2 points) ===
    lex = LexicalRichness(essay)
    ttr = lex.ttr
    mtld = lex.mtld()
    
    # Check for advanced vocabulary (words with 6+ letters)
    advanced_words = [word for word in words if len(word) >= 6 and word.isalpha()]
    advanced_ratio = len(advanced_words) / word_count if word_count > 0 else 0
    
    # Check for academic vocabulary patterns
    academic_suffixes = ['tion', 'ment', 'ness', 'ity', 'ism', 'ize', 'ify', 'ous', 'ive', 'ent', 'ant']
    academic_words = [word for word in words if any(word.endswith(suffix) for suffix in academic_suffixes)]
    academic_ratio = len(academic_words) / word_count if word_count > 0 else 0
    
    # Base vocabulary score
    if ttr > 0.75 and mtld > 25 and advanced_ratio >= 0.3 and academic_ratio >= 0.1:
        base_vocabulary_score = 2
    elif ttr > 0.65 and mtld > 20 and advanced_ratio >= 0.2:
        base_vocabulary_score = 1
    else:
        base_vocabulary_score = 0
    
    # Apply grammar penalty to vocabulary (for 5+ grammar errors)
    vocabulary_penalty = 0
    if num_grammar_errors >= 5:
        vocabulary_penalty = min(num_grammar_errors - 4, base_vocabulary_score)
    
    scores['vocabulary_range'] = max(0, base_vocabulary_score - vocabulary_penalty)
    
    # === 7. Spelling (0-2 points) ===
    try:
        spelling_errors = [m for m in matches if m.ruleIssueType in ("spelling", "misspelling")]
        num_spelling_errors = len(spelling_errors)
        
        # Base spelling score
        if num_spelling_errors == 0:
            base_spelling_score = 2
        elif num_spelling_errors == 1:
            base_spelling_score = 1
        else:
            base_spelling_score = 0
    except Exception as e:
        print(f"Spelling check error: {e}")
        base_spelling_score = 2  # Default to perfect if tool fails
        num_spelling_errors = 0
    
    # === Apply Cascading Penalties ===
    # Grammar penalties affecting spelling and vocabulary
    grammar_spelling_penalty = 0
    grammar_vocabulary_penalty = 0
    if num_grammar_errors >= 3:
        grammar_spelling_penalty = min(num_grammar_errors - 2, base_spelling_score)
    if num_grammar_errors >= 5:
        grammar_vocabulary_penalty = min(num_grammar_errors - 4, base_vocabulary_score)
    
    # Spelling penalties affecting grammar and vocabulary
    spelling_grammar_penalty = 0
    spelling_vocabulary_penalty = 0
    if num_spelling_errors >= 3:
        spelling_grammar_penalty = min(num_spelling_errors - 2, base_grammar_score)
    if num_spelling_errors >= 5:
        spelling_vocabulary_penalty = min(num_spelling_errors - 4, base_vocabulary_score)
    
    # Apply final scores with penalties
    scores['grammar'] = max(0, base_grammar_score - spelling_grammar_penalty)
    scores['spelling'] = max(0, base_spelling_score - grammar_spelling_penalty)
    scores['vocabulary_range'] = max(0, base_vocabulary_score - grammar_vocabulary_penalty - spelling_vocabulary_penalty)
    
    # === Word Highlights for Grammar/Spelling Errors ===
    word_highlights = []
    essay_words = essay.split()
    
    # Initialize all words as correct
    for word in essay_words:
        word_highlights.append({"word": word, "status": "correct", "replacement": None})
    
    # Process grammar errors
    for error in grammar_errors:
        error_text = essay[error.offset:error.offset+error.errorLength]
        error_words = error_text.split()
        
        for i, highlight in enumerate(word_highlights):
            clean_highlight = highlight['word'].rstrip('.,!?;:')
            if clean_highlight in error_words or highlight['word'] in error_words:
                word_highlights[i] = {
                    "word": highlight['word'],
                    "status": "grammar",
                    "replacement": error.replacements[0] if error.replacements else None,
                }
    
    # Process spelling errors
    for error in spelling_errors:
        error_text = essay[error.offset:error.offset+error.errorLength]
        error_words = error_text.split()
        suggestions = error.replacements[:3] if error.replacements else []
        
        for i, highlight in enumerate(word_highlights):
            clean_highlight = highlight['word'].rstrip('.,!?;:')
            if clean_highlight in error_words or highlight['word'] in error_words:
                word_highlights[i] = {
                    "word": highlight['word'],
                    "status": "spelling",
                    "suggestions": suggestions,
                }
    
    # === Final Total ===
    total = sum(scores.values())
    scores['total'] = total
    
    # Add details for API response
    scores['details'] = {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'paragraph_count': paragraph_count,
        'similarity': similarity,
        'content_analysis': {
            'topic_relevance_score': 1 if similarity >= 0.60 else 0,
            'argument_structure_score': 1 if argument_count >= 3 else 0,
            'conclusion_score': 1 if has_conclusion else 0,
            'argument_indicators_found': argument_count,
            'conclusion_indicators_found': sum(1 for indicator in conclusion_indicators if indicator.lower() in essay.lower())
        },
        'grammar_errors': [err.message for err in grammar_errors],
        'spelling_errors': [err.message for err in spelling_errors],
        'ttr': ttr,
        'mtld': mtld,
        'complexity_ratio': complexity_ratio,
        'passive_ratio': passive_ratio,
        'advanced_ratio': advanced_ratio,
        'academic_ratio': academic_ratio,
        'connector_count': connector_count,
        'cascading_penalties': {
            'num_grammar_errors': num_grammar_errors,
            'num_spelling_errors': num_spelling_errors,
            'grammar_spelling_penalty': grammar_spelling_penalty,
            'grammar_vocabulary_penalty': grammar_vocabulary_penalty,
            'spelling_grammar_penalty': spelling_grammar_penalty,
            'spelling_vocabulary_penalty': spelling_vocabulary_penalty,
            'base_scores': {
                'base_grammar_score': base_grammar_score,
                'base_spelling_score': base_spelling_score,
                'base_vocabulary_score': base_vocabulary_score
            }
        },
        'form_issues': {
            'is_all_caps': is_all_caps,
            'has_punctuation': has_punctuation,
            'has_bullets': has_bullets
        }
    }
    
    scores['word_highlights'] = word_highlights
    
    return scores 