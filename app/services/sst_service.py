from sentence_transformers import SentenceTransformer, util
import language_tool_python
import re
import math
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Load models/tools once
sbert_model = SentenceTransformer('all-mpnet-base-v2')
lang_tool = language_tool_python.LanguageTool('en-US')

def cefr_level(word):
    """
    Simplified CEFR level detection - you can replace this with vocab_level.cefr_level
    """
    # Basic CEFR level mapping (simplified)
    a1_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}
    a2_words = {'good', 'bad', 'big', 'small', 'new', 'old', 'young', 'hot', 'cold', 'happy', 'sad', 'easy', 'hard', 'fast', 'slow', 'high', 'low', 'long', 'short', 'right', 'wrong', 'same', 'different', 'first', 'last', 'next', 'last', 'many', 'much', 'few', 'little', 'some', 'any', 'all', 'every', 'each', 'other', 'another'}
    b1_words = {'important', 'necessary', 'possible', 'impossible', 'difficult', 'simple', 'complex', 'basic', 'advanced', 'modern', 'traditional', 'personal', 'public', 'private', 'national', 'international', 'local', 'global', 'economic', 'social', 'political', 'cultural', 'environmental', 'educational', 'professional', 'commercial', 'industrial', 'agricultural'}
    b2_words = {'significant', 'substantial', 'considerable', 'remarkable', 'notable', 'distinctive', 'characteristic', 'representative', 'typical', 'conventional', 'contemporary', 'innovative', 'revolutionary', 'fundamental', 'essential', 'crucial', 'critical', 'vital', 'indispensable', 'comprehensive', 'thorough', 'extensive', 'intensive', 'systematic', 'methodical', 'analytical', 'theoretical', 'practical', 'empirical', 'experimental'}
    c1_words = {'sophisticated', 'elaborate', 'intricate', 'nuanced', 'subtle', 'profound', 'profound', 'comprehensive', 'exhaustive', 'meticulous', 'rigorous', 'methodological', 'conceptual', 'theoretical', 'philosophical', 'ideological', 'paradigmatic', 'epistemological', 'ontological', 'phenomenological', 'hermeneutic', 'dialectical', 'heuristic', 'algorithmic', 'stochastic', 'probabilistic', 'deterministic', 'systemic', 'holistic', 'integrative'}
    c2_words = {'esoteric', 'arcane', 'abstruse', 'recondite', 'cryptic', 'enigmatic', 'paradoxical', 'oxymoronic', 'tautological', 'redundant', 'superfluous', 'extraneous', 'tangential', 'peripheral', 'marginal', 'negligible', 'infinitesimal', 'minuscule', 'microscopic', 'macroscopic', 'cosmic', 'universal', 'omnipresent', 'ubiquitous', 'pervasive', 'permeating', 'saturating', 'infiltrating', 'penetrating', 'percolating'}
    
    word_lower = word.lower()
    if word_lower in a1_words:
        return "A1"
    elif word_lower in a2_words:
        return "A2"
    elif word_lower in b1_words:
        return "B1"
    elif word_lower in b2_words:
        return "B2"
    elif word_lower in c1_words:
        return "C1"
    elif word_lower in c2_words:
        return "C2"
    else:
        return None

def word_frequency(word, lang='en'):
    """
    Simplified word frequency - you can replace this with wordfreq.word_frequency
    """
    # Basic frequency mapping (simplified)
    high_freq = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me'}
    mid_freq = {'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'}
    low_freq = {'important', 'necessary', 'possible', 'difficult', 'simple', 'complex', 'basic', 'advanced', 'modern', 'traditional', 'personal', 'public', 'private', 'national', 'international', 'local', 'global', 'economic', 'social', 'political', 'cultural', 'environmental', 'educational', 'professional', 'commercial', 'industrial', 'agricultural'}
    
    word_lower = word.lower()
    if word_lower in high_freq:
        return 0.001  # High frequency
    elif word_lower in mid_freq:
        return 0.0001  # Mid frequency
    elif word_lower in low_freq:
        return 0.00001  # Low frequency
    else:
        return 0.000001  # Rare

def evaluate_sst_vocabulary(text):
    """
    Advanced vocabulary scoring for SST (0-2 points)
    """
    # Vocabulary Lists
    academic_suffixes = [
        'tion', 'ment', 'ness', 'ity', 'ism', 'ize', 'ify', 'ous', 'ive', 
        'ent', 'ant', 'ary', 'ory', 'ate', 'ure', 'ence', 'ance', 'able', 'ible'
    ]
    
    academic_prefixes = [
        'pre', 'pro', 'anti', 'inter', 'trans', 'sub', 'super', 'multi', 
        'over', 'under', 'semi', 'non', 'dis', 'mis', 'un', 're', 'de', 'ex'
    ]
    
    informal_words = {
        'really', 'very', 'pretty', 'quite', 'totally', 'absolutely', 'definitely',
        'obviously', 'clearly', 'basically', 'actually', 'literally', 'seriously',
        'huge', 'tiny', 'loads', 'tons', 'stuff', 'things', 'guy', 'guys', 'ok', 'okay',
        'awesome', 'cool', 'super', 'mega', 'ultra', 'big', 'small', 'lots', 'bunch'
    }
    
    vague_words = {
        'thing', 'stuff', 'something', 'someone', 'somewhere', 'somehow', 'various',
        'different', 'certain', 'some', 'many', 'several', 'numerous', 'lots', 'bunch',
        'kind', 'sort', 'type', 'way', 'part', 'area', 'aspect', 'factor'
    }
    
    contractions = {
        "won't", "can't", "shouldn't", "wouldn't", "couldn't", "didn't", "doesn't",
        "don't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
        "i'm", "you're", "he's", "she's", "it's", "we're", "they're", "i've", "you've",
        "we've", "they've", "i'll", "you'll", "he'll", "she'll", "it'll", "we'll", "they'll"
    }
    
    if not text:
        return 0
    
    # Preprocess
    original_text = text.lower()
    tokens = word_tokenize(original_text)
    words = [w for w in tokens if w.isalpha() and len(w) > 1]
    
    if len(words) == 0:
        return 0
    
    total_words = len(words)
    
    # CEFR Level Profile
    cefr_profile = Counter()
    for word in words:
        level = cefr_level(word)
        if level:
            cefr_profile[level] += 1
        else:
            cefr_profile["None"] += 1
    
    # Frequency Profile
    freq_profile = {"High_Freq": 0, "Mid_Freq": 0, "Low_Freq": 0, "Rare": 0}
    for word in words:
        freq = word_frequency(word, 'en')
        if freq > 0.0001:
            freq_profile["High_Freq"] += 1
        elif freq > 0.00005:
            freq_profile["Mid_Freq"] += 1
        elif freq > 0.00001:
            freq_profile["Low_Freq"] += 1
        else:
            freq_profile["Rare"] += 1
    
    # Academic Vocabulary Analysis
    academic_words = []
    for word in words:
        if (len(word) >= 6 and 
            (any(word.endswith(suffix) for suffix in academic_suffixes) or
             any(word.startswith(prefix) for prefix in academic_prefixes))):
            academic_words.append(word)
    
    academic_ratio = len(academic_words) / total_words
    
    # Inappropriate Vocabulary Detection
    inappropriate_issues = {
        'informal': [w for w in words if w in informal_words],
        'vague': [w for w in words if w in vague_words],
        'contractions': [w for w in tokens if w in contractions]
    }
    
    total_inappropriate = sum(len(issues) for issues in inappropriate_issues.values())
    
    # Lexical Sophistication Analysis
    sophisticated_words = [w for w in words if len(w) >= 6]
    sophistication_ratio = len(sophisticated_words) / total_words
    
    # Lexical Diversity
    unique_words = len(set(words))
    ttr = unique_words / total_words
    
    # Advanced Metrics
    b2_c_words = cefr_profile["B2"] + cefr_profile["C1"] + cefr_profile["C2"]
    a1_words = cefr_profile["A1"]
    rare_ratio = freq_profile["Rare"] / total_words
    a1_ratio = a1_words / total_words
    high_vocab_ratio = b2_c_words / total_words
    
    # Repetition Analysis
    word_counts = Counter(words)
    overused_words = {word: count for word, count in word_counts.items() 
                     if count > 3 and len(word) > 3}
    
    # Scoring Logic
    base_score = 2
    
    # Major penalties
    if total_inappropriate >= 3:
        base_score -= 1
    elif total_inappropriate >= 5:
        base_score -= 2
    
    # Academic vocabulary requirements
    if academic_ratio < 0.05:
        base_score -= 1
    
    # Sophistication requirements
    if sophistication_ratio < 0.25:
        base_score -= 0.5
    
    # Basic vocabulary over-reliance
    if a1_ratio > 0.6:
        base_score -= 1
    
    # Repetition penalty
    if len(overused_words) > 2:
        base_score -= 0.5
    
    # Lexical diversity penalty
    if ttr < 0.6:
        base_score -= 0.5
    
    # Positive adjustments
    if high_vocab_ratio >= 0.15 and academic_ratio >= 0.10:
        base_score = min(2, base_score + 0.5)
    
    if rare_ratio >= 0.12 and total_inappropriate == 0:
        base_score = min(2, base_score + 0.5)
    
    # Final score
    final_score = max(0, min(2, round(base_score * 2) / 2))
    
    if final_score == 1.5:
        final_score = 1
    
    return int(final_score)

def evaluate_sst_service(summary, reference):
    """
    Evaluate SST (Summarize Spoken Text) across 5 criteria
    """
    scores = {}
    
    # === 1. Content (2 points) ===
    emb_ref = sbert_model.encode(reference, convert_to_tensor=True)
    emb_sum = sbert_model.encode(summary, convert_to_tensor=True)
    similarity = util.cos_sim(emb_ref, emb_sum).item()
    
    if similarity >= 0.85:
        scores['content'] = 2
    elif similarity >= 0.70:
        scores['content'] = 1
    else:
        scores['content'] = 0
    
    # === 2. Form (Length & Structure) (2 points) ===
    words = summary.strip().split()
    word_count = len(words)
    sentence_count = len(re.findall(r'[.!?]', summary.strip()))
    
    # Check for ALL CAPS
    is_all_caps = summary.isupper()
    
    # Check for punctuation
    has_punctuation = bool(re.search(r'[.!?]', summary))
    
    # Check for bullet points or fragments
    has_bullets = bool(re.search(r'^[\s]*[•\-\*]', summary, re.MULTILINE))
    has_fragments = len(re.findall(r'[.!?]', summary)) == 0 or any(len(sent.strip()) < 10 for sent in re.split(r'[.!?]', summary) if sent.strip())
    
    if (50 <= word_count <= 70 and sentence_count >= 1 and 
        not is_all_caps and has_punctuation and not has_bullets and not has_fragments):
        scores['form'] = 2
    elif ((40 <= word_count <= 49 or 71 <= word_count <= 100) and 
          sentence_count >= 1 and not is_all_caps and has_punctuation):
        scores['form'] = 1
    else:
        scores['form'] = 0
    
    # === 3. Grammar (2 points) ===
    try:
        matches = lang_tool.check(summary)
        print(f"LanguageTool found {len(matches)} total matches")
        for match in matches:
            print(f"Error: {match.ruleIssueType} - {match.message} at position {match.offset}")
        
        grammar_errors = [m for m in matches if m.ruleIssueType in ("grammar", "typographical")]
        num_errors = len(grammar_errors)
        print(f"Grammar errors: {num_errors}")
        
        if num_errors == 0:
            scores['grammar'] = 2
        elif num_errors <= 2:
            scores['grammar'] = 1
        else:
            scores['grammar'] = 0
    except Exception as e:
        print(f"Grammar check error: {e}")
        scores['grammar'] = 2  # Default to perfect if tool fails
    
    # === 4. Vocabulary (2 points) ===
    scores['vocabulary'] = evaluate_sst_vocabulary(summary)
    
    # === 5. Spelling (2 points) ===
    try:
        spelling_errors = [m for m in matches if m.ruleIssueType in ("spelling", "misspelling")]
        num_spelling_errors = len(spelling_errors)
        print(f"Spelling errors: {num_spelling_errors}")
        
        if num_spelling_errors == 0:
            scores['spelling'] = 2
        elif num_spelling_errors == 1:
            scores['spelling'] = 1
        else:
            scores['spelling'] = 0  # Changed back to 0 for 2+ spelling errors
    except Exception as e:
        print(f"Spelling check error: {e}")
        scores['spelling'] = 2  # Default to perfect if tool fails
    
    # === Word Highlights for Grammar/Spelling Errors ===
    word_highlights = []
    summary_words = summary.split()
    
    # Initialize all words as correct
    for word in summary_words:
        word_highlights.append({"word": word, "status": "correct", "replacement": None})
    
    # Process grammar errors
    for error in grammar_errors:
        error_text = summary[error.offset:error.offset+error.errorLength]
        error_words = error_text.split()
        
        # Find matching words in highlights
        for i, highlight in enumerate(word_highlights):
            if highlight['word'] in error_words:
                word_highlights[i] = {
                    "word": highlight['word'],
                    "status": "grammar",
                    "replacement": error.replacements[0] if error.replacements else None,
                }
                break  # Only highlight the first occurrence
    
    # Process spelling errors
    for error in spelling_errors:
        error_text = summary[error.offset:error.offset+error.errorLength]
        error_words = error_text.split()
        suggestions = error.replacements[:3] if error.replacements else []
        # Find matching words in highlights
        for i, highlight in enumerate(word_highlights):
            if highlight['word'] in error_words:
                word_highlights[i] = {
                    "word": highlight['word'],
                    "status": "spelling",
                    "suggestions": suggestions,
                }
                break  # Only highlight the first occurrence
    
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
        'form_issues': {
            'is_all_caps': is_all_caps,
            'has_punctuation': has_punctuation,
            'has_bullets': has_bullets,
            'has_fragments': has_fragments
        }
    }
    
    scores['word_highlights'] = word_highlights
    
    # Debug information
    print(f"Final scores: {scores}")
    print(f"Total score: {total}")
    
    return scores

def test_grammar_spelling():
    """
    Test function to verify grammar and spelling detection
    """
    test_text = "I goes to the store yesterday and buyed some apples. The weather was beautifull."
    print(f"Testing text: {test_text}")
    
    try:
        matches = lang_tool.check(test_text)
        print(f"Found {len(matches)} errors:")
        for match in matches:
            print(f"- {match.ruleIssueType}: {match.message}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_grammar_spelling() 