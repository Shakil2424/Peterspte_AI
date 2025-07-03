import difflib
import re
from collections import Counter

def normalize_apostrophes(text):
    return text.replace("’", "'").replace("`", "'")

def normalize_word(word, strip_period=True):
    word = word.lower().replace("’", "'").replace("`", "'").strip()
    if strip_period:
        word = word.rstrip(".")
    return re.sub(r"[^\w']+", "", word)

def dictation_highlight(reference_text, user_speech):
    reference_words = reference_text.split()
    user_speech_words = user_speech.split()
    word_highlight = []
    used_ref_indices = set()
    for user_word in user_speech_words:
        normalized_user = normalize_word(user_word)
        best_match = None
        best_similarity = 0
        best_idx = None
        for ref_idx, ref_word in enumerate(reference_words):
            if ref_idx not in used_ref_indices and normalized_user == normalize_word(ref_word):
                best_match = "correct"
                best_idx = ref_idx
                break
        if best_match is None:
            for ref_idx, ref_word in enumerate(reference_words):
                if ref_idx not in used_ref_indices:
                    similarity = difflib.SequenceMatcher(None, normalize_word(ref_word), normalized_user).ratio()
                    if similarity > best_similarity and similarity >= 0.75:
                        best_match = "misspelled"
                        best_similarity = similarity
                        best_idx = ref_idx
        if best_match is not None:
            used_ref_indices.add(best_idx)
            word_highlight.append((user_word, best_match))
        else:
            word_highlight.append((user_word, "extra"))
    for ref_idx, ref_word in enumerate(reference_words):
        if ref_idx not in used_ref_indices:
            word_highlight.append((ref_word, "missing"))
    return word_highlight

def dictation_ai(user_text, reference_text):
    reference_text = normalize_apostrophes(reference_text)
    user_text = normalize_apostrophes(user_text)
    ref_words = reference_text.split()
    user_words = user_text.split()
    total_score = len(ref_words)
    user_score = total_score
    matching_words = []
    word_highlights = []
    if user_words and not user_words[0][0].isupper():
        user_score -= 0.5
    if user_words and not user_words[-1].endswith('.'):
        user_score -= 0.5
    ref_word_count = {}
    ref_word_positions = []
    for i, word in enumerate(ref_words):
        norm_word = normalize_word(word)
        ref_word_count[norm_word] = ref_word_count.get(norm_word, 0) + 1
        ref_word_positions.append((norm_word, word, i))
    matched_count = {}
    for word in user_words:
        norm_word = normalize_word(word)
        matched = False
        for ref_norm in ref_word_count:
            if norm_word == ref_norm:
                current_matches = matched_count.get(ref_norm, 0)
                if current_matches < ref_word_count[ref_norm]:
                    matching_words.append(norm_word)
                    word_highlights.append([word, "correct"])
                    matched_count[ref_norm] = current_matches + 1
                    matched = True
                    break
        if not matched:
            word_highlights.append([word, "extra"])
    result_highlights = []
    current_user_pos = 0
    while current_user_pos < len(word_highlights) and word_highlights[current_user_pos][1] != "correct":
        result_highlights.append(word_highlights[current_user_pos])
        current_user_pos += 1
    for ref_norm, ref_orig, ref_pos in ref_word_positions:
        if ref_norm in matched_count and matched_count[ref_norm] > 0:
            while current_user_pos < len(word_highlights) and normalize_word(word_highlights[current_user_pos][0]) != ref_norm:
                result_highlights.append(word_highlights[current_user_pos])
                current_user_pos += 1
            if current_user_pos < len(word_highlights):
                result_highlights.append(word_highlights[current_user_pos])
                matched_count[ref_norm] -= 1
                current_user_pos += 1
        else:
            result_highlights.append([ref_orig, "missing"])
    while current_user_pos < len(word_highlights):
        result_highlights.append(word_highlights[current_user_pos])
        current_user_pos += 1
    missing_count = sum(1 for w in result_highlights if w[1] == "missing")
    user_score = min(max(total_score - missing_count, 0), total_score)
    if user_words and not user_words[0][0].isupper():
        user_score = max(0, user_score - 0.5)
    if user_words and not user_words[-1].endswith('.'):
        user_score = max(0, user_score - 0.5)
    # listening_writing_score = (90 / total_score) * user_score if total_score > 0 else 0
    return {
        "score": {
            "matching_words": matching_words,
        
            "word_highlights": result_highlights,
            # "overall": total_score,
            "listening": user_score,
            "writing": user_score,
            "score": user_score,
            "max_score": total_score,
            "summary": user_text
        },
        # "overall": total_score,
        # "max_score": 90
    } 