import nltk
from difflib import SequenceMatcher
import re
import librosa
import numpy as np
import os
import difflib
from jiwer import wer
from app.services.audio_transcriber import transcribe_audio

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def tokenize(sentence):
    """Tokenize a sentence into lowercase words."""
    return nltk.word_tokenize(sentence.lower())

def calculate_sequence_match_ratio(reference_tokens, response_tokens):
    """Calculate the ratio of matching tokens between reference and response."""
    matcher = SequenceMatcher(None, reference_tokens, response_tokens)
    match_blocks = matcher.get_matching_blocks()
    match_count = sum(block.size for block in match_blocks)
    return match_count / len(reference_tokens) if reference_tokens else 0

def content_score(reference, response):
    """Content scoring using the same logic as read aloud"""
    # Direct word-by-word comparison with reference text
    ref_words = reference.lower().split()
    res_words = response.lower().split()
    
    matcher = difflib.SequenceMatcher(None, ref_words, res_words)
    correct_words = 0
    total_words = len(ref_words)
    word_highlights = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            correct_words += (i2 - i1)
            # Add correct words to highlights
            for i in range(i1, i2):
                word_highlights.append({
                    "word": ref_words[i],
                    "status": "correct",
                    "replacement": None
                })
        elif tag == 'replace':
            # Add replaced words (reference words marked as incorrect)
            for i in range(i1, i2):
                word_highlights.append({
                    "word": ref_words[i],
                    "status": "incorrect",
                    "replacement": res_words[j1] if j1 < len(res_words) else None
                })
        elif tag == 'delete':
            # Add missing words
            for i in range(i1, i2):
                word_highlights.append({
                    "word": ref_words[i],
                    "status": "missing",
                    "replacement": None
                })
        elif tag == 'insert':
            # Add extra words (response words not in reference)
            for j in range(j1, j2):
                word_highlights.append({
                    "word": res_words[j],
                    "status": "extra",
                    "replacement": None
                })
    
    # Word accuracy calculation
    word_accuracy = (correct_words / total_words) if total_words > 0 else 0
    
    # Normal scoring without 60% threshold
    score = max(10, min(90, round(word_accuracy * 90, 2)))
    
    return score, word_highlights

def count_syllables(text):
    return sum(len(re.findall(r'[aeiouy]+', word)) for word in text.split())

def rubric_score(wer_val, syllable_acc, inton_std):
    """Get rubric level and description"""
    if wer_val <= 0.05 and syllable_acc >= 95 and inton_std >= 30:
        return 5, "Highly Proficient – All sounds clear; natural stress and connected speech."
    elif wer_val <= 0.10 and syllable_acc >= 90:
        return 4, "Advanced – Minor errors in stress or consonants but fully intelligible."
    elif wer_val <= 0.20 and syllable_acc >= 80:
        return 3, "Good – Some words unclear; occasional distortion or misplaced stress."
    elif wer_val <= 0.35 and syllable_acc >= 60:
        return 2, "Intermediate – Frequent mispronunciations; listener must adapt to accent."
    elif wer_val <= 0.60:
        return 1, "Intrusive – Strong accent; stress unclear; many sounds distorted or dropped."
    else:
        return 0, "Non-English – Mostly unintelligible; stress and sounds non-native."

def rubric_score_ref_free(asr_text, syllables, intonation_std, speech_rate):
    word_count = len(asr_text.split())
    fluent = 2.5 <= speech_rate <= 4.0
    strong_intonation = intonation_std >= 30
    if word_count > 15 and syllables > 20 and fluent and strong_intonation:
        return 5
    elif word_count > 12 and syllables > 15 and intonation_std >= 20:
        return 4
    elif word_count > 9 and syllables > 12 and intonation_std >= 15:
        return 3
    elif word_count > 5 and syllables > 8 and intonation_std >= 10:
        return 2
    elif word_count > 2 and syllables > 4:
        return 1
    else:
        return 0

def scale_pronunciation(val):
    val = np.clip(val, 1.5, 5.0)
    return 10 + ((val - 1.5) / (5.0 - 1.5))**2 * 80

def score_pronunciation(transcript, audio, sr, duration_sec, reference_text):
    """Pronunciation scoring using the same logic as read aloud"""
    # Syllable analysis
    syllables_ref = count_syllables(reference_text)
    syllables_asr = count_syllables(transcript)
    syllable_error = abs(syllables_ref - syllables_asr)
    syllable_accuracy = max(0, 1 - (syllable_error / syllables_ref)) * 100 if syllables_ref > 0 else 0
    
    # Word Error Rate (WER)
    wer_value = wer(reference_text.lower(), transcript.lower())
    pron_accuracy_pct = max(0, (1 - wer_value)) * 100
    
    # Intonation analysis
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    median_mag = np.median(magnitudes)
    voiced_pitches = pitches[magnitudes > median_mag]
    voiced_pitches = voiced_pitches[(voiced_pitches > 50) & (voiced_pitches < 500)]
    intonation_std = np.std(voiced_pitches) if len(voiced_pitches) > 0 else 0
    
    # Rubric scoring
    rubric_level, rubric_desc = rubric_score(wer_value, syllable_accuracy, intonation_std)
    
    # Pronunciation score
    pronunciation_score = max(10, min(90, round((pron_accuracy_pct * 0.5 + syllable_accuracy * 0.3 + (rubric_level / 5) * 90 * 0.2), 2)))
    return pronunciation_score

def scale_fluency(val):
    val = np.clip(val, 1.0, 5.0)
    return 10 + ((val - 1.0) / (5.0 - 1.0))**1.5 * 80

def score_fluency(transcript, audio, sr, duration_sec):
    syllables_asr = count_syllables(transcript)
    speech_rate = syllables_asr / duration_sec if duration_sec > 0 else 0
    try:
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.1)
        median_mag = np.median(magnitudes[magnitudes > 0])
        voiced_pitches = pitches[magnitudes > median_mag * 0.5]
        voiced_pitches = voiced_pitches[(voiced_pitches > 50) & (voiced_pitches < 500)]
        intonation_std = np.std(voiced_pitches) if len(voiced_pitches) > 0 else 0
    except:
        intonation_std = 10
    fluency_score = scale_fluency(speech_rate)
    intonation_score = min(90, intonation_std * 2)
    syllable_score = min(90, (syllables_asr / (duration_sec + 1e-5)) * 15)
    average_raw = (fluency_score + intonation_score + syllable_score) / 3
    final_score = max(10, min(90, round(average_raw, 2)))
    return final_score

def evaluate_repeat_sentence(reference_text, file, upload_folder):
    result, status_code = transcribe_audio(file, upload_folder)
    if status_code != 200:
        return {'error': 'Transcription failed', 'details': result}, status_code
    transcript = result.get('transcript', '').strip()
    file.seek(0)
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        audio, sr = librosa.load(tmp_path, sr=16000)
        duration_sec = librosa.get_duration(y=audio, sr=sr)
        content, word_highlights = content_score(reference_text, transcript)
        pronunciation = score_pronunciation(transcript, audio, sr, duration_sec, reference_text)
        fluency = score_fluency(transcript, audio, sr, duration_sec)
        
        # === ADJUST SCORES BASED ON CONTENT ===
        # If content score is 10, set pronunciation and fluency to 10 as well
        if content <= 10:
            pronunciation = 10
            fluency = 10
    finally:
        os.remove(tmp_path)
    return {
        'transcription': transcript,
        'content_score': content,
        'pronunciation_score': pronunciation,
        'fluency_score': fluency,
        'word_highlights': word_highlights
    }, 200 