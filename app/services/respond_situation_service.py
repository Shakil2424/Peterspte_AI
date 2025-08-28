import os
import numpy as np
import librosa
import torch
from sentence_transformers import SentenceTransformer, util
import nltk
import re
from app.services.audio_transcriber import transcribe_audio

nltk.download('punkt', quiet=True)

# Load models
semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

# --- Content Scoring ---
def semantic_similarity(reference: str, response: str) -> float:
    ref_emb = semantic_model.encode(reference, convert_to_tensor=True)
    resp_emb = semantic_model.encode(response, convert_to_tensor=True)
    return util.pytorch_cos_sim(ref_emb, resp_emb).item()

def extract_keywords(text: str, keywords: list) -> list:
    found = []
    text = text.lower()
    for kw in keywords:
        if kw in text:
            found.append(kw)
    return found

def score_content(reference: str, response: str) -> int:
    similarity = semantic_similarity(reference, response)
    word_count = len(nltk.word_tokenize(response))
    coverage = extract_keywords(response, ['unprepared', 'go later', 'someone else', 'not ready', 'delay', 'present first'])
    if similarity > 0.85 and word_count > 40 and len(coverage) >= 2:
        raw = 90
    elif similarity > 0.75 and word_count > 30 and len(coverage) >= 2:
        raw = 75
    elif similarity > 0.65 and word_count > 25 and len(coverage) >= 1:
        raw = 60
    elif similarity > 0.55 and word_count > 20:
        raw = 45
    elif similarity > 0.4:
        raw = 30
    elif similarity > 0.2:
        raw = 15
    else:
        raw = 10
    return max(10, min(90, raw))

# --- Pronunciation Scoring ---
def count_syllables(text):
    return sum(len(re.findall(r'[aeiouy]+', word)) for word in text.split())

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

def score_pronunciation(transcript, audio, sr, duration_sec):
    syllables_asr = count_syllables(transcript)
    speech_rate = syllables_asr / duration_sec if duration_sec > 0 else 0
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    median_mag = np.median(magnitudes)
    voiced_pitches = pitches[magnitudes > median_mag]
    voiced_pitches = voiced_pitches[(voiced_pitches > 50) & (voiced_pitches < 500)]
    intonation_std = np.std(voiced_pitches) if len(voiced_pitches) > 0 else 0
    rubric_level = rubric_score_ref_free(transcript, syllables_asr, intonation_std, speech_rate)
    fluency_score = scale_pronunciation(speech_rate)
    intonation_score = min(90, intonation_std * 2)
    syllable_score = min(90, (syllables_asr / (duration_sec + 1e-5)) * 10)
    average_raw = (fluency_score + intonation_score + syllable_score) / 3
    final_score = max(10, min(90, round((average_raw / 100) * 90, 2)))
    return final_score

# --- Fluency Scoring ---
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

# --- Content-Based Penalty System ---
def calculate_content_penalty(content_score):
    """Calculate penalty multiplier for pronunciation and fluency based on content score"""
    if content_score >= 80:
        return 1.0  # No penalty
    elif content_score >= 60:
        return 0.9  # 10% penalty
    elif content_score >= 40:
        return 0.7  # 30% penalty
    elif content_score >= 20:
        return 0.5  # 50% penalty
    else:
        return 0.3  # 70% penalty

def apply_content_penalty(original_score, penalty_multiplier, min_score=10):
    """Apply penalty to a score while maintaining minimum threshold"""
    penalized_score = original_score * penalty_multiplier
    return max(min_score, round(penalized_score, 2))

# --- Main Respond Situation Scoring Function ---
def evaluate_respond_situation(reference_text: str, file, upload_folder: str):
    # Use existing transcribe_audio logic
    result, status_code = transcribe_audio(file, upload_folder)
    if status_code != 200:
        return {'error': 'Transcription failed', 'details': result}, status_code
    transcript = result.get('transcript', '').strip()
    # Save file to temp for librosa analysis
    file.seek(0)
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        audio, sr = librosa.load(tmp_path, sr=16000)
        duration_sec = librosa.get_duration(y=audio, sr=sr)
        # Content
        content_score = score_content(reference_text, transcript)
        # Pronunciation
        pronunciation_score = score_pronunciation(transcript, audio, sr, duration_sec)
        # Fluency
        fluency_score = score_fluency(transcript, audio, sr, duration_sec)
        
        # === ENHANCED CONTENT-BASED PENALTY SYSTEM ===
        # Calculate penalty multiplier based on content performance
        penalty_multiplier = calculate_content_penalty(content_score)
        
        # Apply penalties to pronunciation and fluency scores
        original_pronunciation = pronunciation_score
        original_fluency = fluency_score
        
        pronunciation_score = apply_content_penalty(pronunciation_score, penalty_multiplier)
        fluency_score = apply_content_penalty(fluency_score, penalty_multiplier)
        
        # === SPECIAL CASE: IF CONTENT IS 10, SET ALL SCORES TO 10 ===
        if content_score <= 10:
            pronunciation_score = 10
            fluency_score = 10
        
        # === SPEAKING AND LISTENING SCORES ===
        speaking = ((fluency_score * 80) / 100) + ((pronunciation_score * 20) / 100)
        listening = ((content_score * 80) / 100) + ((pronunciation_score * 20) / 100)
        
        # Add penalty information to the response
        penalty_info = {
            "penalty_multiplier": penalty_multiplier,
            "penalty_percentage": round((1 - penalty_multiplier) * 100, 1),
            "original_pronunciation": original_pronunciation,
            "original_fluency": original_fluency,
            "penalized_pronunciation": pronunciation_score,
            "penalized_fluency": fluency_score
        }
    finally:
        os.remove(tmp_path)
    return {
        'transcription': transcript,
        'content_score': content_score,
        'pronunciation_score': pronunciation_score,
        'fluency_score': fluency_score,
        'speaking_score': float(round(speaking, 2)),
        'listening_score': float(round(listening, 2)),
        'penalty_info': penalty_info
    }, 200 