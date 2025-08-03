import whisperx
import librosa
import numpy as np
import nltk
from jiwer import wer
import difflib
import re
from sentence_transformers import SentenceTransformer, util

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load models
sbert_model = SentenceTransformer('all-mpnet-base-v2')

def count_syllables(text):
    """Count syllables in text"""
    return sum(len(re.findall(r'[aeiouy]+', word)) for word in text.split())

def scale_fluency(val):
    """Scale fluency value to 10-90 range"""
    val = np.clip(val, 1.0, 5.0)
    return 10 + ((val - 1.0) / (5.0 - 1.0))**1.5 * 80

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

def evaluate_read_aloud(audio_file, reference_text):
    """
    Evaluate read aloud performance
    Returns: content_score, pronunciation_score, fluency_score (10-90 range)
    """
    try:
        # Load WhisperX model
        model = whisperx.load_model("base", device="cpu", compute_type="int8")
        
        # Transcribe audio
        result = model.transcribe(audio_file, language="en")
        asr_text = " ".join([seg['text'] for seg in result['segments']]).strip().lower()
        
        # Clean reference text
        reference_text = reference_text.strip().lower()
        
        # === 1. CONTENT SCORING (10-90) ===
        # Direct word-by-word comparison with reference text
        ref_words = reference_text.split()
        asr_words = asr_text.split()
        
        matcher = difflib.SequenceMatcher(None, ref_words, asr_words)
        word_feedback = []
        correct_words = 0
        total_words = len(ref_words)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                correct_words += (i2 - i1)
                word_feedback.extend([(ref_words[i], "good") for i in range(i1, i2)])
            elif tag == 'replace':
                word_feedback.extend([(ref_words[i], "average") for i in range(i1, i2)])
            elif tag == 'delete':
                word_feedback.extend([(ref_words[i], "missing") for i in range(i1, i2)])
            elif tag == 'insert':
                word_feedback.extend([(asr_words[j], "extra") for j in range(j1, j2)])
        
        # Content score based on word accuracy only (direct matching)
        word_accuracy = (correct_words / total_words) if total_words > 0 else 0
        
        # If word accuracy is less than 60%, set content score to 10
        if word_accuracy < 0.60:
            content_score = 10
        else:
            content_score = max(10, min(90, round(word_accuracy * 90, 2)))
        
        # === 2. PRONUNCIATION SCORING (10-90) ===
        # Load audio for analysis
        audio, sr = librosa.load(audio_file, sr=16000)
        duration_sec = librosa.get_duration(y=audio, sr=sr)
        
        # Syllable analysis
        syllables_ref = count_syllables(reference_text)
        syllables_asr = count_syllables(asr_text)
        syllable_error = abs(syllables_ref - syllables_asr)
        syllable_accuracy = max(0, 1 - (syllable_error / syllables_ref)) * 100 if syllables_ref > 0 else 0
        
        # Word Error Rate (WER)
        wer_value = wer(reference_text, asr_text)
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
        
        # === 3. FLUENCY SCORING (10-90) ===
        # Speech rate
        speech_rate = syllables_asr / duration_sec if duration_sec > 0 else 0
        
        # Fluency score based on speech rate and intonation (same as repeat_sentence/retell_lecture)
        fluency_score = scale_fluency(speech_rate)
        intonation_score = min(90, intonation_std * 2)
        syllable_score = min(90, (syllables_asr / (duration_sec + 1e-5)) * 15)
        average_raw = (fluency_score + intonation_score + syllable_score) / 3
        final_fluency_score = max(10, min(90, round(average_raw, 2)))
        
        # === ADJUST SCORES BASED ON CONTENT ===
        # If content score is 10, set pronunciation and fluency to 10 as well
        if content_score <= 10:
            pronunciation_score = 10
            final_fluency_score = 10
        
        # === COMPOSITE SCORE ===
        composite_score = round((content_score + pronunciation_score + final_fluency_score) / 3, 2)
        
        # === SPEAKING AND READING SCORES ===
        # Calculate a pronunciation score for speaking calculation
        score = pronunciation_score  # Using pronunciation_score as the base score
        
        speaking = ((final_fluency_score * 80) / 100) + ((score * 20) / 100)
        reading = ((content_score * 80) / 100) + ((final_fluency_score * 20) / 100)
        
        return {
            'content_score': float(content_score),
            'pronunciation_score': float(pronunciation_score),
            'fluency_score': float(final_fluency_score),
            'composite_score': float(composite_score),
            'speaking_score': float(round(speaking, 2)),
            'reading_score': float(round(reading, 2)),
            'word_highlights': word_feedback,
            'details': {
                'transcribed_text': str(asr_text),
                'reference_text': str(reference_text),
                'word_accuracy': float(word_accuracy),
                'speech_rate': float(speech_rate),
                'wer_value': float(wer_value),
                'pron_accuracy_pct': float(pron_accuracy_pct),
                'syllable_accuracy': float(syllable_accuracy),
                'intonation_std': float(intonation_std),
                'rubric_level': int(rubric_level),
                'rubric_description': str(rubric_desc),
                'duration_seconds': float(duration_sec),
                'syllables_reference': int(syllables_ref),
                'syllables_asr': int(syllables_asr)
            }
        }
        
    except Exception as e:
        raise Exception(f"Read aloud evaluation failed: {str(e)}") 