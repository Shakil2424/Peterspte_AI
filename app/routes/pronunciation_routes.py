from flask import Blueprint, request, jsonify
import numpy as np
import librosa
import tempfile
import os
import re
from app.services.audio_transcriber import transcribe_audio

pronunciation_bp = Blueprint('pronunciation', __name__)

def count_syllables(text):
    return sum(len(re.findall(r'[aeiouy]+', word)) for word in text.split())

def scale_fluency(val):
    val = np.clip(val, 1.5, 5.0)
    return 10 + ((val - 1.5) / (5.0 - 1.5))**2 * 80

def rubric_score_ref_free(asr_text, syllables, intonation_std, speech_rate):
    word_count = len(asr_text.split())
    fluent = 2.5 <= speech_rate <= 4.0
    strong_intonation = intonation_std >= 30
    if word_count > 15 and syllables > 20 and fluent and strong_intonation:
        return 5, "Highly Proficient – All vowels and consonants are produced clearly. Natural assimilation/deletion and accurate stress."
    elif word_count > 12 and syllables > 15 and intonation_std >= 20:
        return 4, "Advanced – Minor distortions; good clarity. Stress mostly accurate; speech is clear and well-paced."
    elif word_count > 9 and syllables > 12 and intonation_std >= 15:
        return 3, "Good – Some unclear words or flat pitch. Stress errors or missing syllables occur occasionally."
    elif word_count > 5 and syllables > 8 and intonation_std >= 10:
        return 2, "Intermediate – Many mispronunciations. Accent affects clarity. Stress and sequences may be incorrect."
    elif word_count > 2 and syllables > 4:
        return 1, "Intrusive – Strong accent; stress unclear; many sounds distorted or dropped."
    else:
        return 0, "Non-English – Mostly unintelligible; many words mispronounced or omitted."

@pronunciation_bp.route('/pronunciation', methods=['POST'])
def pronunciation():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Use your existing transcription function
    transcript_result, status_code = transcribe_audio(file, 'uploads')
    if status_code != 200:
        return jsonify({'error': 'Transcription failed', 'details': transcript_result}), 500
    transcript = transcript_result.get('transcript', '').strip().lower()

    # Reset file pointer before saving again
    file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    # Load audio and compute duration
    audio, sr = librosa.load(tmp_path, sr=16000)
    duration_sec = librosa.get_duration(y=audio, sr=sr)

    # Syllable and word count
    syllables_asr = count_syllables(transcript)
    word_count = len(transcript.split())
    speech_rate = syllables_asr / duration_sec if duration_sec > 0 else 0

    # Intonation estimation (pitch variation)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    median_mag = np.median(magnitudes)
    voiced_pitches = pitches[magnitudes > median_mag]
    voiced_pitches = voiced_pitches[(voiced_pitches > 50) & (voiced_pitches < 500)]
    intonation_std = np.std(voiced_pitches) if len(voiced_pitches) > 0 else 0

    # Rubric and composite score
    rubric_level, rubric_desc = rubric_score_ref_free(transcript, syllables_asr, intonation_std, speech_rate)
    fluency_score = scale_fluency(speech_rate)
    intonation_score = min(90, intonation_std * 2)
    syllable_score = min(90, (syllables_asr / (duration_sec + 1e-5)) * 10)
    average_raw = (fluency_score + intonation_score + syllable_score) / 3
    final_score = max(10, min(90, round((average_raw / 100) * 90, 2)))

    # Clean up temp file
    os.remove(tmp_path)

    return jsonify({
        'words_detected': int(word_count),
        'syllables_estimated': int(syllables_asr),
        'duration_sec': float(round(duration_sec, 2)),
        'speech_rate': float(round(speech_rate, 2)),
        'intonation_std': float(round(intonation_std, 2)),
        'composite_pronunciation_score': float(final_score),
        'rubric_level': int(rubric_level),
        'rubric_description': rubric_desc
    }), 200 