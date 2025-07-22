from flask import Blueprint, request, jsonify
import numpy as np
import librosa
import tempfile
import os
import re
import torch
import whisperx

def count_syllables(text):
    return sum(len(re.findall(r'[aeiouy]+', word.lower())) for word in text.split())

def count_filler_words(text):
    fillers = ['um', 'uh', 'erm', 'ah', 'like', 'you know', 'well', 'so', 'actually']
    words = text.lower().split()
    return sum(words.count(f) for f in fillers)

def count_repetitions(text):
    words = text.lower().split()
    return sum(1 for i in range(1, len(words)) if words[i] == words[i-1])

def count_false_starts(text):
    words = text.lower().split()
    return sum(1 for i in range(2, len(words)) if words[i] == words[i-2])

def count_long_pauses(audio, sr, threshold_s=0.3):
    try:
        intervals = librosa.effects.split(audio, top_db=20, frame_length=2048, hop_length=512)
        pauses = 0
        for i in range(1, len(intervals)):
            pause_duration = (intervals[i][0] - intervals[i-1][1]) / sr
            if pause_duration > threshold_s:
                pauses += 1
        return pauses
    except:
        return 0

def longest_smooth_run(asr_text, audio, sr):
    try:
        intervals = librosa.effects.split(audio, top_db=20)
        words = asr_text.split()
        if len(words) == 0 or len(intervals) == 0:
            return 0
        total_duration = librosa.get_duration(y=audio, sr=sr)
        word_times = np.linspace(0, total_duration, num=len(words)+1)
        max_run = 0
        for start, end in intervals:
            start_time = start / sr
            end_time = end / sr
            count = 0
            for i in range(len(words)):
                if word_times[i] >= start_time and word_times[i+1] <= end_time:
                    count += 1
            if count > max_run:
                max_run = count
        return max_run
    except:
        return len(asr_text.split()) // 2

def scale_fluency(val):
    val = np.clip(val, 1.0, 5.0)
    return 10 + ((val - 1.0) / (5.0 - 1.0))**1.5 * 80

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

def fluency_metrics(transcript, audio, sr, duration_sec):
    syllables_asr = count_syllables(transcript)
    word_count = len(transcript.split())
    speech_rate = syllables_asr / duration_sec if duration_sec > 0 else 0
    try:
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.1)
        median_mag = np.median(magnitudes[magnitudes > 0])
        voiced_pitches = pitches[magnitudes > median_mag * 0.5]
        voiced_pitches = voiced_pitches[(voiced_pitches > 50) & (voiced_pitches < 500)]
        intonation_std = np.std(voiced_pitches) if len(voiced_pitches) > 0 else 0
    except:
        intonation_std = 10
    hesitation_count = count_filler_words(transcript)
    repetition_count = count_repetitions(transcript)
    false_start_count = count_false_starts(transcript)
    long_pause_count = count_long_pauses(audio, sr)
    longest_run = longest_smooth_run(transcript, audio, sr)
    rubric_level, rubric_desc = rubric_score_ref_free(
        transcript,
        syllables_asr,
        intonation_std,
        speech_rate,
    )
    fluency_score = scale_fluency(speech_rate)
    intonation_score = min(90, intonation_std * 2)
    syllable_score = min(90, (syllables_asr / (duration_sec + 1e-5)) * 15)
    disfluency_penalty = (hesitation_count + repetition_count + false_start_count) * 5
    average_raw = (fluency_score + intonation_score + syllable_score) / 3
    final_score = max(10, min(90, round(average_raw - disfluency_penalty, 2)))
    return {
        'words_detected': int(word_count),
        'syllables_estimated': int(syllables_asr),
        'duration_sec': float(round(duration_sec, 2)),
        'speech_rate': float(round(speech_rate, 2)),
        'intonation_std': float(round(intonation_std, 2)),
        'composite_fluency_score': float(final_score),
        'rubric_level': int(rubric_level),
        'rubric_description': rubric_desc,
        'hesitation_count': int(hesitation_count),
        'repetition_count': int(repetition_count),
        'false_start_count': int(false_start_count),
        'long_pause_count': int(long_pause_count),
        'longest_smooth_run': int(longest_run)
    }

fluency_bp = Blueprint('fluency', __name__)

@fluency_bp.route('/fluency', methods=['POST'])
def fluency():
    if 'file' not in request.files:
        return jsonify({'error': 'Audio file is required.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        file.seek(0)
        file.save(tmp.name)
        tmp_path = tmp.name
    # Transcribe using WhisperX (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model("base", device=device, compute_type=compute_type)
    result = model.transcribe(tmp_path, language="en")
    if 'segments' in result and len(result['segments']) > 0:
        transcript = " ".join([seg['text'].strip() for seg in result['segments']])
    else:
        transcript = result.get('text', '')
    transcript = transcript.strip().lower()
    audio, sr = librosa.load(tmp_path, sr=None)
    duration_sec = librosa.get_duration(y=audio, sr=sr)
    metrics = fluency_metrics(transcript, audio, sr, duration_sec)
    os.remove(tmp_path)
    metrics['transcript'] = transcript
    return jsonify(metrics), 200 