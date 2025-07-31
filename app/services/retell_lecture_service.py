import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import librosa
import os
from app.services.audio_transcriber import transcribe_audio

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class GracefulContentScorer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.stop_words = set(stopwords.words('english'))

    def extract_key_terms(self, text, top_n=10):
        tfidf = self.vectorizer.fit_transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = tfidf.toarray()[0]
        top_indices = np.argsort(scores)[::-1]
        terms = []
        for i in top_indices:
            term = feature_names[i]
            if term.isalpha() and term not in self.stop_words:
                terms.append(term)
            if len(terms) == top_n:
                break
        return terms

    def _get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower().replace('_', ' '))
        return synonyms

    def compute_semantic_overlap(self, ref_terms, resp_terms):
        if not ref_terms:
            return 0.0
        resp_set = set(resp_terms)
        matched_count = 0
        for ref_term in ref_terms:
            ref_syns = self._get_synonyms(ref_term)
            found_match = False
            for resp_term in resp_set:
                if resp_term == ref_term:
                    found_match = True
                    break
                if resp_term in ref_syns or ref_term in self._get_synonyms(resp_term):
                    found_match = True
                    break
            if found_match:
                matched_count += 1
        return (matched_count / len(ref_terms)) * 100

    def compute_tfidf_similarity(self, reference, response):
        tfidf_matrix = self.vectorizer.fit_transform([reference, response])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity * 100

    def score(self, reference, response):
        ref_terms = self.extract_key_terms(reference)
        resp_terms = self.extract_key_terms(response)
        semantic_overlap = self.compute_semantic_overlap(ref_terms, resp_terms)
        tfidf_similarity = self.compute_tfidf_similarity(reference, response)
        final_score = (0.65 * semantic_overlap) + (0.35 * tfidf_similarity)
        # Scale to [10, 90]
        scaled_score = max(10, min(90, round((final_score / 100) * 80 + 10, 2)))
        return {
            "final_score": scaled_score,
            "semantic_overlap": round(semantic_overlap, 2),
            "tfidf_similarity": round(tfidf_similarity, 2),
            "reference_key_terms": ref_terms,
            "response_key_terms": resp_terms
        }

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

def evaluate_retell_lecture(reference_text, file, upload_folder):
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
        content_result = GracefulContentScorer().score(reference_text, transcript)
        content_score = content_result.pop('final_score')
        pronunciation = score_pronunciation(transcript, audio, sr, duration_sec)
        fluency = score_fluency(transcript, audio, sr, duration_sec)
        
        # === SPEAKING AND LISTENING SCORES ===
        speaking = ((fluency * 80) / 100) + ((pronunciation * 20) / 100)
        listening = ((content_score * 80) / 100) + ((pronunciation * 20) / 100)
    finally:
        os.remove(tmp_path)
    return {
        'transcription': transcript,
        'content_score': content_score,
        'content_details': content_result,
        'pronunciation_score': pronunciation,
        'fluency_score': fluency,
        'speaking_score': float(round(speaking, 2)),
        'listening_score': float(round(listening, 2))
    }, 200 