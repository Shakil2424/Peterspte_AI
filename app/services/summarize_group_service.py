import os
import numpy as np
import librosa
import torch
import nltk
import re
from app.services.audio_transcriber import transcribe_audio
from sentence_transformers import SentenceTransformer, util
import logging

nltk.download('punkt', quiet=True)

# --- Global SentenceTransformer Model Initialization ---
MODEL_NAME = 'all-MiniLM-L6-v2'
logging.info(f"Loading SentenceTransformer model '{MODEL_NAME}' at startup...")
SENTENCE_TRANSFORMER_MODEL = SentenceTransformer(MODEL_NAME, device="cpu")
logging.info("✅ SentenceTransformer model loaded and ready.")

# Provided ContinuousContentScorer
class ContinuousContentScorer:
    """
    A comprehensive content scoring system for evaluating summaries against reference transcripts.
    
    Scores based on:
    - Idea Coverage: How well the summary captures reference content (0-4 scale)
    - Paraphrase Depth: Quality of semantic similarity without literal copying (0-1 scale)
    - Objectivity: Penalty for subjective language (0-1 penalty)
    - Final Score: PTE-style score (10-90 range)
    """
    
    def __init__(self, model=None):
        try:
            self.model = model if model is not None else SENTENCE_TRANSFORMER_MODEL
            logging.info(f"Using SentenceTransformer model: {MODEL_NAME}")
        except Exception as e:
            logging.error(f"Failed to load model {MODEL_NAME}: {e}")
            raise
        self.subjective_words = {
            'think', 'feel', 'believe', 'guess', 'suppose', 'assume', 'reckon',
            'probably', 'maybe', 'perhaps', 'possibly', 'likely', 'unlikely',
            'might', 'could', 'may', 'seem', 'appear', 'suggest', 'suggests',
            'in my opinion', 'i feel', 'i think', 'i believe', 'personally',
            'from my perspective', 'it seems to me',
            'somewhat', 'rather', 'quite', 'fairly', 'relatively',
            'allegedly', 'supposedly', 'reportedly', 'apparently'
        }
    def preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        return text.lower().strip()
    def tokenize_sentences(self, text: str) -> list:
        if not text:
            return []
        try:
            sentences = nltk.sent_tokenize(text)
            return [sent.strip() for sent in sentences if len(sent.strip()) > 5]
        except Exception as e:
            logging.warning(f"Sentence tokenization failed: {e}")
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            return [sent + '.' for sent in sentences if len(sent) > 5]
    def detect_subjectivity(self, text: str) -> float:
        if not text:
            return 0.0
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        if not words:
            return 0.0
        subjective_count = 0
        for word in self.subjective_words:
            if ' ' not in word:
                subjective_count += words.count(word)
            else:
                pattern = r'\b' + re.escape(word) + r'\b'
                subjective_count += len(re.findall(pattern, processed_text))
        return subjective_count / len(words)
    def parse_transcript(self, transcript_text: str) -> dict:
        if not transcript_text:
            return {}
        
        logging.info(f"Parsing transcript: {transcript_text[:200]}...")
        
        text = re.sub(r'Narrator?:\s*[^\.]+\.', '', transcript_text, flags=re.IGNORECASE)
        # Updated regex to handle "Speaker 1:" format with optional spaces
        speaker_pattern = r'([A-Za-z]+\s*\d*|[A-Z][a-z]+):\s*'
        parts = re.split(speaker_pattern, text)
        parts = [part.strip() for part in parts if part.strip()]
        
        logging.info(f"Split parts: {parts[:5]}...")
        
        speakers_dict = {}
        current_speaker = None
        
        for i, part in enumerate(parts):
            # Check if this part is a speaker identifier (ends with colon or matches speaker pattern)
            if re.match(r'^[A-Za-z]+\s*\d*$', part) or part.endswith(':'):
                current_speaker = part.rstrip(':')  # Remove colon if present
                logging.info(f"Found speaker: {current_speaker}")
                continue
            elif current_speaker:
                sentences = self.tokenize_sentences(part)
                if current_speaker not in speakers_dict:
                    speakers_dict[current_speaker] = []
                speakers_dict[current_speaker].extend(sentences)
                logging.info(f"Added {len(sentences)} sentences for {current_speaker}")
                current_speaker = None
        
        # If no speakers were parsed, try alternative parsing
        if not speakers_dict:
            logging.info("Primary parsing failed, trying fallback method...")
            # Fallback: split by lines and look for speaker patterns
            lines = transcript_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Look for "Speaker X:" pattern
                speaker_match = re.match(r'^(Speaker\s*\d+):\s*(.*)', line, re.IGNORECASE)
                if speaker_match:
                    speaker_id = speaker_match.group(1)
                    content = speaker_match.group(2).strip()
                    if content:
                        sentences = self.tokenize_sentences(content)
                        if speaker_id not in speakers_dict:
                            speakers_dict[speaker_id] = []
                        speakers_dict[speaker_id].extend(sentences)
                        logging.info(f"Fallback: Added {len(sentences)} sentences for {speaker_id}")
        
        logging.info(f"Final speakers dict: {list(speakers_dict.keys())}")
        return speakers_dict
    def compute_similarity_metrics(self, ref_embeddings, summary_embeddings):
        if ref_embeddings.shape[0] == 0 or summary_embeddings.shape[0] == 0:
            return np.array([]), np.array([])
        sim_matrix = util.cos_sim(ref_embeddings, summary_embeddings)
        max_sim_per_ref = sim_matrix.max(dim=1).values.cpu().numpy()
        max_sim_per_summary = sim_matrix.max(dim=0).values.cpu().numpy()
        return max_sim_per_ref, max_sim_per_summary
    def score(self, reference_transcript, summary_text: str) -> dict:
        if isinstance(reference_transcript, str):
            parsed_transcript = self.parse_transcript(reference_transcript)
            if not parsed_transcript:
                logging.error(f"Failed to parse transcript. Raw text: {reference_transcript[:500]}...")
                return self._empty_score_result("Failed to parse transcript - no speakers identified")
        elif isinstance(reference_transcript, dict):
            parsed_transcript = reference_transcript
        else:
            return self._empty_score_result("Invalid reference transcript format")
        
        if not parsed_transcript or not summary_text:
            return self._empty_score_result("Empty input provided")
        
        logging.info(f"Successfully parsed {len(parsed_transcript)} speakers")
        
        summary_sentences = self.tokenize_sentences(summary_text)
        if not summary_sentences:
            return self._empty_score_result("No valid summary sentences found")
        
        try:
            summary_embeddings = self.model.encode(summary_sentences, convert_to_tensor=True)
        except Exception as e:
            logging.error(f"Failed to encode summary: {e}")
            return self._empty_score_result("Summary encoding failed")
        
        speaker_scores = {}
        all_max_similarities = []
        total_ref_sentences = 0
        
        for speaker_id, sentences in parsed_transcript.items():
            if not sentences:
                speaker_scores[speaker_id] = {
                    'idea_coverage': 0.0,
                    'sentence_count': 0
                }
                continue
            
            valid_sentences = [s for s in sentences if s and len(s.strip()) > 5]
            total_ref_sentences += len(valid_sentences)
            
            if not valid_sentences:
                speaker_scores[speaker_id] = {
                    'idea_coverage': 0.0,
                    'sentence_count': 0
                }
                continue
            
            try:
                ref_embeddings = self.model.encode(valid_sentences, convert_to_tensor=True)
                max_sim_per_ref, _ = self.compute_similarity_metrics(ref_embeddings, summary_embeddings)
                all_max_similarities.extend(max_sim_per_ref)
                avg_similarity = max_sim_per_ref.mean() if len(max_sim_per_ref) > 0 else 0
                idea_coverage = float(avg_similarity * 4)
                speaker_scores[speaker_id] = {
                    'idea_coverage': idea_coverage,
                    'sentence_count': len(valid_sentences),
                    'avg_similarity': float(avg_similarity)
                }
            except Exception as e:
                logging.warning(f"Processing failed for speaker {speaker_id}: {e}")
                speaker_scores[speaker_id] = {
                    'idea_coverage': 0.0,
                    'sentence_count': len(valid_sentences),
                    'error': str(e)
                }
        
        if not speaker_scores:
            return self._empty_score_result("No valid speaker data processed")
        
        logging.info(f"Processed {len(speaker_scores)} speakers with {total_ref_sentences} total sentences")
        
        total_weighted_coverage = sum(
            scores['idea_coverage'] * scores['sentence_count']
            for scores in speaker_scores.values()
        )
        raw_idea_coverage = (
            total_weighted_coverage / total_ref_sentences
            if total_ref_sentences > 0 else 0
        )
        
        paraphrase_depth = self._calculate_paraphrase_depth(all_max_similarities)
        subjectivity_ratio = self.detect_subjectivity(summary_text)
        objectivity_penalty = min(subjectivity_ratio * 5, 1.0)
        
        final_score = self._calculate_final_score(
            raw_idea_coverage, paraphrase_depth, objectivity_penalty
        )
        
        logging.info(f"Final score: {final_score}, Idea coverage: {raw_idea_coverage}, Paraphrase depth: {paraphrase_depth}")
        
        return {
            'raw_idea_coverage': float(raw_idea_coverage),
            'paraphrase_depth': float(paraphrase_depth),
            'objectivity_penalty': float(objectivity_penalty),
            'final_score': float(final_score),
            'details': {
                'idea_coverage_per_speaker': speaker_scores,
                'subjective_ratio': float(subjectivity_ratio),
                'total_reference_sentences': total_ref_sentences,
                'total_summary_sentences': len(summary_sentences),
                'summary_length_words': len(summary_text.split()),
                'reference_speakers': list(parsed_transcript.keys()),
                'parsed_transcript': parsed_transcript
            }
        }
    def _calculate_paraphrase_depth(self, similarities: list) -> float:
        if not similarities:
            return 0.0
        similarities_array = np.array(similarities)
        good_paraphrase_mask = (similarities_array > 0.3) & (similarities_array < 0.9)
        paraphrase_sims = similarities_array[good_paraphrase_mask]
        if len(paraphrase_sims) == 0:
            return 0.0
        return float(np.clip(paraphrase_sims.mean(), 0, 1))
    def _calculate_final_score(self, idea_coverage: float, paraphrase_depth: float, objectivity_penalty: float) -> float:
        base_score = idea_coverage + paraphrase_depth - objectivity_penalty
        base_score = np.clip(base_score, 0, 5)
        return 10 + (base_score / 5) * 80
    def _empty_score_result(self, reason: str) -> dict:
        logging.warning(f"Returning empty score: {reason}")
        return {
            'raw_idea_coverage': 0.0,
            'paraphrase_depth': 0.0,
            'objectivity_penalty': 1.0,
            'final_score': 10.0,
            'details': {
                'error': reason,
                'idea_coverage_per_speaker': {},
                'subjective_ratio': 0.0,
                'total_reference_sentences': 0,
                'total_summary_sentences': 0
            }
        }

# --- Pronunciation and Fluency Scoring (reuse from respond_situation_service) ---
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

# --- Main Summarize Group Scoring Function ---
def evaluate_summarize_group(reference_text: str, file, upload_folder: str):
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
        # Content scoring
        scorer = ContinuousContentScorer(model=SENTENCE_TRANSFORMER_MODEL)
        content_result = scorer.score(reference_text, transcript)
        content_score = max(10, min(90, round(content_result.get('final_score', 10))))
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
        'content_details': content_result,
        'penalty_info': penalty_info
    }, 200 

# --- Test Function for Debugging ---
def test_transcript_parsing():
    """Test function to verify transcript parsing works correctly"""
    test_transcript = """Narration: Three students are discussing whether to take early morning classes next semester at the campus coffee shop.
Speaker 1: Have you guys planned your class schedule for next term? I'm thinking about taking three 8 a.m. classes.
Speaker 2: I'm with you on that! I took two morning classes this semester and found I'm much more productive in the early hours.
Speaker 3: Actually, I avoid early classes whenever possible. I signed up for an 8 a.m. lecture last semester and ended up sleeping through half of them."""
    
    scorer = ContinuousContentScorer()
    parsed = scorer.parse_transcript(test_transcript)
    
    print("Test transcript parsing:")
    print(f"Original text: {test_transcript[:100]}...")
    print(f"Parsed speakers: {list(parsed.keys())}")
    for speaker, sentences in parsed.items():
        print(f"  {speaker}: {len(sentences)} sentences")
        if sentences:
            print(f"    First sentence: {sentences[0][:50]}...")
    
    return parsed

if __name__ == "__main__":
    test_transcript_parsing() 