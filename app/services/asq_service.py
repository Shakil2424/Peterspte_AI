from app.services.audio_transcriber import transcribe_audio
import os
import re

# Configure correct answers
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def normalize_text(text):
    """
    Normalize text for comparison by removing punctuation and extra spaces
    """
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def assess_audio_answer(file, CORRECT_ANSWERS):
    """
    Assess an audio file answer against the list of correct answers.
    Returns a scoring response based on whether the transcribed audio matches any correct answer.
    """
    # Get transcription
    result, status_code = transcribe_audio(file, UPLOAD_FOLDER)

    if status_code != 200:
        return result

    # Get the transcript text
    transcript = result.get("transcript", "").lower()
    normalized_transcript = normalize_text(transcript)

    # Check for exact matches first, then very close matches
    is_correct = False
    
    for answer in CORRECT_ANSWERS:
        normalized_answer = normalize_text(answer)
        
        # Check for exact match
        if normalized_transcript == normalized_answer:
            is_correct = True
            break
        
        # Check if the answer is a complete phrase within the transcript
        transcript_words = normalized_transcript.split()
        answer_words = normalized_answer.split()
        
        # Calculate word difference
        word_difference = abs(len(transcript_words) - len(answer_words))
        
        # Only accept if the difference is minimal (0-1 extra words)
        if word_difference <= 1:
            # If answer has multiple words, check if all words appear in sequence
            if len(answer_words) > 1:
                # Check if answer appears at the beginning
                if transcript_words[:len(answer_words)] == answer_words:
                    is_correct = True
                    break
                # Check if answer appears at the end
                elif transcript_words[-len(answer_words):] == answer_words:
                    is_correct = True
                    break
                # Check if answer appears in the middle (but only if transcript is not much longer)
                elif len(transcript_words) <= len(answer_words) + 2:
                    for i in range(len(transcript_words) - len(answer_words) + 1):
                        if transcript_words[i:i+len(answer_words)] == answer_words:
                            is_correct = True
                            break
            
            # If answer is a single word, check for exact word match
            elif len(answer_words) == 1:
                if answer_words[0] in transcript_words:
                    is_correct = True
                    break
        
        if is_correct:
            break

    # Prepare response based on correctness
    if is_correct:
        return {
            "content": True,
            "fluency": 1,
            "pronunciation": 1,
            "speaking": 1,
            "listening": 1,
            "transcript": transcript,   
        }
    else:
        return {
            "content": False,
            "fluency": 0,
            "pronunciation": 0,
            "speaking": 0,
            "listening": 0,
            "transcript": transcript,
        }
