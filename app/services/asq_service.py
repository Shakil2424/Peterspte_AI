from app.services.audio_transcriber import transcribe_audio
import os

# Configure correct answers
CORRECT_ANSWERS = ["apple", "Introduction", "orange"]
UPLOAD_FOLDER = 'uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def assess_audio_answer(file):
    """
    Assess an audio file answer against the list of correct answers.
    Returns a scoring response based on whether the transcribed audio matches any correct answer.
    """
    # Get transcription
    result, status_code = transcribe_audio(file, UPLOAD_FOLDER)
    
    if status_code != 200:
        return result

    # Get the transcript text
    transcript = result.get('transcript', '').lower()

    # Check if any correct answer is in the transcript
    is_correct = any(answer.lower() in transcript for answer in CORRECT_ANSWERS)

    # Prepare response based on correctness
    if is_correct:
        return {
            "Content": "Yes",
            "Fluency": "5 out of 5",
            "Pronunciation": "5 out of 5",
            "Speaking": "1 out of 1",
            "Listening": "1 out of 1"
        }
    else:
        return {
            "Content": "No",
            "Fluency": "0 out of 5",
            "Pronunciation": "0 out of 5",
            "Speaking": "0 out of 1",
            "Listening": "0 out of 1"
        } 