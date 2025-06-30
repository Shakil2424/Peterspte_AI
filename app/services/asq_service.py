from app.services.audio_transcriber import transcribe_audio
import os

# Configure correct answers
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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

    # Check if any correct answer is in the transcript
    is_correct = any(answer.lower() in transcript for answer in CORRECT_ANSWERS)

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
