import os
import logging
import torch
from werkzeug.utils import secure_filename
from faster_whisper import WhisperModel

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'wav'}
ALLOWED_MIMETYPES = {
    'audio/wav', 'audio/x-wav'
}

def allowed_file(file):
    if '.' not in file.filename:
        return False
    
    ext = file.filename.rsplit('.', 1)[1].lower()
    mimetype = file.content_type

    logger.debug(f"File extension: {ext}, Mimetype: {mimetype}")
    
    return ext in ALLOWED_EXTENSIONS and mimetype in ALLOWED_MIMETYPES

# Determine device and compute type
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "int8_float16" if device == "cuda" else "int8"
logger.info(f"Initializing WhisperModel on device: {device} with compute_type: {compute_type}")

# Initialize the Whisper model once
model = WhisperModel("large-v3", device=device, compute_type=compute_type)

def transcribe_audio(file, upload_folder):
    # if not allowed_file(file):
    #     return {'error': 'Invalid file type'}, 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    try:
        file.save(filepath)
        logger.debug(f"File saved to {filepath}")

        # Transcribe without conversion, assume wav input
        segments, info = model.transcribe(filepath, language="en", beam_size=5)

        # Combine all segment texts
        full_transcript = " ".join(segment.text.strip() for segment in segments)

        logger.debug("Transcription completed successfully")

        return {'transcript': full_transcript}, 200

    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        return {'error': 'Error during transcription'}, 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
        if device == "cuda":
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")
