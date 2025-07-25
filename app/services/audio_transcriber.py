import whisperx
import torch
import os
import gc
import warnings
import logging
from werkzeug.utils import secure_filename

# Suppress warnings to reduce noise
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'webm', 'ogg'}
ALLOWED_MIMETYPES = {
    'audio/wav', 'audio/x-wav',
    'audio/mpeg', 'audio/mp3', 'audio/mp3; charset=utf-8',
    'audio/mp4', 'audio/x-m4a', 'audio/m4a',
    'audio/flac', 'audio/x-flac',
    'audio/webm', 'audio/ogg',
    # Additional common mimetypes
    'audio/aac', 'audio/x-aac',
    'audio/3gpp', 'audio/3gpp2',
    'audio/amr', 'audio/amr-wb',
    'audio/basic', 'audio/x-basic',
    'audio/midi', 'audio/x-midi',
    'audio/opus',
    # Generic audio types
    'audio/*', 'application/octet-stream'
}

def allowed_file(file):
    if '.' not in file.filename:
        logger.warning(f"No file extension found in filename: {file.filename}")
        return False
    
    ext = file.filename.rsplit('.', 1)[1].lower()
    mimetype = file.content_type

    logger.debug(f"File extension: {ext}, Mimetype: {mimetype}")
    logger.debug(f"Allowed extensions: {ALLOWED_EXTENSIONS}")
    logger.debug(f"Allowed mimetypes: {ALLOWED_MIMETYPES}")
    
    # Check if extension is allowed
    ext_allowed = ext in ALLOWED_EXTENSIONS
    mimetype_allowed = mimetype in ALLOWED_MIMETYPES
    
    logger.debug(f"Extension allowed: {ext_allowed}, Mimetype allowed: {mimetype_allowed}")
    
    # Be more flexible - accept if either extension OR mimetype is valid
    # This handles cases where browsers send different mimetypes
    if ext_allowed or mimetype_allowed:
        logger.info(f"File accepted: {file.filename} (ext: {ext}, mimetype: {mimetype})")
        return True
    
    logger.warning(f"File rejected: {file.filename} (ext: {ext}, mimetype: {mimetype})")
    return False

# --- Global WhisperX Model Initialization ---
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
logger.info(f"🖥️ Loading WhisperX model at startup on device: {device}")
WHISPERX_MODEL = whisperx.load_model("base", device=device, compute_type=compute_type)
logger.info("✅ WhisperX model loaded and ready.")

def simple_transcribe(audio_file):
    """
    Simple, stable transcription function using whisperx
    """
    try:
        logger.info(f"🎤 Processing: {audio_file}")
        
        # Check if file exists
        if not os.path.exists(audio_file):
            logger.error(f"❌ File not found: {audio_file}")
            return None
        
        # Use global model
        logger.info(f"🖥️ Using device: {device}")
        
        # Simple transcription with minimal options
        logger.info("🎯 Transcribing...")
        result = WHISPERX_MODEL.transcribe(audio_file, language="en")
        
        logger.info("✅ Transcription complete!")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return None

def transcribe_audio(file, upload_folder):
    """
    Main transcription function for the Flask app
    """
    # Log file information for debugging
    logger.info(f"Received file: {file.filename}")
    logger.info(f"File content type: {file.content_type}")
    logger.info(f"File size: {len(file.read()) if hasattr(file, 'read') else 'unknown'}")
    
    # Reset file pointer after reading
    if hasattr(file, 'seek'):
        file.seek(0)
    
    if not allowed_file(file):
        return {
            'error': 'Invalid file type', 
            'details': {
                'filename': file.filename,
                'content_type': file.content_type,
                'allowed_extensions': list(ALLOWED_EXTENSIONS),
                'allowed_mimetypes': list(ALLOWED_MIMETYPES)
            }
        }, 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    
    try:
        file.save(filepath)
        logger.debug(f"File saved to {filepath}")

        # Convert WebM to WAV if needed
        if filename.endswith('.webm'):
            try:
                import subprocess
                wav_file = filepath.replace('.webm', '.wav')
                logger.info(f"🔄 Converting {filepath} to {wav_file}...")
                
                cmd = ['ffmpeg', '-i', filepath, '-acodec', 'pcm_s16le', '-ar', '16000', '-y', wav_file]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    filepath = wav_file
                    logger.info(f"✅ Converted to: {wav_file}")
                else:
                    logger.warning(f"⚠️ Conversion failed, trying original file")
            except Exception as e:
                logger.warning(f"⚠️ Conversion error: {e}")

        # Transcribe using whisperx
        logger.info("🚀 Starting transcription...")
        result = simple_transcribe(filepath)
        
        if not result:
            return {'error': 'Transcription failed'}, 500

        # Extract transcript text
        if "text" in result:
            full_transcript = result["text"].strip()
        elif "segments" in result:
            full_transcript = " ".join(seg["text"].strip() for seg in result["segments"])
        else:
            return {'error': 'No transcript generated'}, 500

        if not full_transcript:
            return {'error': 'Empty transcript generated'}, 500

        logger.info("Transcription completed successfully")
        return {'transcript': full_transcript}, 200

    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        return {'error': 'Error during transcription'}, 500

    finally:
        # Clean up temporary files
        if os.path.exists(filepath):
            os.remove(filepath)
        # Clean up converted WAV file if it exists
        if filename.endswith('.webm') and os.path.exists(filepath.replace('.webm', '.wav')):
            os.remove(filepath.replace('.webm', '.wav'))
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")