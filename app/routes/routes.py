from .asq_routes import asq_bp
from .transcription_routes import transcription_bp
from .dictation_routes import dictation_bp


routes = [
    asq_bp,
    transcription_bp,
    dictation_bp,
    # Add more routers here
]