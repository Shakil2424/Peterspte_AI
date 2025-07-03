from .asq_routes import asq_bp
from .transcription_routes import transcription_bp
from .dictation_routes import dictation_bp
from .swt_routes import swt_bp
from .chatbot_routes import chatbot_bp


routes = [
    asq_bp,
    transcription_bp,
    dictation_bp,
    swt_bp,
    chatbot_bp,
    # Add more routers here
]