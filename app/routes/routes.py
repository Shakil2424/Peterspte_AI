from .asq_routes import asq_bp
from .transcription_routes import transcription_bp
from .dictation_routes import dictation_bp
from .swt_routes import swt_bp
from .sst_routes import sst_bp
from .chatbot_routes import chatbot_bp
from .pronunciation_routes import pronunciation_bp
from .fluency_routes import fluency_bp
from .respond_situation_routes import respond_situation_bp
from .summarize_group_routes import summarize_group_bp
from .repeat_sentence_routes import repeat_sentence_bp
from .retell_lecture_routes import retell_lecture_bp

routes = [
    asq_bp,
    transcription_bp,
    dictation_bp,
    swt_bp,
    sst_bp,
    chatbot_bp,
    pronunciation_bp,
    fluency_bp,
    respond_situation_bp,
    summarize_group_bp,
    repeat_sentence_bp,
    retell_lecture_bp,
    # Add more routers here
]