from flask import Flask
from app.routes.transcription_routes import transcription_bp
from app.routes.asq_routes import asq_bp

def create_app():
    app = Flask(__name__)
    
    # Configure upload folder
    app.config['UPLOAD_FOLDER'] = 'uploads'

    # Register blueprints
    app.register_blueprint(transcription_bp)
    app.register_blueprint(asq_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
