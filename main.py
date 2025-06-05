from flask import Flask
from app.routes.transcription_routes import transcription_bp  # adjust import path as needed

def create_app():
    app = Flask(__name__)

    # Register blueprints
    app.register_blueprint(transcription_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
