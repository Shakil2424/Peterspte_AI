from flask import Flask
from app.routes import routes
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    # Configure upload folder
    app.config["UPLOAD_FOLDER"] = "uploads"

    # Register all blueprints
    for bp in routes:
        app.register_blueprint(bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=6000)
