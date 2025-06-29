from flask import Flask
from app.routes import routes


def create_app():
    app = Flask(__name__)

    # Configure upload folder
    app.config["UPLOAD_FOLDER"] = "uploads"

    # Register all blueprints
    for bp in routes:
        app.register_blueprint(bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=6000)
