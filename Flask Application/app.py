import os
import sys
from pathlib import Path
from flask import Flask

# Add base directory to sys.path so imports work regardless of working directory
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import Config, UPLOAD_FOLDER, DEFAULT_MODEL
from core.model_manager import get_model_manager
from routes.api import api_bp
from routes.web import web_bp
from routes.docs import docs_bp


def create_app(config_class=Config) -> Flask:
    """Application factory for Pneumonia Diagnostic Hub."""
    app = Flask(
        __name__,
        static_folder=str(BASE_DIR / "static"),
        template_folder=str(BASE_DIR / "templates")
    )
    app.config.from_object(config_class)

    # Ensure upload folder exists
    upload_path = Path(UPLOAD_FOLDER)
    upload_path.mkdir(parents=True, exist_ok=True)

    # Register Route Blueprints
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(docs_bp)

    # Initialize and pre-warm default model
    with app.app_context():
        try:
            manager = get_model_manager()
            manager.preload(DEFAULT_MODEL)
            app.logger.info(f"Pre-warmed default model: {DEFAULT_MODEL}")
        except Exception as e:
            app.logger.warning(f"Could not pre-warm model on startup: {e}")

    return app


app = create_app()

if __name__ == "__main__":
    print(f"[*] Starting Pneumonia Diagnostic Hub on http://127.0.0.1:5000")
    print(f"[*] Interactive API Docs: http://127.0.0.1:5000/docs")
    print(f"[*] Default Model: {DEFAULT_MODEL} (Cached in RAM)")
    app.run(host="0.0.0.0", port=5000, debug=True)