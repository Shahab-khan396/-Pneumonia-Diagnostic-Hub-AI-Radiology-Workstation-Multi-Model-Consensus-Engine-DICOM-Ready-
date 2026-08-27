import os
import sys
import importlib.util
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
FLASK_APP_DIR = ROOT_DIR / "Flask Application"

if str(FLASK_APP_DIR) not in sys.path:
    sys.path.insert(0, str(FLASK_APP_DIR))

# Explicitly load the app factory from the inner 'Flask Application/app.py'
spec = importlib.util.spec_from_file_location("flask_app_module", str(FLASK_APP_DIR / "app.py"))
flask_app_module = importlib.util.module_from_spec(spec)
sys.modules["flask_app_module"] = flask_app_module
spec.loader.exec_module(flask_app_module)

app = flask_app_module.create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[*] Starting Pneumonia Diagnostic Hub on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
