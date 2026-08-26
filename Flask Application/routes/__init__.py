"""Routes package registering API, Web, and Swagger Docs blueprints."""
from .api import api_bp
from .web import web_bp
from .docs import docs_bp

__all__ = ["api_bp", "web_bp", "docs_bp"]
