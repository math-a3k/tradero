from django.conf import settings
from django.core.cache import cache

from .celery import app as celery_app

try:
    cache.delete(settings.SYMBOLS_UPDATE_ALL_INDICATORS_KEY)
except Exception:  # pragma: no cover
    # Prevent from failing if Redis is not available (for building in ReadTheDocs)
    pass

__all__ = ("celery_app",)
