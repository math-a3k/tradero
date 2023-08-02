from django.conf import settings
from django.core.cache import cache

from .celery import app as celery_app

cache.delete(settings.SYMBOLS_UPDATE_ALL_INDICATORS_KEY)

__all__ = ("celery_app",)
