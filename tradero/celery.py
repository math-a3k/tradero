import os

from celery import Celery
from celery.schedules import crontab

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tradero.settings")

app = Celery("tradero", include=["base.tasks"])

app.conf.task_routes = {
    "base.tasks.retrieve_and_update_symbol": {"queue": "symbols"},
    "base.tasks.update_all_indicators_job": {"queue": "symbols"},
    "base.tasks.update_all_bots_job": {"queue": "bots"},
    "base.tasks.update_bots_group_job": {"queue": "bots"},
    "base.tasks.bots_logrotate": {"queue": "bots"},
    "base.tasks.dummy_user_reset": {"queue": "bots"},
}

app.conf.beat_schedule = {
    # Executes every day.
    "update-symbols-and-indicators": {
        "task": "base.tasks.update_all_indicators_job",
        "schedule": crontab(
            hour=23,
            minute=57,
        ),
        "kwargs": {"load_symbols": True, "all_symbols": True},
    },
    "dummyy-user-reset": {
        "task": "base.tasks.update_all_indicators_job",
        "schedule": crontab(
            hour=23,
            minute=57,
        ),
    },
    # Executes every 5 mins.
    "update-indicators-every-5-mins": {
        "task": "base.tasks.update_all_indicators_job",
        "schedule": crontab(
            minute="*/5",
        ),
        "args": None,
    },
    # Executes every min.
    "update-bots-every-1-min": {
        "task": "base.tasks.update_all_bots_job",
        "schedule": crontab(
            minute="*/1",
        ),
        "args": None,
    },
}
app.conf.timezone = "UTC"
# app.conf.broker_url = "redis://localhost:6379/10"

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Load task modules from all registered Django apps.
app.autodiscover_tasks()
