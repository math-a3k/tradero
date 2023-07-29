import os

from celery import Celery
from celery.schedules import crontab

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tradero.settings")

app = Celery("tradero", include=["base.tasks"])

app.conf.beat_schedule = {
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
