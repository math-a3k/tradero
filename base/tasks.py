import logging

from django.conf import settings
from django.utils import timezone
from django_rq import get_scheduler, job

from .models import Symbol

logger = logging.getLogger(__name__)


def start_scheduling_round(
    job,
    time_interval=None,
    repeat=None,
    meta=None,
    queue_alias="default",
    *args,
    **kwargs,
):
    """
    TODO: Ugly solution, works but should be better
    """
    should_start_at = timezone.now()
    time_interval = time_interval or settings.TIME_INTERVAL
    minutes_pending = (
        ((should_start_at.minute // time_interval) * time_interval)
        + time_interval
        - should_start_at.minute
    )
    should_start_at = (
        should_start_at.replace(second=0, microsecond=0)
        + timezone.timedelta(minutes=minutes_pending)
        + timezone.timedelta(seconds=1)
    )

    scheduler = get_scheduler(queue_alias)

    job_scheduled = scheduler.schedule(
        scheduled_time=should_start_at,
        func=job,
        args=args,
        kwargs=kwargs,
        interval=time_interval * 60,
        repeat=repeat,
        meta=meta,
    )
    logger.warning(
        f"Job {job_scheduled} scheduled at {should_start_at} with "
        f"an interval of {time_interval} mins"
    )


@job("default", timeout=900)
def retrieve_and_update_symbol(symbol):  # pragma: no cover
    symbol.retrieve_and_update()


@job("default", timeout=900)
def update_all_indicators_job():  # pragma: no cover
    if Symbol.objects.available().filter(model_score__isnull=True).count() > 0:
        # Case of cold start
        Symbol.update_all_indicators(push=False)
    Symbol.update_all_indicators(only_top=True)
