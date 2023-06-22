import logging

import django_rq
from django_rq.management.commands import rqscheduler

from base.tasks import start_scheduling_round, update_all_indicators_job

scheduler = django_rq.get_scheduler()
log = logging.getLogger(__name__)


def clear_scheduled_jobs():
    # Delete any existing jobs in the scheduler when the app starts up
    for job in scheduler.get_jobs():
        log.warning("Deleting scheduled job %s", job)
        job.delete()


def register_scheduled_jobs():
    start_scheduling_round(update_all_indicators_job)


class Command(rqscheduler.Command):
    def handle(self, *args, **kwargs):
        # This is necessary to prevent dupes
        clear_scheduled_jobs()
        register_scheduled_jobs()
        super(Command, self).handle(*args, **kwargs)
