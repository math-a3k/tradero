from django.conf import settings
from django.core.management.base import BaseCommand

from base.models import Symbol


class Command(BaseCommand):
    help = "Cleans current data, gets fresh one and updates models"

    def add_arguments(self, parser):
        parser.add_argument(
            "--periods",
            action="store",
            default=settings.WARM_UP_PERIODS,
            help="Periods to request",
            type=int,
        )

    def handle(self, *args, **options):
        Symbol.reset_symbols()
        Symbol.general_warm_up(n_periods=options["periods"])
        Symbol.update_all_indicators(push=False)
        Symbol.update_all_indicators(only_top=True, push=False)

        self.stdout.write(
            self.style.SUCCESS(
                "Successfully got fresh data and updated models"
            )
        )
