from django.core.management.base import BaseCommand

from base.models import Symbol


class Command(BaseCommand):
    help = "Calibrates the window of all prediction models"

    def handle(self, *args, **options):
        Symbol.calibrate_all_windows()

        self.stdout.write(
            self.style.SUCCESS(
                "Successfully calibrated all windows of prediction models"
            )
        )
