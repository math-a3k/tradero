from django.core.management.base import BaseCommand

from base.models import Symbol


class Command(BaseCommand):
    help = "Calibrates the parameters of all prediction models"

    def handle(self, *args, **options):
        Symbol.calibrate_all_models()

        self.stdout.write(
            self.style.SUCCESS(
                "Successfully calibrated all parameters of prediction models"
            )
        )
