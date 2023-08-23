from django.core.management.base import BaseCommand

from base.models import Symbol


class Command(BaseCommand):
    help = "Loads symbols from Binance"

    def handle(self, *args, **options):
        message = Symbol.load_symbols()
        self.stdout.write(self.style.SUCCESS(message))
