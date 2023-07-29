from binance.spot import Spot
from django.conf import settings
from django.core.management.base import BaseCommand

from base.models import Symbol


class Command(BaseCommand):
    help = "Loads symbols from Binance"

    def handle(self, *args, **options):
        client = Spot()
        ei = client.exchange_info()
        symbols_processed, symbols_created = 0, 0
        for symbol in ei["symbols"]:
            if symbol["symbol"].endswith(settings.QUOTE_ASSET):
                s, c = Symbol.objects.update_or_create(
                    symbol=symbol["symbol"],
                    defaults={
                        "status": symbol["status"],
                        "base_asset": symbol["baseAsset"],
                        "quote_asset": symbol["quoteAsset"],
                        "info": symbol,
                    },
                )
                symbols_processed += 1
                if c:
                    symbols_created += 1
        self.stdout.write(
            self.style.SUCCESS(
                (
                    f"Successfully processed {symbols_processed} symbols "
                    f"({symbols_created} created)"
                )
            )
        )
