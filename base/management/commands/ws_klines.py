import logging

from django.conf import settings
from websocketclient.management.commands import runwebsocketclient

from base.models import Symbol
from base.websocket_client import BinanceWebSocketClient

logger = logging.getLogger(__name__)


class Command(runwebsocketclient.Command):
    def __init__(self, websocket_client=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = (
            websocket_client
            if websocket_client
            else BinanceWebSocketClient(stdout=self.stdout)
        )

    def add_arguments(self, parser):
        parser.add_argument(
            "--periods",
            action="store",
            default=settings.WARM_UP_PERIODS,
            help="Periods to do request",
            type=int,
        )
        parser.add_argument(
            "--reset-ms",
            action="store",
            default=False,
            help="Periods to do request",
            type=bool,
        )
        parser.add_argument(
            "--clean",
            action="store",
            default=True,
            help="Periods to do request",
            type=bool,
        )

    def configure_options(self, **options):
        super().configure_options(**options)

        if options["reset_ms"]:
            Symbol.objects.all().update(model_score=None)
        if options["clean"]:
            Symbol.reset_symbols()
            Symbol.general_warm_up(n_periods=options["periods"])
            Symbol.update_all_indicators(push=False)
            Symbol.update_all_indicators(only_top=True)

        symbols = Symbol.objects.top_symbols().values_list("symbol", flat=True)
        path = "/stream?streams=" + "/".join(
            [
                f"{s.lower()}@kline_{settings.TIME_INTERVAL}m"
                for s in sorted(symbols)
            ]
        )
        self.client.path = path
