import json
import logging

from asgiref.sync import sync_to_async
from channels.layers import get_channel_layer

from .models import Kline, Symbol, TrainingData

logger = logging.getLogger(__name__)
channel_layer = get_channel_layer()


def update_in_order(symbol, ws_data):
    Kline.from_binance_kline_ws(symbol, ws_data)
    TrainingData.from_klines(symbol)
    symbol.update_indicators()


async def message_handler(message, websocket):
    # process message here
    message = json.loads(message)
    if message["data"]["k"]["x"]:
        symbol = await sync_to_async(Symbol.objects.get)(
            symbol=message["data"]["s"]
        )
        await sync_to_async(update_in_order)(symbol, message["data"])
