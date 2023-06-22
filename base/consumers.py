import json
import logging
import sys

from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer
from django.utils import timezone

from .models import WSClient

logger = logging.getLogger(__name__)


class TraderoConsumer(WebsocketConsumer):
    group_name = None

    def connect(self):
        self.accept()
        if "pytest" not in sys.modules:  # pragma: no cover
            WSClient.objects.create(
                channel_group=self.group_name, channel_name=self.channel_name
            )
        # join the group
        async_to_sync(self.channel_layer.group_add)(
            self.group_name,
            self.channel_name,
        )
        logger.info(
            f"Should be connected [{self.group_name}] {self.channel_name}"
        )

    def disconnect(self, close_code):
        if "pytest" not in sys.modules:  # pragma: no cover
            WSClient.objects.filter(channel_name=self.channel_name).update(
                time_disconnect=timezone.now()
            )
        async_to_sync(self.channel_layer.group_discard)(
            self.group_name,
            self.channel_name,
        )
        logger.info(
            f"Should be DISconnected [{self.group_name}] {self.channel_name}"
        )

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json["message"]
        self.send(text_data=json.dumps({"message": message}))


class SymbolHTMLConsumer(TraderoConsumer):
    group_name = "symbols_html"

    def symbol_html_message(self, event):
        # Handles the "symbol.html.message" event when it's sent to the consumer.
        self.send(text_data=json.dumps(event["message"]))


class SymbolJSONConsumer(TraderoConsumer):
    group_name = "symbols_json"

    def symbol_json_message(self, event):
        # Handles the "symbol.json.message" event when it's sent to the consumer.
        self.send(text_data=event["message"])
