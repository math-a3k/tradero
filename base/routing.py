from django.urls import path

from . import consumers

websocket_urlpatterns = [
    path("ws/symbols/html", consumers.SymbolHTMLConsumer.as_asgi()),
    path("ws/symbols/json", consumers.SymbolJSONConsumer.as_asgi()),
    path("ws/bots/html", consumers.BotHTMLConsumer.as_asgi()),
]
