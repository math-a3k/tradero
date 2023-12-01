import signal
from contextlib import contextmanager
from decimal import Decimal
from io import StringIO
from unittest import mock

import pytest
import requests_mock
from asgiref.sync import sync_to_async
from channels.layers import get_channel_layer
from channels.testing import WebsocketCommunicator
from django.conf import settings
from django.core.management import call_command
from django.test import Client, TestCase
from django.test.utils import override_settings
from django.urls import reverse
from django.utils import timezone

from base import tasks

from .consumers import BotHTMLConsumer, SymbolHTMLConsumer, SymbolJSONConsumer
from .forms import TraderoBotForm
from .handlers import message_handler
from .indicators import ATR, DC
from .models import (
    Kline,
    Symbol,
    TraderoBot,
    TraderoBotGroup,
    TrainingData,
    User,
    WSClient,
)

BINANCE_API_URL = "https://api.binance.com"
TEST_SETTINGS = {
    "SYNC_EXECUTION": True,
    "EXCHANGE_API_URL": BINANCE_API_URL,
}


@pytest.fixture(scope="session", autouse=True)
def test_settings():
    with override_settings(**TEST_SETTINGS):
        yield


@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"block timedout after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


class TestTrainingData(TestCase):
    fixtures = ["base/fixtures/klines.json"]

    @classmethod
    def setUpClass(cls):
        cls.s1, _ = Symbol.objects.get_or_create(
            symbol="S1BUSD",
            status="TRADING",
            base_asset="S1",
            quote_asset="BUSD",
        )
        super().setUpClass()

    def test_creation(self):
        self.assertEqual(len(TrainingData.from_klines(self.s1)), 35)
        self.assertEqual(len(TrainingData.from_klines(self.s1)), 0)


class TestKlines(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1, _ = Symbol.objects.get_or_create(
            symbol="S1BUSD",
            status="TRADING",
            base_asset="S1",
            quote_asset="BUSD",
        )
        super().setUpClass()

    def test_creation_from_http(self):
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/klines?symbol={self.s1.symbol}"
                f"&interval={settings.TIME_INTERVAL}m",
                text="""
                [
                    [
                        1099040000000,
                        "0.01634790",
                        "0.80000000",
                        "0.01575800",
                        "0.01577100",
                        "148976.11427815",
                        1499644799999,
                        "2434.19055334",
                        308,
                        "1756.87402397",
                        "28.46694368",
                        "0"
                    ]
                ]
                """,
            )
            Kline.load_all_klines()
            self.assertEqual(Kline.objects.all().count(), 1)
            Kline.load_all_klines(symbols=[self.s1])
            self.assertEqual(Kline.objects.all().count(), 1)


class TestViews(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1, _ = Symbol.objects.update_or_create(
            symbol="S1BUSD",
            status="TRADING",
            base_asset="S1",
            quote_asset="BUSD",
            defaults={
                "model_score": 0.99,
                "volume_quote_asset": Decimal(1000000),
                "variation_range_24h": Decimal(9),
                "info": {
                    "filters": [
                        {},
                        {"stepSize": "0.10000000", "filterType": "LOT_SIZE"},
                    ]
                },
            },
        )
        with requests_mock.Mocker() as m:
            with open("base/fixtures/klines_response_mock.json") as f:
                content = f.read()
                m.get(
                    f"{BINANCE_API_URL}/api/v3/klines?symbol=S1BUSD"
                    f"&interval={settings.TIME_INTERVAL}m",
                    text=content,
                )
                cls.s1.retrieve_and_update()
        cls.user1, _ = User.objects.get_or_create(
            username="user1", password="secret", email="admin@example.com"
        )
        cls.user2, _ = User.objects.get_or_create(
            username="user2", password="secret", email="admin@example.com"
        )
        cls.group1, _ = TraderoBotGroup.objects.get_or_create(
            user=cls.user1, name="Test Group 1"
        )
        cls.group2, _ = TraderoBotGroup.objects.get_or_create(
            user=cls.user1, name="Test Group 2"
        )
        cls.bot1 = TraderoBot(
            group=cls.group1,
            symbol=cls.s1,
            user=cls.user1,
            fund_quote_asset_initial=Decimal("20"),
        )
        cls.bot1.save()
        cls.bot1.on()
        cls.bot2 = TraderoBot(
            group=cls.group1,
            symbol=cls.s1,
            user=cls.user1,
            fund_quote_asset_initial=Decimal("20"),
        )
        cls.bot2.save()
        cls.bot2.on()
        cls.bot3 = TraderoBot(
            group=cls.group1,
            symbol=cls.s1,
            user=cls.user1,
            fund_quote_asset_initial=Decimal("20"),
        )
        cls.bot3.save()
        cls.bot3.on()
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/ticker/price",
                json={"symbol": "S1BUSD", "price": "1.0"},
            )
            cls.bot1.buy()
            cls.bot1.sell()
            cls.bot1.buy()
            cls.bot1.price_current = cls.bot1.price_buying
            cls.bot1.save()
        super().setUpClass()

    def test_home(self):
        url = reverse("base:home")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

    def test_instrucoes(self):
        url = reverse("base:instrucoes")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

    def test_user_detail(self):
        self.client.force_login(self.user1)
        url = reverse("base:users-detail")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

    def test_user_update(self):
        self.client.force_login(self.user1)
        url = reverse("base:users-update")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        response = self.client.post(url, {"first_name": "Test"}, follow=True)
        self.assertEqual(response.status_code, 200)
        self.user1.refresh_from_db()
        self.assertEqual(self.user1.first_name, "Test")

    def test_botzinhos_list(self):
        self.client.force_login(self.user1)
        url = reverse("base:botzinhos-list")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

    def test_botzinhos_logs(self):
        self.client.force_login(self.user1)
        url = reverse("base:botzinhos-logs", args=[self.bot1.pk])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

    def test_botzinhos_detail(self):
        self.client.force_login(self.user2)
        url = reverse("base:botzinhos-detail", args=[self.bot1.pk])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)
        self.client.force_login(self.user1)
        self.bot1.off()
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.bot1.on()
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/ticker/price",
                json={"symbol": "S1BUSD", "price": "1.0"},
            )
            self.bot1.sell()
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.bot1.off()
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

    def test_botzinhos_create(self):
        self.client.force_login(self.user1)
        url = reverse("base:botzinhos-create")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f"{url}?group={self.group1.pk}")
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"selected>Test Group 1",
            response.content,
        )
        response = self.client.post(
            url,
            {
                "bots_quantity": 1,
                "name": "testing botzinho",
                "group": self.group1.pk,
                "symbol": self.s1.pk,
                "strategy": "acmadness",
                "strategy_params": "microgain=0.3",
                "fund_quote_asset_initial": Decimal("20"),
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            TraderoBot.objects.filter(
                name__startswith="testing botzinho"
            ).count(),
            1,
        )
        self.user1.bot_quota = 1
        self.user1.save()
        response = self.client.post(
            url,
            {
                "bots_quantity": 1,
                "name": "testing botzinho",
                "group": self.group1.pk,
                "symbol": self.s1.pk,
                "strategy": "acmadness",
                "fund_quote_asset_initial": Decimal("20"),
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Bot Quota (1) Exceeded.",
            response.content,
        )
        self.assertEqual(
            TraderoBot.objects.filter(
                name__startswith="testing botzinho"
            ).count(),
            1,
        )
        self.user1.bot_quota = 0
        self.user1.save()
        response = self.client.post(
            url,
            {
                "bots_quantity": 1,
                "name": "testing botzinho 2",
                "group": self.group1.pk,
                "symbol": self.s1.pk,
                "strategy": "acmadness",
                "strategy_params": "microgain=0.3",
                "fund_quote_asset_initial": Decimal("20"),
                "jumpy_whitelist": "s1busd,s8978ygbusd",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Unrecognized Symbols",
            response.content,
        )
        response = self.client.post(
            url,
            {
                "bots_quantity": 1,
                "name": "testing botzinho 2",
                "group": self.group1.pk,
                "symbol": self.s1.pk,
                "strategy": "acmadness",
                "strategy_params": "microgain=0.3",
                "fund_quote_asset_initial": Decimal("20"),
                "jumpy_blacklist": "sghkgjhvbusd",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Unrecognized Symbols",
            response.content,
        )
        response = self.client.post(
            url,
            {
                "bots_quantity": 2,
                "group": self.group1.pk,
                "symbol": self.s1.pk,
                "strategy": "acmadness",
                "strategy_params": "microgain=0.3",
                "fund_quote_asset_initial": Decimal("3.1416") * 10,
                "jumpy_whitelist": "s1busd",
                "jumpy_blacklist": "s1busd",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            TraderoBot.objects.filter(
                fund_quote_asset_initial=Decimal("3.1416") * 10
            ).count(),
            2,
        )
        response = self.client.post(
            url,
            {
                "bots_quantity": 1,
                "name": "testing botzinho 2",
                "group": self.group1.pk,
                "symbol": self.s1.pk,
                "strategy": "acmadness",
                "strategy_params": "microgain=0.3",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"This field is required.",
            response.content,
        )
        response = self.client.post(
            url,
            {
                "bots_quantity": 1,
                "name": "testing botzinho 2",
                "group": self.group1.pk,
                "symbol": self.s1.pk,
                "strategy": "acmadness",
                "strategy_params": "microgain=0.3,abc123",
                "fund_quote_asset_initial": Decimal("20"),
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Unrecognized parameters",
            response.content,
        )
        response = self.client.post(
            url,
            {
                "bots_quantity": 1,
                "name": "testing botzinho 2",
                "group": self.group1.pk,
                "symbol": self.s1.pk,
                "strategy": "acmadness",
                "strategy_params": "microgain=0.3,abc123=1",
                "fund_quote_asset_initial": Decimal("20"),
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Unrecognized parameters",
            response.content,
        )

    def test_botzinhos_update(self):
        self.client.force_login(self.user1)
        url = reverse("base:botzinhos-update", args=[self.bot1.pk])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        form = TraderoBotForm(user=self.user1, instance=self.bot1)
        form.initial.update(name="testing botzinho update")
        response = self.client.post(
            url,
            {key: value for key, value in form.initial.items() if value},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            TraderoBot.objects.filter(name="testing botzinho update").count(),
            1,
        )

    def test_botzinhos_group_detail(self):
        self.client.force_login(self.user2)
        url = reverse("base:botzinhos-group-detail", args=[self.group1.pk])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    def test_botzinhos_group_create(self):
        self.client.force_login(self.user1)
        url = reverse("base:botzinhos-group-create")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        response = self.client.post(
            url,
            {
                "name": "testing botzinhos group",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            TraderoBotGroup.objects.filter(
                name="testing botzinhos group"
            ).count(),
            1,
        )
        response = self.client.post(
            url,
            {
                "name": "testing group bot create",
                "add_edit_bots": True,
                "bots_quantity": 2,
                "bot_name": "tt",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"This field is required",
            response.content,
        )
        response = self.client.post(
            url,
            {
                "name": "testing group bot create",
                "add_edit_bots": True,
                "bots_quantity": 2,
                "bot_name": "tt",
                "bot_strategy": "acmadness",
                "bot_symbol": self.s1.pk,
                "bot_fund_quote_asset_initial": 20,
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            TraderoBot.objects.filter(name__startswith="tt").count(),
            2,
        )
        response = self.client.post(
            url,
            {
                "name": "testing group bot create 2",
                "add_edit_bots": True,
                "bots_quantity": 2,
                "bot_strategy": "acmadness",
                "bot_symbol": self.s1.pk,
                "bot_fund_quote_asset_initial": 55,
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            TraderoBot.objects.filter(
                name__startswith="BT", fund_quote_asset_initial=55
            ).count(),
            2,
        )
        self.user1.bot_quota = 1
        self.user1.save()
        response = self.client.post(
            url,
            {
                "name": "testing group bot create 2",
                "add_edit_bots": True,
                "bots_quantity": 2,
                "bot_strategy": "acmadness",
                "bot_symbol": self.s1.pk,
                "bot_fund_quote_asset_initial": 55,
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Bot Quota (1) Exceeded.",
            response.content,
        )
        self.assertEqual(
            TraderoBot.objects.filter(
                name__startswith="BT", fund_quote_asset_initial=55
            ).count(),
            2,
        )

    def test_botzinhos_group_update(self):
        self.client.force_login(self.user1)
        url = reverse("base:botzinhos-group-update", args=[self.group1.pk])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        response = self.client.post(
            url,
            {
                "name": "testing botzinhos group update",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            TraderoBotGroup.objects.filter(
                name="testing botzinhos group update"
            ).count(),
            1,
        )
        response = self.client.post(
            url,
            {
                "add_edit_bots": True,
                "bot_name": "Group 1",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            TraderoBot.objects.filter(name__startswith="Group 1").count(),
            3,
        )
        response = self.client.post(
            url,
            {
                "add_edit_bots": True,
                "bot_fund_quote_asset_initial": 25,
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            TraderoBot.objects.filter(
                name__startswith="Group 1", fund_quote_asset_initial=25
            ).count(),
            3,
        )
        response = self.client.post(
            url,
            {
                "add_edit_bots": True,
                "bot_fund_quote_asset_initial": 5,
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"The following fields produce erros",
            response.content,
        )

    def test_botzinhos_actions(self):
        self.client.force_login(self.user1)
        url = reverse(
            "base:botzinhos-action",
            kwargs={
                "pk": self.bot1.pk,
                "action": "start",
            },
        )
        response = self.client.post(url, follow=True)
        self.assertEqual(response.status_code, 404)
        url = reverse(
            "base:botzinhos-action",
            kwargs={
                "pk": self.bot1.pk,
                "action": "on",
            },
        )
        response = self.client.post(url, follow=True)
        self.assertEqual(response.status_code, 200)
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/ticker/price",
                json={"symbol": "S1BUSD", "price": "1.0"},
            )
            url = reverse(
                "base:botzinhos-action",
                kwargs={
                    "pk": self.bot1.pk,
                    "action": "jump",
                },
            )
            response = self.client.post(
                url, {"to_symbol": self.s1.pk}, follow=True
            )
            self.assertEqual(response.status_code, 200)
            self.bot1.refresh_from_db()
            self.assertIn("Holding JUMP", self.bot1.others["last_logs"][-1])
            self.assertIn(b"SUCCESS at", response.content)
            with mock.patch(
                "base.views.BotzinhosActionView.ACTIONS",
                {"on": {"params": {"test_param": {"type": "test"}}}},
            ):
                url = reverse(
                    "base:botzinhos-action",
                    kwargs={
                        "pk": self.bot1.pk,
                        "action": "on",
                    },
                )
                response = self.client.post(url, follow=True)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"SUCCESS at", response.content)
        with mock.patch("base.models.TraderoBot.on") as bot_on_mock:
            bot_on_mock.side_effect = Exception("New Exception")
            url = reverse(
                "base:botzinhos-action",
                kwargs={
                    "pk": self.bot1.pk,
                    "action": "on",
                },
            )
            response = self.client.post(url, follow=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"ERROR at", response.content)

    def test_botzinhos_group_actions(self):
        self.client.force_login(self.user1)
        self.bot3.off()
        url = reverse(
            "base:botzinhos-group-action",
            kwargs={
                "pk": self.group1.pk,
                "action": "off",
            },
        )
        response = self.client.post(url, follow=True)
        self.assertEqual(response.status_code, 200)
        self.bot1.refresh_from_db()
        self.bot2.refresh_from_db()
        self.assertEqual(self.bot1.status, TraderoBot.Status.INACTIVE)
        self.assertEqual(self.bot2.status, TraderoBot.Status.INACTIVE)
        self.bot3.on()
        url = reverse(
            "base:botzinhos-group-action",
            kwargs={
                "pk": self.group1.pk,
                "action": "on",
            },
        )
        response = self.client.post(url, follow=True)
        self.assertEqual(response.status_code, 200)
        self.bot1.refresh_from_db()
        self.bot2.refresh_from_db()
        self.assertEqual(self.bot1.status, TraderoBot.Status.SELLING)
        self.assertEqual(self.bot2.status, TraderoBot.Status.BUYING)
        self.bot3.off()
        url = reverse(
            "base:botzinhos-group-action",
            kwargs={
                "pk": self.group1.pk,
                "action": "liquidate",
            },
        )
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/ticker/price",
                json={"symbol": "S1BUSD", "price": "1.0"},
            )
            response = self.client.post(url, follow=True)
            self.assertEqual(response.status_code, 200)
        self.bot1.refresh_from_db()
        self.bot2.refresh_from_db()
        self.assertEqual(self.bot1.status, TraderoBot.Status.INACTIVE)
        self.assertEqual(self.bot2.status, TraderoBot.Status.INACTIVE)
        self.assertIn("Sold", self.bot1.others["last_logs"][-2])
        self.assertIn("OFF", self.bot1.others["last_logs"][-1])
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/ticker/price",
                json={"symbol": "S1BUSD", "price": "1.0"},
            )
            self.bot2.buy()
            url = reverse(
                "base:botzinhos-group-action",
                kwargs={
                    "pk": self.group1.pk,
                    "action": "jump",
                },
            )
            response = self.client.post(
                url, {"to_symbol": self.s1.pk}, follow=True
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"SUCCESS at", response.content)
        url = reverse(
            "base:botzinhos-group-action",
            kwargs={
                "pk": self.group1.pk,
                "action": "reset_soft",
            },
        )
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/ticker/price",
                json={"symbol": "S1BUSD", "price": "1.0"},
            )
            response = self.client.post(url, follow=True)
            self.assertEqual(response.status_code, 200)
        self.bot1.refresh_from_db()
        self.bot2.refresh_from_db()
        self.bot3.refresh_from_db()
        self.assertEqual(self.bot1.status, TraderoBot.Status.BUYING)
        self.assertEqual(self.bot2.status, TraderoBot.Status.SELLING)
        self.assertEqual(self.bot3.status, TraderoBot.Status.BUYING)

    def test_botzinhos_group_move(self):
        self.client.force_login(self.user1)
        url = reverse(
            "base:botzinhos-group-move",
            kwargs={"pk": self.group1.pk},
        )
        response = self.client.get(
            url,
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.post(
            url,
            {
                "to_group": self.group2.pk,
                "bots": [self.bot3.pk],
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.bot3.refresh_from_db()
        self.assertEqual(self.bot3.group, self.group2)


class TestAdmin(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = Client()
        cls.superuser = User.objects.create_superuser(
            username="superuser", password="secret", email="admin@example.com"
        )
        cls.ws_client, _ = WSClient.objects.get_or_create(
            channel_name="a_test_channel"
        )
        cls.s1, _ = Symbol.objects.get_or_create(
            symbol="S1BUSD",
            status="TRADING",
            base_asset="S1",
            quote_asset="BUSD",
        )
        cls.group1, _ = TraderoBotGroup.objects.get_or_create(
            user=cls.superuser, name="Test Group 1"
        )
        # Review why get_or_create does not work (probably because of save)
        cls.bot1 = TraderoBot(
            symbol=cls.s1,
            user=cls.superuser,
            group=cls.group1,
            fund_quote_asset_initial=Decimal("20"),
        )
        cls.bot1.save()
        cls.bot1.on()
        cls.bot2 = TraderoBot(
            symbol=cls.s1,
            user=cls.superuser,
            group=cls.group1,
            fund_quote_asset_initial=Decimal("20"),
        )
        cls.bot2.save()
        super().setUpClass()

    def test_ws_client_admin(self):
        self.client.force_login(self.superuser)
        url = reverse("admin:base_wsclient_changelist")
        response = self.client.get(url, follow=True)
        self.assertEqual(response.status_code, 200)
        response = self.client.get(url + "?is_open=true", follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"a_test_channel", response.content)
        response = self.client.get(url + "?is_open=false", follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"a_test_channel", response.content)

    def test_other_admin(self):
        self.client.force_login(self.superuser)
        # Test UserAdmmin
        url = reverse("admin:base_user_changelist")
        response = self.client.get(url, follow=True)
        self.assertEqual(response.status_code, 200)
        url = reverse("admin:base_user_change", args=[self.superuser.pk])
        response = self.client.get(url, follow=True)
        self.assertEqual(response.status_code, 200)
        # Test TraderoBotAdmmin
        url = reverse("admin:base_traderobot_changelist")
        response = self.client.get(url, follow=True)
        self.assertEqual(response.status_code, 200)
        # Test TraderoBotLogAdmmin
        url = reverse("admin:base_traderobotlog_changelist")
        response = self.client.get(url, follow=True)
        self.assertEqual(response.status_code, 200)


class TestCommands(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1, _ = Symbol.objects.update_or_create(
            symbol="S1BUSD",
            status="TRADING",
            base_asset="S1",
            quote_asset="BUSD",
            defaults={
                "model_score": None,
                "volume_quote_asset": None,
            },
        )
        super().setUpClass()

    @pytest.mark.xdist_group("commands")
    def test_load_symbols(self):
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/exchangeInfo",
                text="""
                {
                  "timezone": "UTC",
                  "serverTime": 1565246363776,
                  "rateLimits": [{}],
                  "exchangeFilters": [],
                  "symbols": [
                    {
                      "symbol": "S1BUSD",
                      "status": "TRADING",
                      "baseAsset": "S1",
                      "baseAssetPrecision": 8,
                      "quoteAsset": "BUSD",
                      "quotePrecision": 8,
                      "quoteAssetPrecision": 8,
                      "orderTypes": [
                        "LIMIT",
                        "LIMIT_MAKER",
                        "MARKET",
                        "STOP_LOSS",
                        "STOP_LOSS_LIMIT",
                        "TAKE_PROFIT",
                        "TAKE_PROFIT_LIMIT"
                      ],
                      "icebergAllowed": true,
                      "ocoAllowed": true,
                      "quoteOrderQtyMarketAllowed": true,
                      "allowTrailingStop": false,
                      "cancelReplaceAllowed": false,
                      "isSpotTradingAllowed": true,
                      "isMarginTradingAllowed": true,
                      "filters": [],
                      "permissions": [
                         "SPOT",
                         "MARGIN"
                      ],
                      "defaultSelfTradePreventionMode": "NONE",
                      "allowedSelfTradePreventionModes": [
                        "NONE"
                      ]
                    },
                    {
                      "symbol": "S1USDT",
                      "status": "TRADING",
                      "baseAsset": "S1",
                      "quoteAsset": "USDT"
                    },
                    {
                      "symbol": "S2BUSD",
                      "status": "TRADING",
                      "baseAsset": "S1",
                      "quoteAsset": "BUSD"
                    }
                  ]
                }
                """,
            )
            out = StringIO()
            call_command("load_symbols", stdout=out)
            self.assertIn("Successfully processed", out.getvalue())
            self.assertEqual(Symbol.objects.all().count(), 2)
            call_command("load_symbols", stdout=out)

    @pytest.mark.xdist_group("commands")
    @override_settings(
        PREDICTION_ENABLED=False,
        OUTLIERS_ENABLED=False,
        INDICATORS=["macdcg"],
    )
    def test_warm_and_ready(self):
        with requests_mock.Mocker() as m:
            with open("base/fixtures/klines_response_mock.json") as f:
                content = f.read()
                m.get(
                    f"{BINANCE_API_URL}/api/v3/klines?symbol=S1BUSD"
                    f"&interval={settings.TIME_INTERVAL}m",
                    text=content,
                )
                m.get(
                    f"{BINANCE_API_URL}/api/v3/klines?symbol=S2BUSD"
                    f"&interval={settings.TIME_INTERVAL}m",
                    text=content,
                )
            out = StringIO()
            call_command("warm_and_ready", periods=1, stdout=out)
            self.assertIn("Successfully", out.getvalue())

    @pytest.mark.xdist_group("commands")
    def test_ws_klines(self):
        out = StringIO()
        with self.assertRaises(TimeoutError):
            with timeout(2):
                call_command("ws_klines", stdout=out)
        with self.assertRaises(TimeoutError):
            with timeout(2):
                call_command(
                    "ws_klines", reset_ms=True, clean=False, stdout=out
                )

    @pytest.mark.xdist_group("commands")
    def test_calibrate_all_models(self):
        # Test command call without data
        out = StringIO()
        call_command("calibrate_all_models", stdout=out)
        self.assertLogs("ALL MODELS CALIBRATION COMPLETE")
        self.assertIn("Successfully", out.getvalue())

    @pytest.mark.xdist_group("commands")
    def test_calibrate_all_windows(self):
        # Test command call without data
        out = StringIO()
        call_command("calibrate_all_windows", stdout=out)
        self.assertLogs("ALL WINDOWS CALIBRATION COMPLETE")
        self.assertIn("Successfully", out.getvalue())

    @pytest.mark.xdist_group("commands")
    @override_settings(
        DUMMY_USER_ENABLED=True,
        DUMMY_USER_SYMBOL="S1",
        DUMMY_USER_BOTS=2,
        DUMMY_USER_BOT_QUOTA=3,
    )
    def test_dummy_user(self):
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/ticker/price",
                json={"symbol": "S1BUSD", "price": "1.0"},
            )
            with open("base/fixtures/klines_response_mock.json") as f:
                content = f.read()
                m.get(f"{BINANCE_API_URL}/api/v3/klines", text=content)
            out = StringIO()
            call_command("dummy_user", stdout=out)
            self.assertIn("SUCCESS", out.getvalue())
            self.assertEqual(User.objects.filter(username="dummy").count(), 1)
            call_command("dummy_user", reset=False, stdout=out)
            self.assertIn("SUCCESS", out.getvalue())
            self.assertEqual(User.objects.filter(username="dummy").count(), 1)
            call_command("dummy_user", remove=True, stdout=out)
            self.assertIn("SUCCESS", out.getvalue())
            self.assertEqual(User.objects.filter(username="dummy").count(), 0)
            with override_settings(DUMMY_USER_ENABLED=False):
                call_command("dummy_user", reset=True, stdout=out)
            self.assertIn("SUCCESS", out.getvalue())
            self.assertEqual(User.objects.filter(username="dummy").count(), 0)


class TestSymbols(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1, _ = Symbol.objects.get_or_create(
            symbol="S1BUSD",
            status="TRADING",
            base_asset="S1",
            quote_asset="BUSD",
        )
        with requests_mock.Mocker() as m:
            with open("base/fixtures/klines_response_mock.json") as f:
                content = f.read()
                m.get(
                    f"{BINANCE_API_URL}/api/v3/klines?symbol=S1BUSD"
                    f"&interval={settings.TIME_INTERVAL}m",
                    text=content,
                )
                cls.s1.retrieve_and_update()
        super().setUpClass()

    @pytest.mark.xdist_group("symbols")
    def test_load_all_data(self):
        with requests_mock.Mocker() as m:
            with open("base/fixtures/klines_response_mock.json") as f:
                content = f.read()
                m.get(
                    f"{BINANCE_API_URL}/api/v3/klines?symbol=S1BUSD"
                    f"&interval={settings.TIME_INTERVAL}m",
                    text=content,
                )
                Symbol.load_all_data()
                self.assertLogs("Data loading complete")

    @pytest.mark.xdist_group("symbols")
    def test_calibrate_all_windows(self):
        Symbol.calibrate_all_windows()
        self.assertLogs("ALL WINDOWS CALIBRATION COMPLETE")

    @pytest.mark.xdist_group("symbols")
    def test_calibrate_all_models(self):
        Symbol.calibrate_all_models()
        self.assertLogs("ALL MODELS CALIBRATION COMPLETE")


class TestTraderoMixin(TestCase):
    @classmethod
    @override_settings(
        INDICATORS=["__all__"],
    )
    def setUpClass(cls):
        cls.s1, _ = Symbol.objects.get_or_create(
            symbol="S1BUSD",
            status="TRADING",
            base_asset="S1",
            quote_asset="BUSD",
        )
        with requests_mock.Mocker() as m:
            with open("base/fixtures/klines_response_mock.json") as f:
                content = f.read()
                m.get(
                    f"{BINANCE_API_URL}/api/v3/klines?symbol=S1BUSD"
                    f"&interval={settings.TIME_INTERVAL}m",
                    text=content,
                )
                cls.s1.load_data()
                cls.s1.update_indicators(push=False)
        super().setUpClass()

    def test_h_predict(self):
        tds = self.s1.training_data.all()[:2]
        os = self.s1.get_outlier_classifiers()
        tds = [os[0].td_to_dict(td) for td in tds]
        self.assertEqual(list(os[0].h_predict(tds)), [1, 1])


class TestConsumers(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1, _ = Symbol.objects.get_or_create(
            symbol="S1BUSD",
            status="TRADING",
            base_asset="S1",
            quote_asset="BUSD",
        )
        cls.user, _ = User.objects.get_or_create(
            username="user", password="secret", email="admin@example.com"
        )
        cls.group1, _ = TraderoBotGroup.objects.get_or_create(
            user=cls.user, name="Test Group 1"
        )
        cls.bot = TraderoBot(
            user=cls.user,
            symbol=cls.s1,
            name="BTZN",
            group=cls.group1,
            fund_quote_asset_initial=Decimal("20"),
        )
        cls.bot.save()
        cls.channel_layer = get_channel_layer()

        super().setUpClass()

    async def test_symbol_consumer(self):
        communicator = WebsocketCommunicator(
            SymbolHTMLConsumer.as_asgi(), "/ws/symbols/html"
        )
        connected, subprotocol = await communicator.connect()
        self.assertTrue(connected)
        await communicator.send_to(text_data='{"message": "hello"}')
        response = await communicator.receive_from()
        self.assertEqual(response, '{"message": "hello"}')
        await self.s1.push_to_ws()
        response = await communicator.receive_from(timeout=15)
        self.assertTrue("S1BUSD" in response)
        # Close
        await communicator.disconnect()

    async def test_symbol_json_consumer(self):
        communicator = WebsocketCommunicator(
            SymbolJSONConsumer.as_asgi(), "/ws/symbols/json"
        )
        connected, subprotocol = await communicator.connect()
        self.assertTrue(connected)
        await communicator.send_to(text_data='{"message": "hello"}')
        response = await communicator.receive_from()
        self.assertEqual(response, '{"message": "hello"}')
        await self.s1.push_to_ws()
        response = await communicator.receive_from(timeout=15)
        self.assertTrue("S1BUSD" in response)
        # Close
        await communicator.disconnect()

    async def test_bot_consumer(self):
        await sync_to_async(self.client.force_login)(self.user)
        communicator = WebsocketCommunicator(
            BotHTMLConsumer.as_asgi(), "/ws/bots/html"
        )
        communicator.scope["user"] = self.user
        connected, subprotocol = await communicator.connect()
        self.assertTrue(connected)
        await communicator.send_to(text_data='{"message": "hello"}')
        response = await communicator.receive_from()
        self.assertEqual(response, '{"message": "hello"}')
        await self.bot.push_to_ws()
        response = await communicator.receive_from(timeout=15)
        self.assertTrue("BTZN" in response)
        # Close
        await communicator.disconnect()


class TestHandlers(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1, _ = Symbol.objects.get_or_create(
            symbol="S1BUSD",
            status="TRADING",
            base_asset="S1",
            quote_asset="BUSD",
        )
        with requests_mock.Mocker() as m:
            with open("base/fixtures/klines_response_mock.json") as f:
                content = f.read()
                m.get(
                    f"{BINANCE_API_URL}/api/v3/klines?symbol=S1BUSD"
                    f"&interval={settings.TIME_INTERVAL}m",
                    text=content,
                )
                Symbol.load_all_data()
        super().setUpClass()

    async def test_ws_handler(self):
        message_1 = """{
            "data" :{
              "e": "kline",
              "E": 123456789,
              "s": "S1BUSD",
              "k": {
                "t": 1684765200000,
                "T": 1684769200000,
                "s": "S1BUSD",
                "i": "5m",
                "f": 100,
                "L": 200,
                "o": "0.0010",
                "c": "0.0020",
                "h": "0.0025",
                "l": "0.0015",
                "v": "1000",
                "n": 100,
                "x": true,
                "q": "1.0000",
                "V": "500",
                "Q": "0.500",
                "B": "123456"
              }
            }
        }"""
        await message_handler(message_1, None)
        symbol = await sync_to_async(Symbol.objects.last)()
        self.assertEqual(symbol.last_variation, 100)
        message_2 = """{
            "data" :{
              "e": "kline",
              "E": 123456789,
              "s": "S1BUSD",
              "k": {
                "t": 1684765300000,
                "T": 1684769200000,
                "s": "S1BUSD",
                "i": "5m",
                "f": 100,
                "L": 200,
                "o": "0.0010",
                "c": "0.0020",
                "h": "0.0025",
                "l": "0.0015",
                "v": "1000",
                "n": 100,
                "x": false,
                "q": "1.0000",
                "V": "500",
                "Q": "0.500",
                "B": "123456"
              }
            }
        }"""
        await message_handler(message_2, None)
        symbol = await sync_to_async(Symbol.objects.last)()
        self.assertEqual(symbol.last_variation, 100)


class BotTestCase(TestCase):
    @classmethod
    @override_settings(
        INDICATORS=["__all__"],
    )
    def setUpClass(cls):
        cls.s1, _ = Symbol.objects.update_or_create(
            symbol="S1BUSD",
            status="TRADING",
            base_asset="S1",
            quote_asset="BUSD",
            defaults={
                "model_score": 0.99,
                "volume_quote_asset": Decimal(1000000),
                "variation_range_24h": Decimal(9),
                "info": {
                    "filters": [
                        {},
                        {"stepSize": "0.10000000", "filterType": "LOT_SIZE"},
                    ]
                },
            },
        )
        cls.s2, _ = Symbol.objects.update_or_create(
            symbol="S2BUSD",
            status="TRADING",
            base_asset="S2",
            quote_asset="BUSD",
            defaults={
                "model_score": 0.99,
                "volume_quote_asset": Decimal(1000000),
                "variation_range_24h": Decimal(9),
                "info": {
                    "filters": [
                        {},
                        {"stepSize": "0.10000000", "filterType": "LOT_SIZE"},
                    ]
                },
            },
        )
        cls.s3, _ = Symbol.objects.update_or_create(
            symbol="S3BUSD",
            status="TRADING",
            base_asset="S3",
            quote_asset="BUSD",
            defaults={
                "model_score": 0.99,
                "volume_quote_asset": Decimal(1000000),
                "variation_range_24h": Decimal(9),
                "info": {
                    "filters": [
                        {},
                        {"stepSize": "0.10000000", "filterType": "LOT_SIZE"},
                    ]
                },
            },
        )
        cls.user1, _ = User.objects.get_or_create(
            username="user1", password="secret", email="admin@example.com"
        )
        cls.group1, _ = TraderoBotGroup.objects.get_or_create(
            user=cls.user1, name="BotTest Group 1"
        )
        cls.bot1, _ = TraderoBot.objects.get_or_create(
            name="test_bot1",
            symbol=cls.s1,
            user=cls.user1,
            group=cls.group1,
            strategy="acmadness",
            fund_quote_asset_initial=20,
            is_dummy=True,
            is_jumpy=True,
        )
        cls.bot1.on()
        cls.bot2, _ = TraderoBot.objects.get_or_create(
            name="test_bot2",
            symbol=cls.s3,
            user=cls.user1,
            group=cls.group1,
            strategy="acmadness",
            strategy_params="microgain=0.3",
            fund_quote_asset_initial=20,
            is_dummy=True,
            is_jumpy=True,
        )
        cls.bot2.on()
        with requests_mock.Mocker() as m:
            with open("base/fixtures/klines_response_mock.json") as f:
                content = f.read()
                m.get(
                    f"{BINANCE_API_URL}/api/v3/klines?symbol=S1BUSD"
                    f"&interval={settings.TIME_INTERVAL}m",
                    text=content,
                )
                m.get(
                    f"{BINANCE_API_URL}/api/v3/klines?symbol=S2BUSD"
                    f"&interval={settings.TIME_INTERVAL}m",
                    text=content,
                )
                m.get(
                    f"{BINANCE_API_URL}/api/v3/klines?symbol=S3BUSD"
                    f"&interval={settings.TIME_INTERVAL}m",
                    text=content,
                )
            Symbol.update_all_indicators()
        cls.s1.refresh_from_db()
        cls.s2.refresh_from_db()
        cls.s3.refresh_from_db()
        cls.s1.volume_quote_asset = 1000000
        cls.s2.volume_quote_asset = 1000000
        cls.s3.volume_quote_asset = 1000000
        cls.s1.save()
        cls.s2.save()
        cls.s3.save()
        super().setUpClass()


class TestTraderoBots(BotTestCase):
    def test_update_all_bots(self):
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/ticker/price",
                json=[
                    {"symbol": "S1BUSD", "price": "1.0"},
                    {"symbol": "S2BUSD", "price": "1.0"},
                    {"symbol": "S3BUSD", "price": "1.0"},
                ],
            )
            TraderoBot.update_all_bots()
            self.bot2.refresh_from_db()
            self.assertEqual(self.bot2.price_current, 1)

    def test_botzinhos_actions(self):
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/ticker/price",
                json={"symbol": "S1BUSD", "price": "1.0"},
            )
            self.bot1.off()
            self.assertEqual(self.bot1.status, TraderoBot.Status.INACTIVE)
            self.bot1.on()
            self.assertEqual(self.bot1.status, TraderoBot.Status.BUYING)
            self.bot1.buy()
            self.assertEqual(self.bot1.receipt_buying is not None, True)
            self.assertEqual(self.bot1.price_buying is not None, True)
            self.assertEqual(self.bot1.status, TraderoBot.Status.SELLING)
            # Test restore of state
            self.bot1.off()
            self.bot1.on()
            self.assertEqual(self.bot1.status, TraderoBot.Status.SELLING)
            self.bot1.sell()
            self.assertEqual(self.bot1.trades.count(), 1)
            self.assertEqual(self.bot1.status, TraderoBot.Status.BUYING)
            # Test conserving the state
            self.bot1.off()
            self.bot1.buy()
            self.assertEqual(self.bot1.status, TraderoBot.Status.INACTIVE)
            self.bot1.sell()
            # Test exception while buying
            with mock.patch(
                "base.client.TraderoClient.ticker_price"
            ) as client_tp_mock:
                client_tp_mock.side_effect = Exception("New Exception")
                self.bot1.buy()
                self.assertIn(
                    "New Exception", self.bot1.others["last_logs"][-1]
                )
            self.bot1.should_stop = True
            self.bot1.should_reinvest = False
            self.bot1.buy()
            self.bot1.sell()
            self.assertEqual(
                self.bot1.fund_quote_asset, self.bot1.fund_quote_asset_initial
            )
            self.assertEqual(self.bot1.status, TraderoBot.Status.INACTIVE)
            self.bot1.fund_base_asset = 5
            # Test exception while selling
            with mock.patch(
                "base.client.TraderoClient.ticker_price"
            ) as client_tp_mock:
                client_tp_mock.side_effect = Exception("New Exception")
                self.bot1.sell()
                self.assertIn(
                    "New Exception", self.bot1.others["last_logs"][-1]
                )
            # Test already bought
            self.bot1.on()
            with mock.patch(
                "base.models.TraderoBot.log_trade"
            ) as bot_log_trade_mock:
                bot_log_trade_mock.side_effect = Exception("New Exception")
                self.bot1.buy()
                self.assertIn("Bought", self.bot1.others["last_logs"][-4])
                self.assertIn(
                    "New Exception", self.bot1.others["last_logs"][-3]
                )
                self.assertIn(
                    "Problems encountered", self.bot1.others["last_logs"][-2]
                )
                self.assertIn("OFF", self.bot1.others["last_logs"][-1])
                self.bot1.buy()
                self.assertIn(
                    "already done",
                    self.bot1.logs.all().order_by("-id")[3].message,
                )
                self.assertIn(
                    "New Exception", self.bot1.others["last_logs"][-3]
                )
                self.assertIn(
                    "Problems encountered", self.bot1.others["last_logs"][-2]
                )
                self.assertIn("OFF", self.bot1.others["last_logs"][-1])
            self.bot1.buy()
            self.assertIn("already done", self.bot1.others["last_logs"][-2])
            self.assertIn("Bought", self.bot1.others["last_logs"][-1])
            # Test already sold
            with mock.patch(
                "base.models.TraderoBot.log_trade"
            ) as bot_log_trade_mock:
                bot_log_trade_mock.side_effect = Exception("New Exception")
                self.bot1.sell()
                self.assertIn(
                    "New Exception", self.bot1.others["last_logs"][-3]
                )
                self.assertIn(
                    "Problems encountered", self.bot1.others["last_logs"][-2]
                )
                self.assertIn("OFF", self.bot1.others["last_logs"][-1])
                self.bot1.sell()
                self.assertIn(
                    "already done",
                    self.bot1.logs.all().order_by("-id")[3].message,
                )
                self.assertIn(
                    "New Exception", self.bot1.others["last_logs"][-3]
                )
                self.assertIn(
                    "Problems encountered", self.bot1.others["last_logs"][-2]
                )
                self.assertIn("OFF", self.bot1.others["last_logs"][-1])
            self.bot1.sell()
            self.assertIn("already done", self.bot1.others["last_logs"][-2])
            self.assertIn("Sold", self.bot1.others["last_logs"][-1])
            # Test hard reset
            self.bot1.reset_hard()
            self.assertIn("RESET - HARD", self.bot1.others["last_logs"][-1])
            self.assertEqual(self.bot1.status, TraderoBot.Status.INACTIVE)
            self.assertEqual(
                self.bot1.trades.last().timestamp_cancelled is not None, True
            )
            # Test soft reset
            self.bot1.buy()
            self.bot1.symbol = self.s2
            self.bot1.reset_soft()
            self.assertIn("RESET - SOFT", self.bot1.others["last_logs"][-4])
            self.assertEqual(self.bot1.symbol, self.s1)
            self.assertEqual(self.bot1.status, TraderoBot.Status.SELLING)
            with mock.patch(
                "base.models.TraderoBot.log_trade"
            ) as bot_log_trade_mock:
                bot_log_trade_mock.side_effect = Exception("New Exception")
                self.bot1.sell()
                self.assertEqual(self.bot1.status, TraderoBot.Status.INACTIVE)
            self.bot1.reset_soft()
            self.assertEqual(self.bot1.status, TraderoBot.Status.BUYING)
            self.bot1.buy()
            with mock.patch(
                "base.models.TraderoBot.log_trade"
            ) as bot_log_trade_mock:
                bot_log_trade_mock.side_effect = Exception("New Exception")
                self.bot1.symbol = self.s2
                self.bot1.sell()
                self.assertEqual(self.bot1.status, TraderoBot.Status.INACTIVE)
            self.bot1.reset_soft()
            self.assertIn(
                "ERROR - Hard reset or manual intervention required",
                self.bot1.others["last_logs"][-2],
            )
            self.assertEqual(self.bot1.status, TraderoBot.Status.INACTIVE)
            self.bot1.reset_hard()
            self.bot1.reset_soft()
            self.assertEqual(self.bot1.status, TraderoBot.Status.BUYING)
            self.bot1.reset_soft()
            self.assertEqual(self.bot1.status, TraderoBot.Status.BUYING)
            m.get(
                f"{BINANCE_API_URL}/api/v3/ticker/price",
                json={"symbol": "S2BUSD", "price": "2.0"},
            )
            self.bot1.fund_base_asset = 0.1
            self.bot1.jump(self.s2)
            self.assertEqual(self.bot1.symbol, self.s2)
            self.assertEqual(self.bot1.price_current, 2)
            self.assertIn(
                "leaving 0.100000 S1BUSD behind",
                self.bot1.others["last_logs"][-1],
            )
            # Test decide
            self.bot1.on()
            self.bot1.buy()
            self.bot1.decide()
            self.assertIn("below threshold", self.bot1.others["last_logs"][-1])
            self.bot1.price_current = 2.5
            self.bot1.decide()
            self.assertIn("BOOM!", self.bot1.others["last_logs"][-1])
            self.bot1.on()
            self.s2.others["stp"]["next_n_sum"] = 4
            self.bot1.symbol = self.s2
            self.bot1.decide()
            self.assertIn("Bought", self.bot1.others["last_logs"][-1])
            self.bot1.sell()
            self.bot1.on()
            self.s2.others["stp"]["next_n_sum"] = 0
            self.s1.others["stp"]["next_n_sum"] = 4
            self.s1.save()
            self.bot1.symbol = self.s2
            self.bot1.decide()
            self.assertIn("Jump", self.bot1.others["last_logs"][-3])
            self.assertIn("Bought", self.bot1.others["last_logs"][-1])
            self.assertIn("S1BUSD", self.bot1.others["last_logs"][-1])
            self.bot1.sell()
            self.bot1.on()
            self.s1.others["stp"]["next_n_sum"] = 0
            self.s1.save()
            self.bot1.symbol = self.s1
            self.bot1.is_jumpy = False
            self.bot1.decide()
            self.assertIn("below threshold", self.bot1.others["last_logs"][-1])

    def test_local_memory(self):
        self.assertTrue(self.bot1.has_local_memory() is None)
        self.bot1.set_local_memory(value={"test": [1]})
        lm = self.bot1.get_local_memory()
        self.assertEqual(lm, {"test": [1]})
        self.bot1.local_memory_reset()
        self.assertEqual(self.bot1.get_local_memory(), {})
        with mock.patch(
            "base.strategies.TradingStrategy.local_memory_update"
        ) as lmu_mock:
            lmu_mock.side_effect = self.bot1.set_local_memory(
                self.s1, {"test": [2]}
            )
            self.bot1.decide()
            self.assertEqual(self.bot1.get_local_memory(), {"test": [2]})
        with mock.patch(
            "base.strategies.TradingStrategy.has_local_memory"
        ) as hlm_mock:
            hlm_mock.return_value = {
                "test": [2]
            } == self.bot1.get_local_memory()
            self.assertTrue(self.bot1.has_local_memory() is True)
        # Test a running TraderoBot which hasn't been re-init'ed after local
        # memory has been introduced
        bot3 = TraderoBot(symbol=self.s1, status=TraderoBot.Status.BUYING)
        self.assertEqual(bot3.get_local_memory(), {})

    def test_valuation(self):
        self.bot2.reset_hard()
        self.assertEqual(self.bot2.fund_base_asset_executable, None)
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/ticker/price",
                json={"symbol": "S1BUSD", "price": "1.0"},
            )
            self.bot2.price_current = self.bot2.get_current_price()
            self.bot2.buy()
            self.assertEqual(
                self.bot2.fund_base_asset_executable, Decimal("19.9000")
            )
            self.bot2.sell()
            with mock.patch(
                "base.models.TraderoClient.ticker_price"
            ) as bot_log_trade_mock:
                bot_log_trade_mock.side_effect = Exception("New Exception")
                self.bot2.buy()
                self.assertIn(
                    "New Exception", self.bot2.others["last_logs"][-1]
                )
                self.assertEqual(
                    self.bot2.valuation_current, Decimal("19.8801")
                )


class TestStrategies(BotTestCase):
    def test_acmadness(self):
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/ticker/price",
                json={"symbol": "S1BUSD", "price": "1.0"},
            )
            self.bot1.strategy = "acmadness"
            self.bot1.symbol = self.s1
            #
            self.s1.others["stp"]["next_n_sum"] = 4
            self.s1.last_updated = timezone.now() - timezone.timedelta(
                minutes=2
            )
            self.bot1.on()
            self.bot1.decide()
            self.assertIn("Time Safeguard", self.bot1.others["last_logs"][-1])
            self.s1.last_updated = timezone.now()
            self.s1.others["outliers"]["o1"] = True
            self.bot1.decide()
            self.assertIn(
                "Outlier Protection", self.bot1.others["last_logs"][-1]
            )
            self.s1.others["outliers"]["o1"] = False
            self.s1.variation_range_24h = 11
            self.bot1.decide()
            self.assertIn("VR24h", self.bot1.others["last_logs"][-1])
            self.s1.variation_range_24h = 9
            self.s1.last_variation = Decimal("12")
            self.bot1.strategy_params = "microgain=0.3,max_var_prot=10"
            self.bot1.decide()
            self.assertIn(
                "Max Var Protection", self.bot1.others["last_logs"][-1]
            )
            self.s1.others["stp"]["next_n_sum"] = 0
            self.s2.others["stp"]["next_n_sum"] = 10
            self.s2.others["outliers"]["o1"] = True
            self.s2.save()
            self.bot1.decide()
            self.assertIn(
                "no other symbol to go", self.bot1.others["last_logs"][-1]
            )
            self.bot1.strategy_params = "microgain=0.3,ol_prot=0"
            self.bot1.jumpy_whitelist = None
            self.bot1.jumpy_blacklist = "S2BUSD"
            self.bot1.decide()
            self.assertIn(
                "no other symbol to go", self.bot1.others["last_logs"][-1]
            )
            self.bot1.jumpy_whitelist = "S1BUSD"
            self.bot1.jumpy_blacklist = None
            self.bot1.decide()
            self.assertIn(
                "no other symbol to go", self.bot1.others["last_logs"][-1]
            )
            self.bot1.jumpy_whitelist = "S2BUSD"
            self.s2.others["outliers"]["o1"] = False
            self.s2.save()
            self.bot1.decide()
            self.assertIn("Jumped", self.bot1.others["last_logs"][-3])
            self.assertIn("Bought", self.bot1.others["last_logs"][-1])
            self.bot1.strategy_params = (
                "microgain=0.3,ac_adjusted=0,keep_going=1"
            )
            self.s1.others["stp"]["next_n_sum"] = 10
            self.s1.last_updated = timezone.now()
            self.s1.save()
            self.bot1.symbol = self.s1
            self.bot1.status = self.bot1.Status.SELLING
            self.bot1.fund_base_asset = Decimal("20")
            self.bot1.price_current = Decimal("1.5")
            self.bot1.decide()
            self.assertIn("Kept goin'", self.bot1.others["last_logs"][-1])
            self.s1.others["stp"]["next_n_sum"] = 0
            self.bot1.decide()
            self.assertIn("Sold", self.bot1.others["last_logs"][-1])

    def test_catchthewave(self):
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/ticker/price",
                json={"symbol": "S1BUSD", "price": "1.0"},
            )
            self.bot1.symbol = self.s1
            self.bot1.strategy = "catchthewave"
            self.bot1.strategy_params = "use_local_memory=0"
            #
            self.s1.others["scg"]["current_good"] = True
            self.s1.others["scg"]["line_s_var"] = [1, 1, 1]
            self.bot1.on()
            self.bot1.decide()
            self.assertIn("Bought", self.bot1.others["last_logs"][-1])
            self.bot1.price_current = 0.9
            self.bot1.decide()
            self.assertIn("below min", self.bot1.others["last_logs"][-1])
            self.bot1.price_current = 1.1
            self.bot1.decide()
            self.assertIn("Kept goin'", self.bot1.others["last_logs"][-1])
            self.s1.others["scg"]["line_diff_sm"] = [
                0,
            ]
            self.bot1.strategy_params = "use_local_memory=0,sell_on_maxima=0"
            self.bot1.decide()
            self.assertIn("Sold", self.bot1.others["last_logs"][-1])
            self.bot1.buy()
            self.bot1.strategy_params = "sell_on_maxima=1"
            self.s1.others["scg"]["line_s_var"] = [1, 1, -0.11]
            self.bot1.decide()
            self.assertIn("Sold", self.bot1.others["last_logs"][-1])
            self.s1.others["scg"]["current_good"] = False
            self.s2.others["scg"]["current_good"] = True
            self.s2.others["scg"]["line_s_var"] = [1, 1, 2]
            self.s2.save()
            self.bot1.decide()
            self.assertIn("Jumped", self.bot1.others["last_logs"][-3])
            self.assertIn("Bought", self.bot1.others["last_logs"][-1])
            self.bot1.sell()
            self.s2.others["scg"]["current_good"] = False
            self.s2.save()
            self.bot1.refresh_from_db()
            self.bot1.decide()
            self.assertIn(
                "no other symbol to go", self.bot1.others["last_logs"][-1]
            )
            self.bot1.strategy_params = "use_local_memory=0,early_onset=1"
            self.bot1.symbol.others["scg"]["early_onset"] = True
            self.bot1.symbol.others["scg"]["line_l_var"] = [0.1, 0, -0.11]
            self.s1.others["scg"]["current_good"] = False
            self.s1.others["scg"]["early_onset"] = True
            self.s1.others["scg"]["line_l_var"] = [1, 1, 2]
            self.s1.others["scg"]["line_s_var"] = [1, 1, 2]
            self.s1.save()
            self.bot1.jumpy_whitelist = None
            self.bot1.jumpy_blacklist = "S1BUSD, S2BUSD"
            self.bot1.decide()
            self.assertIn(
                "no other symbol to go", self.bot1.others["last_logs"][-1]
            )
            self.bot1.jumpy_whitelist = "S2BUSD"
            self.bot1.jumpy_blacklist = None
            self.bot1.decide()
            self.assertIn(
                "no other symbol to go", self.bot1.others["last_logs"][-1]
            )
            self.bot1.jumpy_whitelist = None
            self.bot1.jumpy_blacklist = None
            self.bot1.decide()
            self.assertIn("Jumped", self.bot1.others["last_logs"][-3])
            self.assertIn("Bought", self.bot1.others["last_logs"][-1])
            #
            self.bot1.strategy_params = "use_local_memory=1"
            self.bot1.reset_hard()
            self.bot1.on()
            self.bot1.symbol = self.s1
            #
            self.s1.others["scg"]["current_good"] = True
            self.s1.others["scg"]["line_s_var"] = [1, 1, 1]
            self.s1.others["scg"]["line_diff_sm"] = [1, 1, 1]
            self.assertEqual(self.bot1.has_local_memory(), False)
            self.bot1.decide()
            self.assertIn("Bought", self.bot1.others["last_logs"][-1])
            self.assertEqual(self.bot1.has_local_memory(), False)
            self.bot1.price_current = 1
            self.bot1.decide()
            self.assertEqual(self.bot1.has_local_memory(), False)
            self.assertIn("below min", self.bot1.others["last_logs"][-1])
            self.bot1.price_current = 1.001
            self.bot1.decide()
            self.assertEqual(self.bot1.has_local_memory(), True)
            self.assertIn("below min", self.bot1.others["last_logs"][-1])
            self.bot1.price_current = 1.3
            self.bot1.decide()
            self.assertIn("Kept goin", self.bot1.others["last_logs"][-1])
            self.bot1.price_current = 1.299
            self.bot1.decide()
            self.assertIn("Kept goin", self.bot1.others["last_logs"][-1])
            self.bot1.price_current = 1.2
            self.bot1.decide()
            self.assertIn("Sold", self.bot1.others["last_logs"][-1])
            self.bot1.price_current = 1.5
            self.bot1.decide()
            self.assertIn(
                "not in good status and ascending",
                self.bot1.others["last_logs"][-1],
            )
            self.bot1.price_current = 1.51
            self.bot1.decide()
            self.assertIn("Bought", self.bot1.others["last_logs"][-1])
            self.s1.others["scg"]["line_diff_sm"] = [1, 1, -0.1]
            self.bot1.price_current = 1.6
            self.bot1.decide()
            self.assertIn("Sold", self.bot1.others["last_logs"][-1])
            self.s2.others["scg"]["current_good"] = True
            self.s2.save()
            self.bot1.decide()
            self.assertIn("Jumped", self.bot1.others["last_logs"][-3])
            self.assertIn("Bought", self.bot1.others["last_logs"][-1])
            self.assertEqual(self.bot1.has_local_memory(), False)
            #
            self.bot1.strategy_params = (
                "use_local_memory=1,use_matrix_time_res=1"
            )
            #
            self.bot1.price_current = 1.3
            self.bot1.decide()
            self.assertIn("Kept goin", self.bot1.others["last_logs"][-1])
            self.bot1.symbol.last_updated = (
                timezone.now() - timezone.timedelta(minutes=2)
            )
            self.bot1.price_current = 1.1
            self.bot1.decide()
            self.assertIn(
                "Using matrix's time resolution",
                self.bot1.others["last_logs"][-1],
            )
            self.bot1.sell()
            self.bot1.decide()
            self.assertIn(
                "Using matrix's time resolution",
                self.bot1.others["last_logs"][-1],
            )
            self.bot1.symbol.last_updated = timezone.now()
            self.bot1.symbol.variation_range_24h = 1
            self.bot1.decide()
            self.assertIn(
                "VR24h",
                self.bot1.others["last_logs"][-1],
            )

    def test_turtle(self):
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/ticker/price",
                json={"symbol": "S1BUSD", "price": "1.0"},
            )
            self.bot1.symbol = self.s1
            self.bot1.strategy = "turtle"
            #
            self.s1.others["scg"]["line_m_var"] = [1, 1, 1]
            self.s1.others["atr"]["current"] = 0.1
            self.s1.others["dc"]["upper_break"] = True
            self.bot1.on()
            self.bot1.decide()
            self.assertIn("Bought", self.bot1.others["last_logs"][-1])
            self.bot1.price_current = 0.9
            self.bot1.decide()
            self.assertIn("Still on board", self.bot1.others["last_logs"][-1])
            self.bot1.price_current = 0.7
            self.bot1.decide()
            self.assertIn("Sold", self.bot1.others["last_logs"][-1])
            self.s1.others["dc"]["upper_break"] = False
            self.bot1.decide()
            self.assertIn(
                "not in good status", self.bot1.others["last_logs"][-1]
            )
            self.bot1.symbol.last_updated = timezone.now()
            self.bot1.symbol.variation_range_24h = 1
            self.bot1.decide()
            self.assertIn(
                "VR24h",
                self.bot1.others["last_logs"][-1],
            )
            self.bot1.strategy_params = "use_matrix_time_res=1"
            #
            self.s1.others["dc"]["upper_break"] = True
            self.bot1.symbol.last_updated = (
                timezone.now() - timezone.timedelta(minutes=2)
            )
            self.bot1.price_current = 1.1
            self.bot1.decide()
            self.assertIn(
                "Using matrix's time resolution",
                self.bot1.others["last_logs"][-1],
            )
            #
            self.bot1.buy()
            self.bot1.decide()
            self.assertIn(
                "Using matrix's time resolution",
                self.bot1.others["last_logs"][-1],
            )
            self.bot1.strategy_params = None
            self.s1.others["dc"]["lower_break"] = True
            self.bot1.decide()
            self.assertIn("Sold", self.bot1.others["last_logs"][-1])
            #
            self.bot1.symbol.variation_range_24h = 4
            self.s1.others["dc"]["upper_break"] = False
            self.s2.others["dc"]["upper_break"] = True
            self.s2.others["scg"]["line_m_var"] = [1, 1, 1]
            self.s2.others["atr"]["current"] = 0.1
            self.s2.save()
            self.bot1.decide()
            self.assertIn("Jumped", self.bot1.others["last_logs"][-3])
            self.assertEqual(self.bot1.has_local_memory(), False)


@pytest.mark.usefixtures("celery_session_app")
@pytest.mark.usefixtures("celery_session_worker")
class TestTasks(BotTestCase):
    def test_update_all_bots(self):
        with requests_mock.Mocker() as m:
            m.get(
                f"{BINANCE_API_URL}/api/v3/ticker/price",
                json=[
                    {"symbol": "S1BUSD", "price": "1.0"},
                    {"symbol": "S2BUSD", "price": "1.0"},
                    {"symbol": "S3BUSD", "price": "1.0"},
                ],
            )
            task = tasks.update_all_bots_job.delay()
            result = task.get()
            groups = TraderoBotGroup.objects.all().count()
            self.assertIn(str(groups), result)

    def test_update_all_symbols(self):
        with requests_mock.Mocker() as m:
            with open("base/fixtures/klines_response_mock.json") as f:
                content = f.read()
                m.get(
                    f"{BINANCE_API_URL}/api/v3/klines?symbol=S1BUSD"
                    f"&interval={settings.TIME_INTERVAL}m",
                    text=content,
                )
                m.get(
                    f"{BINANCE_API_URL}/api/v3/klines?symbol=S2BUSD"
                    f"&interval={settings.TIME_INTERVAL}m",
                    text=content,
                )
                m.get(
                    f"{BINANCE_API_URL}/api/v3/ticker/price",
                    json=[
                        {"symbol": "S1BUSD", "price": "1.0"},
                        {"symbol": "S2BUSD", "price": "1.0"},
                    ],
                )
            # Symbol.update_all_indicators()
            task = tasks.update_all_indicators_job.delay()
            result = task.get()
            self.assertIn("3", result)

    @override_settings(
        USE_TASKS=True,
        PREDICTION_ENABLED=False,
        INDICATORS=["scg"],
    )
    def test_task_scheduling(self):
        with requests_mock.Mocker() as m:
            with open("base/fixtures/klines_response_mock.json") as f:
                content = f.read()
                m.get(
                    f"{BINANCE_API_URL}/api/v3/klines"
                    f"&interval={settings.TIME_INTERVAL}m",
                    text=content,
                )
                m.get(
                    f"{BINANCE_API_URL}/api/v3/ticker/price",
                    json=[
                        {"symbol": "S1BUSD", "price": "1.0"},
                        {"symbol": "S2BUSD", "price": "1.0"},
                        {"symbol": "S3BUSD", "price": "1.0"},
                    ],
                )
            Symbol.update_all_indicators()
            TraderoBot.update_all_bots()
            self.assertLogs("Retrieved and Updated")
            self.assertLogs("bots updated")


class TestIndicators(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1, _ = Symbol.objects.update_or_create(
            symbol="S1BUSD",
            status="TRADING",
            base_asset="S1",
            quote_asset="BUSD",
            defaults={
                "model_score": 0.99,
                "volume_quote_asset": Decimal(1000000),
                "variation_range_24h": Decimal(9),
                "info": {
                    "filters": [
                        {},
                        {"stepSize": "0.10000000", "filterType": "LOT_SIZE"},
                    ]
                },
            },
        )
        super().setUpClass()

    @override_settings(
        INDICATORS=["atr"],
    )
    def test_atr(self):
        atr_data_mock = [
            # 0
            {
                "price_close": 1.3111,
                "price_high": None,
                "price_low": None,
            },
            # 1
            {
                "price_close": 1.3075,
                "price_high": 1.3140,
                "price_low": 1.3053,
            },
            # 2
            {
                "price_close": 1.3078,
                "price_high": 1.3131,
                "price_low": 1.3067,
            },
            # 3
            {
                "price_close": 1.3151,
                "price_high": 1.3194,
                "price_low": 1.3071,
            },
            # 4
            {
                "price_close": 1.3041,
                "price_high": 1.3176,
                "price_low": 1.3009,
            },
            # 5
            {
                "price_close": 1.2935,
                "price_high": 1.3050,
                "price_low": 1.2935,
            },
            # 6
            {
                "price_close": 1.2974,
                "price_high": 1.2999,
                "price_low": 1.2941,
            },
            # 7
            {
                "price_close": 1.2919,
                "price_high": 1.3029,
                "price_low": 1.2912,
            },
            # 8
            {
                "price_close": 1.2884,
                "price_high": 1.2942,
                "price_low": 1.2842,
            },
            # 9
            {
                "price_close": 1.2881,
                "price_high": 1.2929,
                "price_low": 1.2846,
            },
            # 10
            {
                "price_close": 1.2836,
                "price_high": 1.2889,
                "price_low": 1.2796,
            },
            # 11
            {
                "price_close": 1.2881,
                "price_high": 1.2900,
                "price_low": 1.2819,
            },
            # 12
            {
                "price_close": 1.2905,
                "price_high": 1.2933,
                "price_low": 1.2840,
            },
            # 13
            {
                "price_close": 1.2857,
                "price_high": 1.2997,
                "price_low": 1.2833,
            },
            # 14
            {
                "price_close": 1.2932,
                "price_high": 1.2956,
                "price_low": 1.2821,
            },
            # 15  -   1.2993  1.2904
        ]
        with mock.patch.object(ATR, "get_data", return_value=atr_data_mock):
            atr_indicator = ATR(symbol=self.s1, periods=14)
            result = atr_indicator.calculate()
            assert result["current"] == 0.010614285714285703

    def test_dc(self):
        dc_data_mock = [
            # 0
            {
                "price_high": 1.2,
                "price_low": 1.2,
            },
            # 1
            {
                "price_high": 1.3140,
                "price_low": 1.3053,
            },
            # 2
            {
                "price_high": 1.3131,
                "price_low": 1.3067,
            },
            # 3
            {
                "price_high": 1.3194,
                "price_low": 1.3071,
            },
        ]
        with mock.patch.object(DC, "get_data", return_value=dc_data_mock):
            dc_indicator = DC(symbol=self.s1, periods=3, amount=2)
            result = dc_indicator.calculate()
            assert result == {
                "params": {"periods": 3, "amount": 2},
                "upper": [1.314, 1.3194],
                "lower": [1.2, 1.3053],
                "upper_break": True,
                "lower_break": False,
            }
