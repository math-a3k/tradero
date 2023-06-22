import signal
from contextlib import contextmanager
from io import StringIO

import django_rq
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

from .consumers import SymbolHTMLConsumer, SymbolJSONConsumer
from .handlers import message_handler
from .models import Kline, Symbol, TrainingData, User, WSClient

BINANCE_API_URL = "https://api.binance.com"
TEST_SETTINGS = {
    "SYNC_EXECUTION": True,
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
        self.assertEqual(TrainingData.from_klines(self.s1), 35)
        self.assertEqual(TrainingData.from_klines(self.s1), 0)


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
                "volume_quote_asset": 1000000,
            },
        )
        super().setUpClass()

    def test_home(self):
        url = reverse("base:home")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

    def test_instrucoes(self):
        url = reverse("base:instrucoes")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)


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
    def test_scheduler(self):
        scheduler = django_rq.get_scheduler()
        out = StringIO()
        with self.assertRaises(TimeoutError):
            with timeout(2):
                call_command("scheduler", stdout=out)
        jobs = [str(j) for j in scheduler.get_jobs()]
        self.assertIn("update_all_indicators_job", jobs[0])

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
