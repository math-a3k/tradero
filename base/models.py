import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import timezone as tz
from decimal import Decimal

import pandas as pd
from asgiref.sync import async_to_sync
from binance.spot import Spot
from channels.layers import get_channel_layer
from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.core.cache import cache
from django.db import models
from django.template.loader import render_to_string
from django.utils import timezone
from django.utils.module_loading import import_string
from django_ai.supervised_learning.models import HGBTreeRegressor, OneClassSVC
from encore.concurrent.futures.synchronous import SynchronousExecutor
from sklearn.model_selection import GridSearchCV

from .client import TraderoClient
from .indicators import get_indicators
from .utils import datetime_minutes_rounder

channel_layer = get_channel_layer()
logger = logging.getLogger(__name__)


def thread_pool_executor(
    threads=settings.EXECUTOR_THREADS,
):  # pragma: no cover
    if settings.SYNC_EXECUTION:
        # Meant to be used in testing
        return SynchronousExecutor()
    else:
        return ThreadPoolExecutor(max_workers=threads)


class User(AbstractUser):
    """
    Custom User Model for tradero
    """

    _client = None

    api_key = models.CharField(
        "Binance's API key", max_length=255, blank=True, null=True
    )
    api_secret = models.CharField(
        "Binance's API key", max_length=255, blank=True, null=True
    )
    trading_active = models.BooleanField("Is Trading Active?", default=True)
    others = models.JSONField("Others", default=dict)

    class Meta:
        verbose_name = "User"
        verbose_name_plural = "Users"

    def get_client(self, reinit=False):  # pragma: no cover
        if not self._client or reinit:
            self._client = TraderoClient(self)
        return self._client


class WSClient(models.Model):
    channel_group = models.CharField(max_length=256)
    channel_name = models.CharField(max_length=256)
    time_connect = models.DateTimeField("Time - Connection", auto_now_add=True)
    time_disconnect = models.DateTimeField(
        "Time - Disconnection",
        blank=True,
        null=True,
    )

    class Meta:
        verbose_name = "WS Client"
        verbose_name_plural = "WS Clients"

    def __str__(self):
        return self.channel_name

    @property
    def is_open(self):
        return not self.time_disconnect


class SymbolManager(models.Manager):
    def top_symbols(self, n=settings.SYMBOLS_QUANTITY):
        return self.filter(
            status="TRADING",
            is_enabled=True,
            model_score__gte=settings.MODEL_SCORE_THRESHOLD,
            volume_quote_asset__gte=settings.MARKET_SIZE_THRESHOLD,
        ).order_by("-model_score")[:n]

    def available(self):
        return self.filter(
            status="TRADING",
            is_enabled=True,
        )

    def all_top_symbols(self, n=settings.SYMBOLS_QUANTITY):
        return self.filter(
            status="TRADING",
            is_enabled=True,
            volume_quote_asset__gte=settings.MARKET_SIZE_THRESHOLD,
        ).order_by("-model_score")[:n]


class Symbol(models.Model):
    _indicators = None
    _prediction_model_class = None
    _outliers_model_class = None
    _last_td = None
    _serializer_class = None

    symbol = models.CharField("Symbol", max_length=20)
    status = models.CharField("Status", max_length=20)
    is_enabled = models.BooleanField("Is Enabled?", default=True)
    base_asset = models.CharField("Base Asset", max_length=20)
    quote_asset = models.CharField("Quote Asset", max_length=20)
    volume_quote_asset = models.DecimalField(
        "Quote Asset Volume (24h)",
        max_digits=20,
        decimal_places=8,
        blank=True,
        null=True,
    )
    last_value = models.DecimalField(
        "Last Value",
        max_digits=16,
        decimal_places=8,
        blank=True,
        null=True,
    )
    last_value_time = models.DateTimeField(
        "Last Value (time)",
        blank=True,
        null=True,
    )
    last_variation = models.DecimalField(
        "Last Variation",
        max_digits=16,
        decimal_places=8,
        blank=True,
        null=True,
    )
    last_variation_24h = models.DecimalField(
        "Last Variation - 24 hs",
        max_digits=16,
        decimal_places=8,
        blank=True,
        null=True,
    )
    model_score = models.DecimalField(
        "Model Score",
        max_digits=16,
        decimal_places=8,
        blank=True,
        null=True,
    )
    prediction_time_interval = models.CharField(
        "Prediction (time interval)",
        max_length=2,
        blank=True,
        null=True,
    )
    prediction_time = models.DateTimeField(
        "Prediction (time)",
        blank=True,
        null=True,
    )
    prediction_value = models.DecimalField(
        "Prediction (value)",
        max_digits=16,
        decimal_places=8,
        blank=True,
        null=True,
    )
    prediction_variation = models.DecimalField(
        "Prediction (variation)",
        max_digits=16,
        decimal_places=8,
        blank=True,
        null=True,
    )
    others = models.JSONField("Others", default=dict)

    objects = SymbolManager()

    class Meta:
        verbose_name = "Symbol"
        verbose_name_plural = "Symbols"
        ordering = ["symbol"]

    def __str__(self):
        return self.symbol

    @classmethod
    def load_all_data(
        cls, start_time=None, end_time=None, threads=settings.EXECUTOR_THREADS
    ):
        symbols = cls.objects.available()
        with thread_pool_executor(threads) as executor:
            for symbol in symbols:
                executor.submit(
                    symbol.load_data(start_time=start_time, end_time=end_time)
                )
        logger.warning("-> Data loading complete <-")

    @classmethod
    def load_all_data_sync(cls):  # pragma: no cover
        """
        For debugging purposes
        """
        symbols = cls.objects.all()
        for s in symbols:
            s.load_data()

    def load_data(self, start_time=None, end_time=None):
        n_klines = Kline.load_klines(
            self, start_time=start_time, end_time=end_time
        )
        n_td = TrainingData.from_klines(self)
        logger.warning(f"{self}: Data loaded")
        return (n_klines, n_td)

    def get_prediction_model(self):
        if not self._prediction_model_class:
            self._prediction_model_class = import_string(
                settings.PREDICTION_MODEL_CLASS
            )
        r, c = self._prediction_model_class.objects.get_or_create(
            symbol=self,
            name=self.symbol,
            data_model="base.TrainingData",
        )
        return r

    def get_indicators(self):
        if not self._indicators:
            self._indicators = get_indicators()
        return self._indicators

    def get_serializer_class(self):
        if not self._serializer_class:  # pragma: no cover
            self._serializer_class = import_string(settings.SERIALIZER_CLASS)
        return self._serializer_class

    def get_outlier_classifiers(self):
        if not self._outliers_model_class:
            self._outliers_model_class = import_string(
                settings.OUTLIERS_MODEL_CLASS
            )
        o_m = self._outliers_model_class
        o1, _ = o_m.objects.get_or_create(
            symbol=self,
            name=f"{self.symbol}_o1",
            data_model="base.TrainingData",
            window=1,
            defaults={
                o_m.OUTLIERS_PROPORTION_PARAMETER: settings.OUTLIERS_THRESHOLD,
            },
        )
        o2, _ = o_m.objects.get_or_create(
            symbol=self,
            name=f"{self.symbol}_o2",
            data_model="base.TrainingData",
            window=2,
            defaults={
                o_m.OUTLIERS_PROPORTION_PARAMETER: settings.OUTLIERS_THRESHOLD,
            },
        )
        o3, _ = o_m.objects.get_or_create(
            symbol=self,
            name=f"{self.symbol}_o3",
            data_model="base.TrainingData",
            window=3,
            defaults={
                o_m.OUTLIERS_PROPORTION_PARAMETER: settings.OUTLIERS_THRESHOLD,
            },
        )
        return [o1, o2, o3]

    def update_and_classify_outliers(self):
        os = self.get_outlier_classifiers()
        os_res = {}
        for o in os:
            o.perform_inference()
            os_res[f"o{o.window}"] = o.classify_current()
        return os_res

    def update_prediction_model(self):
        predictor = self.get_prediction_model()
        predictor.perform_inference()
        self.model_score = predictor.metadata["inference"]["current"][
            "scores"
        ]["r2"]
        return predictor

    def update_indicators(self, push=True):
        self._last_td = self.training_data.last()
        klines_24h = self.klines.filter(
            time_close__gte=timezone.now() - timezone.timedelta(days=1)
        )
        self.volume_quote_asset = klines_24h.aggregate(
            vol=models.Sum("volume_quote_asset")
        )["vol"]
        self.last_variation = self._last_td.variation
        if klines_24h:  # pragma: no cover
            self.last_variation_24h = (
                self._last_td.price_close / klines_24h[0].price_close * 100
            ) - 100
        self.last_value = self._last_td.price_close
        self.last_value_time = self._last_td.time + timezone.timedelta(
            seconds=1
        )
        self.others["last_3"] = [
            float(d)
            for d in [
                getattr(self._last_td, f"variation_{i:02d}")
                for i in range(1, 4)
            ]
        ]
        if not self.model_score:
            # There should always be a model score for the symbol, even if
            # prediction is not enabled
            predictor = self.update_prediction_model()
        if settings.PREDICTION_ENABLED:
            predictor = self.update_prediction_model()
            self.prediction_variation = Decimal(predictor.predict_next())
            self.prediction_value = self.last_value * (
                Decimal("1") + self.prediction_variation / 100
            )
            self.prediction_time_interval = settings.TIME_INTERVAL
            # TODO: Better parametrize the prediction time
            self.prediction_time = self._last_td.time + timezone.timedelta(
                minutes=settings.TIME_INTERVAL, seconds=1
            )
        if settings.OUTLIERS_ENABLED:
            self.others["outliers"] = self.update_and_classify_outliers()
        for indicator in self.get_indicators():
            self.others[indicator] = self.get_indicators()[indicator](
                self
            ).calculate()

        self.render_html_snippet(set_cache=True)
        self.render_json_snippet(set_cache=True)

        self.save()
        if push:
            async_to_sync(self.push_to_ws)()
        # Use warning to make sure it goes
        logger.warning(f"{self}: Indicators Updated.")

    async def push_to_ws(self):
        await channel_layer.group_send(
            "symbols_html",
            {
                "type": "symbol.html.message",
                "message": {
                    "type": "symbol_update",
                    "symbol": self.symbol,
                    "text": self.render_html_snippet(),
                },
            },
        )
        await channel_layer.group_send(
            "symbols_json",
            {
                "type": "symbol.json.message",
                "message": self.render_json_snippet(),
            },
        )

    def render_html_snippet(self, set_cache=False):
        cache_key = f"{self.symbol}_html"
        html = cache.get(cache_key)
        if not html or set_cache:
            html = render_to_string(
                "base/symbol_snippet.html",
                {
                    "symbol": self,
                    "settings": {
                        "PREDICTION_ENABLED": settings.PREDICTION_ENABLED,
                        "OUTLIERS_ENABLED": settings.OUTLIERS_ENABLED,
                    },
                },
            )
            cache.set(cache_key, html, settings.TIME_INTERVAL * 60 + 9)
        return html

    def render_json_snippet(self, set_cache=False):
        cache_key = f"{self.symbol}_json"
        j = cache.get(cache_key)
        if not j or set_cache:
            serializer = self.get_serializer_class()
            j = json.dumps(
                serializer().to_representation(self, set_cache=True)
            )
            cache.set(cache_key, j, settings.TIME_INTERVAL * 60 + 9)
        return j

    def retrieve_and_update(self):
        self.load_data()
        self.update_indicators()

    @classmethod
    def update_all_indicators(
        cls,
        only_top=False,
        push=True,
        model_score_threshold=settings.MODEL_SCORE_THRESHOLD,
        threads=settings.EXECUTOR_THREADS,
    ):
        Kline.load_all_klines(model_score=model_score_threshold)
        if only_top:
            symbols = cls.objects.all_top_symbols()
        else:
            symbols = cls.objects.available()
        with thread_pool_executor(threads) as executor:
            for symbol in symbols:
                executor.submit(TrainingData.from_klines, symbol)
        with thread_pool_executor(threads) as executor:
            for symbol in symbols:
                executor.submit(symbol.update_indicators, push=push)
        logger.warning(
            f"-> UPDATE ALL PREDICTIONS DONE (MST: {model_score_threshold}) <-"
        )

    @classmethod
    def general_warm_up(
        cls,
        n_periods=10,
        symbols=None,
        threads=settings.EXECUTOR_THREADS,
    ):
        now = timezone.now()
        for i in range(n_periods, 0, -1):
            logger.info(f"Period: {i}")
            start_time = now - timezone.timedelta(
                minutes=i * settings.TIME_INTERVAL * 1000
            )
            start_time = start_time.replace(
                minute=(
                    (start_time.minute // settings.TIME_INTERVAL)
                    * settings.TIME_INTERVAL
                ),
                second=0,
                microsecond=0,
            )
            end_time = (
                start_time
                + timezone.timedelta(minutes=settings.TIME_INTERVAL * 1000)
                - timezone.timedelta(seconds=1)
            )
            Kline.load_all_klines(start_time=start_time, end_time=end_time)
        symbols = symbols or cls.objects.available()
        with thread_pool_executor(threads) as executor:
            for symbol in symbols:
                executor.submit(TrainingData.from_klines, symbol)
        logger.warning("-> WARMING UP COMPLETE <-")

    @classmethod
    def calibrate_all_windows(
        cls,
        symbols=None,
        threads=settings.EXECUTOR_THREADS,
    ):
        symbols = symbols or cls.objects.available()
        with thread_pool_executor(threads) as executor:
            for symbol in symbols:
                pm = symbol.get_prediction_model()
                executor.submit(pm.calibrate_window)
        logger.warning("-> ALL WINDOWS CALIBRATION COMPLETE <-")

    @classmethod
    def calibrate_all_models(
        cls,
        symbols=None,
        only_top=False,
        threads=settings.EXECUTOR_THREADS,
    ):
        symbols = symbols or cls.objects.available()
        with thread_pool_executor(threads) as executor:
            for symbol in symbols:
                pm = symbol.get_prediction_model()
                executor.submit(pm.calibrate_model)
        logger.warning("-> ALL MODEL CALIBRATION COMPLETE <-")

    @classmethod
    def reset_symbols(
        cls,
        symbols=None,
    ):
        symbols = symbols or cls.objects.available()
        for symbol in symbols:
            symbol.training_data.all().delete()
            symbol.klines.all().delete()
            logger.warning(f"{symbol}: RESET")


class Kline(models.Model):
    client = Spot()

    symbol = models.ForeignKey(
        Symbol,
        verbose_name="Symbol",
        related_name="klines",
        on_delete=models.PROTECT,
    )
    time_open = models.DateTimeField("Open Time")
    time_close = models.DateTimeField("Close Time")
    time_interval = models.CharField("Time Interval", max_length=2)
    price_open = models.DecimalField(
        "Price Open", max_digits=20, decimal_places=8
    )
    price_high = models.DecimalField(
        "Price High", max_digits=20, decimal_places=8
    )
    price_low = models.DecimalField(
        "Price Low", max_digits=20, decimal_places=8
    )
    price_close = models.DecimalField(
        "Price Close", max_digits=20, decimal_places=8
    )
    volume = models.DecimalField("Volume", max_digits=20, decimal_places=8)
    volume_quote_asset = models.DecimalField(
        "Quote Asset Volume", max_digits=20, decimal_places=8
    )
    volume_tb_base_asset = models.DecimalField(
        "Taker buy base asset volume", max_digits=20, decimal_places=8
    )
    volume_tb_quote_asset = models.DecimalField(
        "Taker buy quote asset volume Price", max_digits=20, decimal_places=8
    )
    number_of_trades = models.DecimalField(
        "Number of Trades", max_digits=20, decimal_places=8
    )
    variation = models.DecimalField(
        "Variation", max_digits=20, decimal_places=8
    )

    class Meta:
        verbose_name = "Kline"
        verbose_name_plural = "Klines"

    def __str__(self):
        f_str = (
            f"[{self.symbol}] {self.time_open.strftime('%Y-%m-%d')} | "
            f"{self.time_open.strftime('%H:%M:%S')} - "
            f"{self.time_close.strftime('%H:%M:%S')} | {self.price_open} - "
            f"{self.price_close} | {self.variation}%"
        )
        return f_str

    @classmethod
    def from_binance_kline(cls, symbol, time_interval, b_kline):
        """
        Binance's response:
          [
            1499040000000,      // Kline open time
            "0.01634790",       // Open price
            "0.80000000",       // High price
            "0.01575800",       // Low price
            "0.01577100",       // Close price
            "148976.11427815",  // Volume
            1499644799999,      // Kline Close time
            "2434.19055334",    // Quote asset volume
            308,                // Number of trades
            "1756.87402397",    // Taker buy base asset volume
            "28.46694368",      // Taker buy quote asset volume
            "0"                 // Unused field, ignore.
          ]
        """
        kline = cls(
            symbol=symbol,
            time_open=timezone.datetime.fromtimestamp(
                b_kline[0] // 1000, tz=tz.utc
            ),
            time_close=timezone.datetime.fromtimestamp(
                b_kline[6] // 1000, tz=tz.utc
            ),
            time_interval=time_interval,
            price_open=b_kline[1],
            price_high=b_kline[2],
            price_low=b_kline[3],
            price_close=b_kline[4],
            volume=b_kline[5],
            volume_quote_asset=b_kline[7],
            number_of_trades=b_kline[8],
            volume_tb_base_asset=b_kline[9],
            volume_tb_quote_asset=b_kline[10],
            variation=((Decimal(b_kline[4]) / Decimal(b_kline[1])) * 100)
            - 100,
        )
        return kline

    @classmethod
    def from_binance_kline_ws(
        cls, symbol, ws_kline, save=True
    ):  # pragma: no cover
        """
        Binance's response:
          {
              "e": "kline",     // Event type
              "E": 123456789,   // Event time
              "s": "BNBBTC",    // Symbol
              "k": {
                "t": 123400000, // Kline start time
                "T": 123460000, // Kline close time
                "s": "BNBBTC",  // Symbol
                "i": "1m",      // Interval
                "f": 100,       // First trade ID
                "L": 200,       // Last trade ID
                "o": "0.0010",  // Open price
                "c": "0.0020",  // Close price
                "h": "0.0025",  // High price
                "l": "0.0015",  // Low price
                "v": "1000",    // Base asset volume
                "n": 100,       // Number of trades
                "x": false,     // Is this kline closed?
                "q": "1.0000",  // Quote asset volume
                "V": "500",     // Taker buy base asset volume
                "Q": "0.500",   // Taker buy quote asset volume
                "B": "123456"   // Ignore
              }
            }
        """
        kline = cls(
            symbol=symbol,
            time_open=timezone.datetime.fromtimestamp(
                ws_kline["k"]["t"] // 1000, tz=tz.utc
            ),
            time_close=timezone.datetime.fromtimestamp(
                ws_kline["k"]["T"] // 1000, tz=tz.utc
            ),
            time_interval=ws_kline["k"]["i"],
            price_open=ws_kline["k"]["o"],
            price_high=ws_kline["k"]["h"],
            price_low=ws_kline["k"]["l"],
            price_close=ws_kline["k"]["c"],
            volume=ws_kline["k"]["v"],
            volume_quote_asset=ws_kline["k"]["q"],
            number_of_trades=ws_kline["k"]["n"],
            volume_tb_base_asset=ws_kline["k"]["V"],
            volume_tb_quote_asset=ws_kline["k"]["Q"],
            variation=(
                (Decimal(ws_kline["k"]["c"]) / Decimal(ws_kline["k"]["o"]))
                * 100
            )
            - 100,
        )
        if save:
            kline.save()
            logger.warning(f"{symbol}: Created Kline: {kline}.")
        return kline

    @classmethod
    def load_klines(
        cls,
        symbol,
        interval=settings.TIME_INTERVAL,
        start_time=None,
        end_time=None,
        limit=1000,
    ):
        last_kline = symbol.klines.order_by("time_open").last()
        if last_kline and not start_time:
            logger.warning(f"{symbol}: Last kline is {last_kline}")
            start_time = datetime_minutes_rounder(last_kline.time_close)
        if not start_time:
            start_time = timezone.now() - timezone.timedelta(
                minutes=settings.TIME_INTERVAL * 1000
            )
            start_time = start_time.replace(
                minute=(
                    (start_time.minute // settings.TIME_INTERVAL)
                    * settings.TIME_INTERVAL
                ),
                second=0,
                microsecond=0,
            )
        if not end_time:
            end_time = timezone.now()
            end_time = end_time.replace(
                minute=(
                    (end_time.minute // settings.TIME_INTERVAL)
                    * settings.TIME_INTERVAL
                ),
                second=0,
                microsecond=0,
            ) - timezone.timedelta(seconds=1)
        logger.warning(
            f"{symbol}: Requesting klines from {start_time} to {end_time}."
        )
        klines_binance = cls.client.klines(
            symbol.symbol,
            interval=f"{interval}m",
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=1000,
        )
        klines, times = [], set()
        for k_b in klines_binance:
            kline = cls.from_binance_kline(symbol, f"{interval}m", k_b)
            klines.append(kline)
            times.add(kline.time_open.replace(tzinfo=tz.utc))
        existing_data = [
            t.replace(tzinfo=tz.utc)
            for t in cls.objects.filter(
                symbol=symbol,
                time_open__in=times,
                time_interval=f"{interval}m",
            ).values_list("time_open", flat=True)
        ]
        klines = [k for k in klines if k.time_open not in existing_data]
        created = cls.objects.bulk_create(klines)
        logger.warning(
            f"{symbol}: Received {len(klines_binance)} klines, "
            f"created {len(created)}."
        )
        return created

    @classmethod
    def load_all_klines(
        cls,
        interval=settings.TIME_INTERVAL,
        start_time=None,
        end_time=None,
        model_score=settings.MODEL_SCORE_THRESHOLD,
        symbols=None,
        threads=settings.EXECUTOR_THREADS,
    ):
        symbols = symbols or Symbol.objects.available()

        with thread_pool_executor(threads) as executor:
            for symbol in symbols:
                executor.submit(
                    cls.load_klines,
                    symbol,
                    interval=settings.TIME_INTERVAL,
                    start_time=start_time,
                    end_time=end_time,
                )


class TrainingData(models.Model):
    WINDOW = 20
    LEARNING_TARGET = "variation"
    LEARNING_FIELDS = [
        "variation_01",
        "variation_02",
        "variation_03",
        "variation_04",
        "variation_05",
        "variation_06",
        "variation_07",
        "variation_08",
        "variation_09",
        "variation_10",
        "variation_11",
        "variation_12",
        "variation_13",
        "variation_14",
        "variation_15",
        "variation_16",
        "variation_17",
        "variation_18",
        "variation_19",
        "variation_20",
        # "variation_21",
        # "variation_22",
        # "variation_23",
        # "variation_24",
        # "variation_25",
        # "variation_26",
        # "variation_27",
        # "variation_28",
        # "variation_29",
        # "variation_30",
        # "variation_31",
        # "variation_32",
        # "variation_33",
        # "variation_34",
        # "variation_35",
        # "variation_36",
        # "variation_37",
        # "variation_38",
        # "variation_39",
        # "variation_40",
    ]

    symbol = models.ForeignKey(
        Symbol,
        verbose_name="Symbol",
        related_name="training_data",
        on_delete=models.PROTECT,
    )
    time = models.DateTimeField("Time (End of Interval)")
    time_interval = models.CharField("Time Interval", max_length=2)
    price_close = models.DecimalField(
        "Price - End of Interval",
        max_digits=16,
        decimal_places=8,
    )
    variation = models.DecimalField(
        "Variation (t)", max_digits=16, decimal_places=8
    )
    variation_01 = models.DecimalField(
        "Variation (t-1)", max_digits=20, decimal_places=8
    )
    variation_02 = models.DecimalField(
        "Variation (t-2)", max_digits=20, decimal_places=8
    )
    variation_03 = models.DecimalField(
        "Variation (t-3)", max_digits=20, decimal_places=8
    )
    variation_04 = models.DecimalField(
        "Variation (t-4)", max_digits=20, decimal_places=8
    )
    variation_05 = models.DecimalField(
        "Variation (t-5)", max_digits=20, decimal_places=8
    )
    variation_06 = models.DecimalField(
        "Variation (t-6)", max_digits=20, decimal_places=8
    )
    variation_07 = models.DecimalField(
        "Variation (t-7)", max_digits=20, decimal_places=8
    )
    variation_08 = models.DecimalField(
        "Variation (t-8)", max_digits=20, decimal_places=8
    )
    variation_09 = models.DecimalField(
        "Variation (t-9)", max_digits=20, decimal_places=8
    )
    variation_10 = models.DecimalField(
        "Variation (t-10)", max_digits=20, decimal_places=8
    )
    variation_11 = models.DecimalField(
        "Variation (t-11)", max_digits=20, decimal_places=8
    )
    variation_12 = models.DecimalField(
        "Variation (t-12)", max_digits=20, decimal_places=8
    )
    variation_13 = models.DecimalField(
        "Variation (t-13)", max_digits=20, decimal_places=8
    )
    variation_14 = models.DecimalField(
        "Variation (t-14)", max_digits=20, decimal_places=8
    )
    variation_15 = models.DecimalField(
        "Variation (t-15)", max_digits=20, decimal_places=8
    )
    variation_16 = models.DecimalField(
        "Variation (t-16)", max_digits=20, decimal_places=8
    )
    variation_17 = models.DecimalField(
        "Variation (t-17)", max_digits=20, decimal_places=8
    )
    variation_18 = models.DecimalField(
        "Variation (t-18)", max_digits=20, decimal_places=8
    )
    variation_19 = models.DecimalField(
        "Variation (t-19)", max_digits=20, decimal_places=8
    )
    variation_20 = models.DecimalField(
        "Variation (t-20)", max_digits=20, decimal_places=8
    )
    # variation_21 = models.DecimalField(
    #     "Variation (t-21)", max_digits=20, decimal_places=8
    # )
    # variation_22 = models.DecimalField(
    #     "Variation (t-22)", max_digits=20, decimal_places=8
    # )
    # variation_23 = models.DecimalField(
    #     "Variation (t-23)", max_digits=20, decimal_places=8
    # )
    # variation_24 = models.DecimalField(
    #     "Variation (t-24)", max_digits=20, decimal_places=8
    # )
    # variation_25 = models.DecimalField(
    #     "Variation (t-25)", max_digits=20, decimal_places=8
    # )
    # variation_26 = models.DecimalField(
    #     "Variation (t-26)", max_digits=20, decimal_places=8
    # )
    # variation_27 = models.DecimalField(
    #     "Variation (t-27)", max_digits=20, decimal_places=8
    # )
    # variation_28 = models.DecimalField(
    #     "Variation (t-28)", max_digits=20, decimal_places=8
    # )
    # variation_29 = models.DecimalField(
    #     "Variation (t-29)", max_digits=20, decimal_places=8
    # )
    # variation_30 = models.DecimalField(
    #     "Variation (t-30)", max_digits=20, decimal_places=8
    # )
    # variation_31 = models.DecimalField(
    #     "Variation (t-31)", max_digits=20, decimal_places=8
    # )
    # variation_32 = models.DecimalField(
    #     "Variation (t-32)", max_digits=20, decimal_places=8
    # )
    # variation_33 = models.DecimalField(
    #     "Variation (t-33)", max_digits=20, decimal_places=8
    # )
    # variation_34 = models.DecimalField(
    #     "Variation (t-34)", max_digits=20, decimal_places=8
    # )
    # variation_35 = models.DecimalField(
    #     "Variation (t-35)", max_digits=20, decimal_places=8
    # )
    # variation_36 = models.DecimalField(
    #     "Variation (t-36)", max_digits=20, decimal_places=8
    # )
    # variation_37 = models.DecimalField(
    #     "Variation (t-37)", max_digits=20, decimal_places=8
    # )
    # variation_38 = models.DecimalField(
    #     "Variation (t-38)", max_digits=20, decimal_places=8
    # )
    # variation_39 = models.DecimalField(
    #     "Variation (t-39)", max_digits=20, decimal_places=8
    # )
    # variation_40 = models.DecimalField(
    #     "Variation (t-40)", max_digits=20, decimal_places=8
    # )

    class Meta:
        verbose_name = "Training Data"
        verbose_name_plural = "Training Data"
        ordering = [
            "time",
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["symbol", "time", "time_interval"],
                name="unique_symbol_time",
            )
        ]

    def __str__(self):
        return (
            f"[{self.symbol}] {self.time} ({self.time_interval}): "
            f"{self.variation}% [Window: {self.WINDOW}]"
        )

    @classmethod
    def from_klines(cls, symbol):
        """
        Only one Training Data will be generated for each kline
        """
        columns = ["time_close", "time_interval", "price_close", "variation"]
        last_td = symbol.training_data.order_by("time").last()
        if not last_td:
            ts = (
                symbol.klines.all()
                .order_by("-time_close")
                .values_list(*columns)
            )
        else:
            ts = (
                symbol.klines.filter(
                    time_open__gte=last_td.time
                    - timezone.timedelta(
                        minutes=settings.TIME_INTERVAL * (cls.WINDOW + 1),
                        seconds=1,
                    )
                )
                .order_by("-time_close")
                .values_list(*columns)
            )

        df = pd.DataFrame(ts, columns=columns)

        for i in range(1, cls.WINDOW + 1):
            df[f"variation_{i:02d}"] = df["variation"].shift(-i)

        # Drop rows where there is a NaN - TODO: slow, check
        df = df.dropna(axis=0)

        tds, times, time_intervals = [], set(), set()
        for row in df.values.tolist():
            tds.append(
                cls(
                    symbol=symbol,
                    time=row[0],
                    time_interval=row[1],
                    price_close=row[2],
                    variation=row[3],
                    variation_01=row[4],
                    variation_02=row[5],
                    variation_03=row[6],
                    variation_04=row[7],
                    variation_05=row[8],
                    variation_06=row[9],
                    variation_07=row[10],
                    variation_08=row[11],
                    variation_09=row[12],
                    variation_10=row[13],
                    variation_11=row[14],
                    variation_12=row[15],
                    variation_13=row[16],
                    variation_14=row[17],
                    variation_15=row[18],
                    variation_16=row[19],
                    variation_17=row[20],
                    variation_18=row[21],
                    variation_19=row[22],
                    variation_20=row[23],
                    # variation_21=row[24],
                    # variation_22=row[25],
                    # variation_23=row[26],
                    # variation_24=row[27],
                    # variation_25=row[28],
                    # variation_26=row[29],
                    # variation_27=row[30],
                    # variation_28=row[31],
                    # variation_29=row[32],
                    # variation_30=row[33],
                    # variation_31=row[34],
                    # variation_32=row[35],
                    # variation_33=row[36],
                    # variation_34=row[37],
                    # variation_35=row[38],
                    # variation_36=row[39],
                    # variation_37=row[40],
                    # variation_38=row[41],
                    # variation_39=row[42],
                    # variation_40=row[43],
                )
            )
            times.add(row[0])
            time_intervals.add(row[1])
        existing_data = cls.objects.filter(
            symbol=symbol, time__in=times, time_interval__in=time_intervals
        ).values_list("time", flat=True)
        tds_to_create = [td for td in tds if td.time not in existing_data]
        created = len(cls.objects.bulk_create(tds_to_create))
        if created > 0:
            logger.warning(f"{symbol}: Created {created} Training Data.")
        else:
            logger.warning(
                f"{symbol}: TD already exist: {tds} {tds_to_create}"
            )
        return created


class TraderoMixin:
    # Hack in the meantime, review LearningTechnique prediction for regression
    def h_predict(self, obs):
        eo = self.get_engine_object()
        if isinstance(obs, dict):
            obs = [[obs[key] for key in obs]]
        else:
            obs = [[o[key] for key in o] for o in obs]
        return eo.predict(obs)

    def td_to_dict(self, td):
        obs = {"variation": td.variation}
        for i in range(1, self.window):
            obs[f"variation_{i:02d}"] = getattr(td, f"variation_{i:02d}")
        return obs

    def _get_data_queryset(self):
        data_model = self._get_data_model()
        qs = data_model.objects.filter(symbol=self.symbol)
        return qs


class PredictionModel(TraderoMixin, models.Model):
    CALIBRATION_PARAMS = {}

    symbol = models.ForeignKey(
        "Symbol",
        verbose_name="Symbol",
        related_name="%(app_label)s_%(class)s_related",
        related_query_name="%(app_label)s_%(class)ss",
        on_delete=models.PROTECT,
    )

    def get_targets(self):
        targets = super().get_targets()
        return [float(t) for t in targets]

    def predict_next(self):
        last_td = self.symbol.training_data.last()
        obs = {"variation_01": last_td.variation}
        for i in range(2, self.get_window_size() + 1):
            obs[f"variation_{i:02d}"] = getattr(
                last_td, f"variation_{(i - 1):02d}"
            )

        pred = self.h_predict(obs)
        return pred[0]

    def predict_next_times(self, n):
        """
        Assumption: n < self.WINDOW
        """
        last_td = self.symbol.training_data.order_by("time").last()
        preds = []
        obss = []
        preds.append(self.predict_next())
        for i in range(2, n + 1):
            obs = {}
            for w in range(1, self.get_window_size() + 1):
                try:
                    obs[f"variation_{w:02d}"] = preds[w - 1]
                except Exception:
                    obs[f"variation_{w:02d}"] = getattr(
                        last_td, f"variation_{w - i + 1:02d}"
                    )
            obss.append(obs)
            preds.append(self.h_predict(obs)[0])
        return (preds, obss)

    def calibrate_window(self, save=True):
        scores = []
        for i in range(1, TrainingData.WINDOW + 1):
            self.learning_fields = ", ".join(
                [f"variation_{j:02d}" for j in range(1, i + 1)]
            )
            self.perform_inference(save=False)
            score = self.metadata["inference"]["current"]["scores"]["r2"]
            scores.append(score)
            logger.warning(f"[{self.symbol}] Window size: {i}: {score}")
        win_size = scores.index(max(scores)) + 1
        logger.warning(
            f"[{self.symbol}] Best available window size: {win_size} - {scores}"
        )
        self.learning_fields = ", ".join(
            [f"variation_{j:02d}" for j in range(1, win_size + 1)]
        )
        if save:  # pragma: no cover
            self.save()

    def calibrate_model(self, save=True):
        gs_eo = GridSearchCV(self.get_engine_object(), self.CALIBRATION_PARAMS)
        data, targets = self.get_data(), self.get_targets()
        gs_eo.fit(data, targets)
        for param, value in gs_eo.best_params_.items():
            setattr(self, param, value)
        logger.warning(f"{self.symbol}: Best params: {gs_eo.best_params_}")
        if save:  # pragma: no cover
            self.save()
        return gs_eo.best_params_

    def get_window_size(self):
        return len(self._get_data_learning_fields())

    class Meta:
        abstract = True


class DecisionTreeRegressor(PredictionModel, HGBTreeRegressor):
    """
    https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
    """

    CALIBRATION_PARAMS = {
        "loss": ["squared_error", "absolute_error"],
        "learning_rate": [0.1, 0.2, 0.3],
        # "max_iter": [100, 200, ],
        # "max_leaf_nodes": [31, None, ],
        "l2_regularization": [
            0,
            0.1,
            0.2,
            0.4,
        ],
    }

    class Meta:
        verbose_name = "Decision Trees Regressor"
        verbose_name_plural = "Decision Trees Regressors"


class OutlierDetectionModel(TraderoMixin, models.Model):
    OUTLIERS_PROPORTION_PARAMETER = None

    symbol = models.ForeignKey(
        "Symbol",
        verbose_name="Symbol",
        related_name="%(app_label)s_%(class)s_related",
        related_query_name="%(app_label)s_%(class)ss",
        on_delete=models.PROTECT,
    )
    window = models.IntegerField(
        "Window",
        default=1,
        help_text=("Window to be used to consider outlier"),
    )

    def save(self, *args, **kwargs):
        self.learning_fields = ", ".join(
            ["variation"]
            + [f"variation_{j:02d}" for j in range(1, self.window)]
        )
        super().save(*args, **kwargs)

    def classify_current(self):
        last_td = self.symbol.training_data.last()
        pred = self.h_predict(self.td_to_dict(last_td))
        return bool(pred[0] < 0)

    class Meta:
        abstract = True


class OutliersSVC(OutlierDetectionModel, OneClassSVC):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM
    """

    OUTLIERS_PROPORTION_PARAMETER = "nu"

    class Meta:
        verbose_name = "Outliers SVC"
        verbose_name_plural = "Outliers SVCs"
