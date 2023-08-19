import json
import logging
import sys
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
from django.core.exceptions import ValidationError
from django.core.validators import MinValueValidator
from django.db import models
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils import timezone
from django.utils.module_loading import import_string
from django_ai.supervised_learning.models import HGBTreeRegressor, OneClassSVC
from encore.concurrent.futures.synchronous import SynchronousExecutor
from requests.adapters import HTTPAdapter
from sklearn.model_selection import GridSearchCV

from .client import TraderoClient
from .indicators import get_indicators
from .strategies import get_strategies
from .utils import datetime_minutes_rounder, get_commission

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
        "Binance's API key secret", max_length=255, blank=True, null=True
    )
    trading_active = models.BooleanField("Is Trading Active?", default=True)
    checkpoint = models.DateTimeField(
        "Checkpoint",
        blank=True,
        null=True,
    )
    others = models.JSONField("Others", default=dict)

    class Meta:
        verbose_name = "User"
        verbose_name_plural = "Users"

    def get_client(self, reinit=False):  # pragma: no cover
        if not self._client or reinit:
            self._client = TraderoClient(self)
        return self._client

    @property
    def trade_summary(self):
        return TradeHistory.summary_for_object(self)

    def save(self, *args, **kwargs):
        created = True if not self.pk else False
        super().save(*args, **kwargs)
        if created:
            TraderoBotGroup.objects.create(user=self)


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
    def available(self):
        return self.filter(
            status="TRADING",
            is_enabled=True,
        )

    def top_symbols(self, n=settings.SYMBOLS_QUANTITY):
        qs1 = self.filter(
            status="TRADING",
            is_enabled=True,
            model_score__gte=settings.MODEL_SCORE_THRESHOLD,
            volume_quote_asset__gte=settings.MARKET_SIZE_THRESHOLD,
        ).prefetch_related(
            models.Prefetch(
                "bots",
                queryset=TraderoBot.objects.enabled(),
                to_attr="bots_prefetched",
            ),
        )
        qs2 = self.filter(
            bots__status__gt=TraderoBot.Status.INACTIVE,
            bots__user__trading_active=True,
        ).prefetch_related(
            models.Prefetch(
                "bots",
                queryset=TraderoBot.objects.enabled(),
                to_attr="bots_prefetched",
            ),
        )
        return qs1.order_by("-model_score")[:n].union(qs2)

    def all_top_symbols(self, n=settings.SYMBOLS_QUANTITY):
        qs1 = self.filter(
            status="TRADING",
            is_enabled=True,
            volume_quote_asset__gte=settings.MARKET_SIZE_THRESHOLD,
        ).prefetch_related(
            models.Prefetch(
                "bots",
                queryset=TraderoBot.objects.enabled(),
                to_attr="bots_prefetched",
            ),
        )
        qs2 = self.filter(
            bots__status__gt=TraderoBot.Status.INACTIVE,
            bots__user__trading_active=True,
        ).prefetch_related(
            models.Prefetch(
                "bots",
                queryset=TraderoBot.objects.enabled(),
                to_attr="bots_prefetched",
            ),
        )
        return qs1.order_by("-model_score")[:n].union(qs2)


class Symbol(models.Model):
    _indicators = None
    _prediction_model_class = None
    _outliers_model_class = None
    _last_td = None
    _serializer_class = None

    client = Spot()

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
    info = models.JSONField("Symmbol's Binance's Info", default=dict)
    last_updated = models.DateTimeField(
        "Last Updated (timestamp)",
        blank=True,
        null=True,
        auto_now=True,
    )

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
        Kline.load_klines(self, start_time=start_time, end_time=end_time)
        td = TrainingData.from_klines(self)
        logger.warning(f"{self}: Data loaded")
        return td

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

    def update_indicators(
        self, push=True, last_td=None, bot_early_notification=False
    ):
        self._last_td = (
            last_td or self.training_data.first()  # negative ordering)
        )
        klines_24h = self.klines.order_by("time_close").filter(
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

        # Early bot notification if available
        if bot_early_notification:  # pragma: no cover
            if getattr(self, "bots_prefetched", []):
                price = self.client.ticker_price(symbol=self.symbol)["price"]
                # Re-fetch to avoid errors
                for bot in self.bots.enabled():
                    bot.price_current = Decimal(price)
                    bot.symbol = self
                    bot.decide()

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

    def retrieve_and_update(self, push=False):
        tds = self.load_data()
        last_td = tds[0] if tds else None
        self.update_indicators(push=push, last_td=last_td)

    @classmethod
    def clean_data(cls):
        # TODO: return n records
        date_threshold = timezone.now() - timezone.timedelta(
            minutes=settings.TIME_INTERVAL * settings.CLEANING_WINDOW
        )
        Kline.objects.filter(time_close__lt=date_threshold).delete()
        TrainingData.objects.filter(time__lt=date_threshold).delete()
        logger.warning("-->: Data cleaned for all Symbols")

    @classmethod
    def update_all_indicators(
        cls,
        only_top=False,
        push=True,
        model_score_threshold=settings.MODEL_SCORE_THRESHOLD,
        threads=settings.EXECUTOR_THREADS,
    ):
        cache_key = settings.SYMBOLS_UPDATE_ALL_INDICATORS_KEY
        if not cache.get(cache_key, False) or "pytest" in sys.modules:
            cache.set(cache_key, True, 2400)
            timestamp = timezone.now()
            if only_top:
                symbols = cls.objects.all_top_symbols()
            else:
                symbols = cls.objects.available()

            with thread_pool_executor(threads) as executor:
                for symbol in symbols:
                    executor.submit(symbol.retrieve_and_update, push=push)
            logger.warning(
                f"-> UPDATE ALL PREDICTIONS DONE (MST: {model_score_threshold}) <-"
            )
            cls.clean_data()
            message = (
                f"---> Elapsed Time: "
                f"{ (timezone.now() - timestamp).total_seconds() } "
                f"({len(symbols)} Symbols) <----"
            )
            cache.set(cache_key, False)
        else:  # pragma: no cover
            message = (
                "Other process updating all indicators is running, please "
                "wait."
            )
        logger.warning(message)
        return message

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
    _client = None

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
    variation_range = models.DecimalField(
        "Variation Range", max_digits=20, decimal_places=8
    )

    class Meta:
        verbose_name = "Kline"
        verbose_name_plural = "Klines"
        ordering = ["-time_close"]  # Descending ordering
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "symbol",
                    "time_close",
                    "time_interval",
                ],  # leverage index on time_close
                name="klines_unique_symbol_time",
            )
        ]

    def __str__(self):
        f_str = (
            f"[{self.symbol}] {self.time_open.strftime('%Y-%m-%d')} | "
            f"{self.time_open.strftime('%H:%M:%S')} - "
            f"{self.time_close.strftime('%H:%M:%S')} | {self.price_open} - "
            f"{self.price_close} | {self.variation}%"
        )
        return f_str

    @classmethod
    def get_client(cls):
        if not cls._client:
            cls._client = Spot()
            cls._client.session.mount("https://", HTTPAdapter(pool_maxsize=36))
        return cls._client

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
            variation_range=((Decimal(b_kline[2]) / Decimal(b_kline[3])) * 100)
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
            variation_range=(
                (Decimal(ws_kline["k"]["h"]) / Decimal(ws_kline["k"]["l"]))
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
        last_kline = symbol.klines.first()
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
        client = cls.get_client()
        klines_binance = client.klines(
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
        return sorted(created, key=lambda x: x.time_close, reverse=True)

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
        "variation_range_01",
        "variation_range_02",
        "variation_range_03",
        "variation_range_04",
        "variation_range_05",
        "variation_range_06",
        "variation_range_07",
        "variation_range_08",
        "variation_range_09",
        "variation_range_10",
        "variation_range_11",
        "variation_range_12",
        "variation_range_13",
        "variation_range_14",
        "variation_range_15",
        "variation_range_16",
        "variation_range_17",
        "variation_range_18",
        "variation_range_19",
        "variation_range_20",
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
    variation_range_01 = models.DecimalField(
        "Variation Range (t-1)", max_digits=20, decimal_places=8
    )
    variation_range_02 = models.DecimalField(
        "Variation Range (t-2)", max_digits=20, decimal_places=8
    )
    variation_range_03 = models.DecimalField(
        "Variation Range (t-3)", max_digits=20, decimal_places=8
    )
    variation_range_04 = models.DecimalField(
        "Variation Range (t-4)", max_digits=20, decimal_places=8
    )
    variation_range_05 = models.DecimalField(
        "Variation Range (t-5)", max_digits=20, decimal_places=8
    )
    variation_range_06 = models.DecimalField(
        "Variation Range (t-6)", max_digits=20, decimal_places=8
    )
    variation_range_07 = models.DecimalField(
        "Variation Range (t-7)", max_digits=20, decimal_places=8
    )
    variation_range_08 = models.DecimalField(
        "Variation Range (t-8)", max_digits=20, decimal_places=8
    )
    variation_range_09 = models.DecimalField(
        "Variation Range (t-9)", max_digits=20, decimal_places=8
    )
    variation_range_10 = models.DecimalField(
        "Variation Range (t-10)", max_digits=20, decimal_places=8
    )
    variation_range_11 = models.DecimalField(
        "Variation Range (t-11)", max_digits=20, decimal_places=8
    )
    variation_range_12 = models.DecimalField(
        "Variation Range (t-12)", max_digits=20, decimal_places=8
    )
    variation_range_13 = models.DecimalField(
        "Variation Range (t-13)", max_digits=20, decimal_places=8
    )
    variation_range_14 = models.DecimalField(
        "Variation Range (t-14)", max_digits=20, decimal_places=8
    )
    variation_range_15 = models.DecimalField(
        "Variation Range (t-15)", max_digits=20, decimal_places=8
    )
    variation_range_16 = models.DecimalField(
        "Variation Range (t-16)", max_digits=20, decimal_places=8
    )
    variation_range_17 = models.DecimalField(
        "Variation Range (t-17)", max_digits=20, decimal_places=8
    )
    variation_range_18 = models.DecimalField(
        "Variation Range (t-18)", max_digits=20, decimal_places=8
    )
    variation_range_19 = models.DecimalField(
        "Variation Range (t-19)", max_digits=20, decimal_places=8
    )
    variation_range_20 = models.DecimalField(
        "Variation Range (t-20)", max_digits=20, decimal_places=8
    )

    class Meta:
        verbose_name = "Training Data"
        verbose_name_plural = "Training Data"
        ordering = ["-time"]  # Descending ordering
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
        columns = [
            "time_close",
            "time_interval",
            "price_close",
            "variation",
            "variation_range",
        ]
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
        #
        for i in range(1, cls.WINDOW + 1):
            df[f"variation_range_{i:02d}"] = df["variation_range"].shift(-i)

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
                    variation_range_01=row[24],
                    variation_range_02=row[25],
                    variation_range_03=row[26],
                    variation_range_04=row[27],
                    variation_range_05=row[28],
                    variation_range_06=row[29],
                    variation_range_07=row[30],
                    variation_range_08=row[31],
                    variation_range_09=row[32],
                    variation_range_10=row[33],
                    variation_range_11=row[34],
                    variation_range_12=row[35],
                    variation_range_13=row[36],
                    variation_range_14=row[37],
                    variation_range_15=row[38],
                    variation_range_16=row[39],
                    variation_range_17=row[40],
                    variation_range_18=row[41],
                    variation_range_19=row[42],
                    variation_range_20=row[43],
                )
            )
            times.add(row[0])
            time_intervals.add(row[1])
        existing_data = cls.objects.filter(
            symbol=symbol, time__in=times, time_interval__in=time_intervals
        ).values_list("time", flat=True)
        tds_to_create = [td for td in tds if td.time not in existing_data]
        created = cls.objects.bulk_create(tds_to_create)
        if len(created) > 0:
            logger.warning(f"{symbol}: Created {len(created)} Training Data.")
        else:
            logger.warning(
                f"{symbol}: TD already exist: {tds} {tds_to_create}"
            )
        return sorted(created, key=lambda x: x.time, reverse=True)


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
        last_td = self.symbol.training_data.first()
        obs = {
            "variation_01": last_td.variation,
            "variation_range_01": None,
        }
        for i in range(2, self.get_window_size() + 1):
            obs[f"variation_{i:02d}"] = getattr(
                last_td, f"variation_{(i - 1):02d}"
            )
            obs[f"variation_range_{i:02d}"] = getattr(
                last_td, f"variation_range_{(i - 1):02d}"
            )

        pred = self.h_predict(obs)
        return pred[0]

    def predict_next_times(self, n):
        """
        Assumption: n < self.WINDOW
        """
        last_td = self.symbol.training_data.first()
        preds = []
        obss = []
        preds.append(self.predict_next())
        for i in range(2, n + 1):
            obs = {}
            for w in range(1, self.get_window_size() + 1):
                try:
                    obs[f"variation_{w:02d}"] = preds[w - 1]
                    obs[f"variation_range_{w:02d}"] = None
                except Exception:
                    obs[f"variation_{w:02d}"] = getattr(
                        last_td, f"variation_{w - i + 1:02d}"
                    )
                    obs[f"variation_range_{w:02d}"] = getattr(
                        last_td, f"variation_range_{w - i + 1:02d}"
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
            self.learning_fields = (
                self.learning_fields
                + ", "
                + ", ".join(
                    [f"variation_range_{j:02d}" for j in range(1, i + 1)]
                )
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
        self.learning_fields = (
            self.learning_fields
            + ", "
            + ", ".join(
                [f"variation_range_{j:02d}" for j in range(1, win_size + 1)]
            )
        )
        if save:  # pragma: no cover
            self.save()

    def calibrate_model(self, save=True):
        """
        TOOD: Review this
        """
        print("Before:", self.symbol.model_score)
        gs_eo = GridSearchCV(self.get_engine_object(), self.CALIBRATION_PARAMS)
        data, targets = self.get_data(), self.get_targets()
        gs_eo.fit(data, targets)
        for param, value in gs_eo.best_params_.items():
            setattr(self, param, value)
        logger.warning(f"{self.symbol}: Best params: {gs_eo.best_params_}")
        self.perform_inference(save=False)
        print("After:", self.metadata["inference"]["current"]["scores"]["r2"])
        if save:  # pragma: no cover
            self.save()
        return gs_eo.best_params_

    def get_window_size(self):
        return TrainingData.WINDOW

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


class TraderoBotGroupManager(models.Manager):
    def get_queryset(self):
        return (
            super()
            .get_queryset()
            .prefetch_related(
                models.Prefetch(
                    "bots", queryset=TraderoBot.objects.order_by("name")
                )
            )
        )


class TraderoBotGroup(models.Model):
    user = models.ForeignKey(
        User,
        verbose_name="User",
        related_name="botgroups",
        on_delete=models.PROTECT,
    )
    name = models.CharField(
        "Name",
        max_length=255,
        blank=True,
        null=True,
    )
    others = models.JSONField("Others", default=dict, blank=True)

    objects = TraderoBotGroupManager()

    class Meta:
        verbose_name = "Tradero Bots Group"
        verbose_name_plural = "Tradero Bots Groups"

    def __str__(self):
        return f"{self.name}"

    def get_absolute_url(self):
        return reverse("base:botzinhos-group-detail", kwargs={"pk": self.pk})

    def on(self):
        for bot in self.bots.all():
            if bot.status == TraderoBot.Status.INACTIVE:
                bot.on()
        return True

    def off(self):
        for bot in self.bots.all():
            if bot.status != TraderoBot.Status.INACTIVE:
                bot.off()
        return True

    def liquidate(self):
        for bot in self.bots.all():
            if bot.status == TraderoBot.Status.SELLING:
                bot.sell()
                bot.off()
            elif bot.status == TraderoBot.Status.BUYING:
                bot.off()
        return True

    @property
    def current_valuation(self):
        valuations = [
            bot.current_valuation
            for bot in self.bots.all()
            if bot.current_valuation
        ]
        return sum(valuations)

    @property
    def initial_valuation(self):
        valuations = [
            bot.executed_quote_asset
            or bot.fund_quote_asset
            or bot.fund_quote_asset_initial
            for bot in self.bots.all()
            if bot.executed_quote_asset
            or bot.fund_quote_asset
            or bot.fund_quote_asset_initial
        ]
        return sum(valuations)


class TraderoBotManager(models.Manager):
    def enabled(self):
        return self.get_queryset().exclude(
            models.Q(status=TraderoBot.Status.INACTIVE)
            | models.Q(user__trading_active=False)
        )

    def get_queryset(self):
        return super().get_queryset().select_related("symbol", "user")


class TraderoBot(models.Model):
    _client = None
    _strategies = get_strategies()
    _last_klines = None

    class Status(models.IntegerChoices):
        INACTIVE = 0, "Inactive"
        BUYING = 1, "Buying"
        SELLING = 2, "Selling"

    class Action(models.IntegerChoices):
        ERROR = -1, "Error"
        HOLD = 0, "Hold"
        BUY = 1, "Buy"
        SELL = 2, "Sell"
        JUMP = 3, "Jump"
        TURN_ON = 4, "Turn ON"
        TURN_OFF = 5, "Turn OFF"
        RESET = 6, "Reset"

    user = models.ForeignKey(
        User,
        verbose_name="User",
        related_name="bots",
        on_delete=models.PROTECT,
    )
    group = models.ForeignKey(
        TraderoBotGroup,
        verbose_name="Group",
        related_name="bots",
        on_delete=models.PROTECT,
    )
    symbol = models.ForeignKey(
        Symbol,
        verbose_name="Symbol",
        related_name="bots",
        on_delete=models.PROTECT,
    )
    name = models.CharField(
        "Name",
        max_length=255,
        blank=True,
        null=True,
    )
    strategy = models.CharField("Strategy", max_length=50, default="acmadness")
    strategy_params = models.CharField(
        "Strategy parameters",
        max_length=510,
        blank=True,
        null=True,
    )
    is_jumpy = models.BooleanField("Jumpy?", default=False)
    jumpy_whitelist = models.CharField(
        "Jumpy Symbols' Whitelist", max_length=1024, blank=True, null=True
    )
    jumpy_blacklist = models.CharField(
        "Jumpy Symbols' Blacklist", max_length=1024, blank=True, null=True
    )
    should_reinvest = models.BooleanField("Should Reinvest?", default=True)
    should_stop = models.BooleanField(
        "Should Stop After Selling?", default=False
    )
    is_dummy = models.BooleanField("Dummy?", default=True)
    status = models.SmallIntegerField(
        "Bot Status", choices=Status.choices, default=Status.INACTIVE
    )
    fund_base_asset = models.DecimalField(
        "Fund (Base Asset)",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    fund_quote_asset = models.DecimalField(
        "Fund (Quote Asset)",
        max_digits=40,
        decimal_places=8,
        validators=[MinValueValidator(Decimal(15))],
        blank=True,
        null=True,
    )
    fund_quote_asset_initial = models.DecimalField(
        "Initial Fund (Quote Asset)",
        max_digits=40,
        decimal_places=8,
        validators=[MinValueValidator(Decimal(15))],
        blank=True,
        null=True,
    )
    price_buying = models.DecimalField(
        "Buying Price the Base Asset",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    price_current = models.DecimalField(
        "Current Price of the Base Asset",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    price_selling = models.DecimalField(
        "Selling Price of the Base Asset",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    timestamp_start = models.DateTimeField(
        "Timestamp of Start Buying",
        blank=True,
        null=True,
    )
    timestamp_buying = models.DateTimeField(
        "Timestamp of Buying",
        blank=True,
        null=True,
    )
    timestamp_selling = models.DateTimeField(
        "Timestamp of Selling",
        blank=True,
        null=True,
    )
    receipt_buying = models.JSONField(
        "Receipt - Buying", default=dict, blank=True, null=True
    )
    receipt_selling = models.JSONField(
        "Receipt - Selling", default=dict, blank=True, null=True
    )
    others = models.JSONField("Others", default=dict)

    class Meta:
        verbose_name = "Tradero Bot"
        verbose_name_plural = "Tradero Bots"

    objects = TraderoBotManager()

    def __str__(self):
        return f"BOT #{self.id}: {self.symbol.symbol} | {self.get_status_display()}"

    def get_absolute_url(self):
        return reverse("base:botzinhos-detail", kwargs={"pk": self.pk})

    def save(self, *args, **kwargs):
        self.others["ws_group_name"] = f"bots_html_{self.user.username}"
        if self.jumpy_whitelist:
            self.jumpy_whitelist = self.jumpy_whitelist.upper()
        if self.jumpy_blacklist:
            self.jumpy_blacklist = self.jumpy_blacklist.upper()
        super().save(*args, **kwargs)
        if not self.name:
            self.name = f"{settings.BOT_DEFAULT_NAME}-{self.pk:03d}"
            super().save(*args, **kwargs)
        self.render_html_snippet(set_cache=True)
        async_to_sync(self.push_to_ws)()

    @property
    def current_valuation(self):
        if self.status == self.Status.INACTIVE:
            if self.receipt_buying:
                return self.fund_base_asset * self.price_current
            else:
                return self.fund_quote_asset or self.fund_quote_asset_initial
        elif self.status == self.Status.SELLING:
            return self.fund_base_asset * self.price_current
        else:
            return self.fund_quote_asset or self.fund_quote_asset_initial

    @property
    def executed_quote_asset(self):
        if self.receipt_buying:
            return Decimal(self.receipt_buying["cummulativeQuoteQty"])
        return None

    def get_client(self, reinit=False):  # pragma: no cover
        if not self._client or reinit:
            self._client = self.user.get_client()
        return self._client

    def get_strategy(self, symbol=None):
        return self._strategies[self.strategy](
            self, symbol, **self.get_strategy_params()
        )

    def get_strategy_params(self):
        params = {}
        if self.strategy_params:
            for pv in self.strategy_params.split(","):
                pv = pv.split("=")
                params[pv[0]] = pv[1]
        return params

    def on(self):
        if self.price_buying:
            self.status = self.Status.SELLING
        else:
            self.timestamp_start = self.timestamp_start or timezone.now()
            self.status = self.Status.BUYING
        self.local_memory_reset()
        self.log_trade()
        self.log(self.Action.TURN_ON, "Turned ON")
        self.save()
        return True

    def off(self):
        self.status = self.Status.INACTIVE
        self.local_memory_reset()
        self.log(self.Action.TURN_OFF, "Turned OFF")
        self.save()
        return True

    def reset(self):
        self.log_trade(cancelled=True)
        self.status = self.Status.INACTIVE
        self.receipt_buying, self.receipt_selling = None, None
        self.fund_base_asset, self.fund_quote_asset = None, None
        self.price_buying, self.price_selling, self.price_current = (
            None,
            None,
            None,
        )
        self.timestamp_start, self.timestamp_buying, self.timestamp_selling = (
            None,
            None,
            None,
        )
        self.log(self.Action.RESET, "RESET")
        self.save()
        return True

    def buy(self):
        if not self.receipt_buying:
            try:
                client = self.get_client()
                amount = self.fund_quote_asset or self.fund_quote_asset_initial
                (
                    success,
                    price,
                    quantity,
                    receipt,
                    message,
                ) = client.tradero_buy(
                    self.symbol,
                    amount,
                    dummy=self.is_dummy,
                )
                if success:
                    self.receipt_buying = receipt
                    self.save()
                    self.timestamp_buying = timezone.now()
                    if not self.timestamp_start:  # pragma: no cover
                        self.timestamp_start = self.timestamp_buying
                    self.price_buying = price
                    self.fund_base_asset = quantity
                    self.log(self.Action.BUY)
                    self.log_trade()
                    self.fund_quote_asset = (
                        None  # Unexecuted FQA is not logged when Buying
                    )
                    if self.status != self.Status.INACTIVE:
                        self.status = self.Status.SELLING
                    self.save()
                    return True
            except Exception as e:
                message = str(e)
        else:
            message = "Already bought, reset the bot if the problem persists"
        self.log(self.Action.ERROR, message=message)
        self.save()
        return False

    def sell(self):
        if not self.receipt_selling:
            try:
                client = self.get_client()
                (
                    success,
                    price,
                    quantity,
                    receipt,
                    message,
                ) = client.tradero_sell(
                    self.symbol,
                    self.fund_base_asset,
                    dummy=self.is_dummy,
                )
                if success:
                    self.receipt_selling = receipt
                    self.save()
                    self.price_selling = price
                    self.fund_quote_asset = quantity
                    self.timestamp_selling = timezone.now()
                    self.log(self.Action.SELL)
                    self.log_trade()
                    # Reset state
                    self.fund_base_asset = self.fund_base_asset - Decimal(
                        self.receipt_selling["executedQty"]
                    )
                    if not self.should_reinvest:
                        self.fund_quote_asset = self.fund_quote_asset_initial
                    self.timestamp_selling, self.timestamp_buying = None, None
                    self.receipt_selling, self.receipt_buying = None, None
                    self.price_selling, self.price_buying = None, None
                    if self.status != self.Status.INACTIVE:
                        self.status = self.Status.BUYING
                    if self.should_stop:
                        self.status = self.Status.INACTIVE
                        self.timestamp_start = None
                    else:
                        self.timestamp_start = timezone.now()
                    self.save()
                    return True
            except Exception as e:
                message = str(e)
        else:
            message = "Already sold, reset the bot if the problem persists"
        self.log(self.Action.ERROR, message=message)
        self.save()
        return False

    def jump(self, to_symbol):
        current_symbol = self.symbol
        fba = self.fund_base_asset
        fba_msg = (
            f" (leaving {fba:.6f} {current_symbol.symbol} behind)"
            if fba
            else ""
        )
        self.symbol = to_symbol
        self.fund_base_asset = None
        self.price_current = self.get_current_price()
        self.local_memory_reset()
        self.log(
            self.Action.JUMP,
            f"Jumped from {current_symbol} to {to_symbol}{fba_msg}",
        )
        self.save()

    def decide(self):
        strategy = self.get_strategy()
        self.local_memory_update(strategy)
        if self.status == self.Status.BUYING:
            should_buy, message = strategy.evaluate_buy()
            if should_buy:
                self.buy()
                return
            if self.is_jumpy:
                should_jump, symbol = strategy.evaluate_jump()
                if should_jump:
                    self.jump(symbol)
                    self.get_strategy()  # Update values
                    self.decide()
                    return
                message += " and no other symbol to go was found."
        if self.status == self.Status.SELLING:
            should_sell, message = strategy.evaluate_sell()
            if should_sell:
                self.sell()
                return
        self.log(self.Action.HOLD, message)
        self.save()

    def local_memory_update(self, strategy=None):
        strategy = strategy or self.get_strategy()
        strategy.local_memory_update()

    def has_local_memory(self, symbol=None, strategy=None):
        symbol = symbol or self.symbol
        strategy = strategy or self.get_strategy()
        return strategy.has_local_memory(symbol)

    def local_memory_reset(self):
        self.others["local_memory"] = {}

    def get_local_memory(self, symbol=None):
        symbol = symbol or self.symbol
        if self.others.get("local_memory", {}):
            return self.others["local_memory"].get(symbol.symbol, {})
        else:
            self.others["local_memory"] = {}
            return self.others["local_memory"]

    def set_local_memory(self, symbol=None, value={}):
        symbol = symbol or self.symbol
        self.others["local_memory"][self.symbol.symbol] = value

    def get_current_price(self):
        client = self.get_client()
        price = client.ticker_price(symbol=self.symbol.symbol)["price"]
        return Decimal(price)

    def log(self, action, message=None):
        log = TraderoBotLog(
            bot=self,
            is_dummy=self.is_dummy,
            status=self.status,
            action=action,
            fund_base_asset=self.fund_base_asset,
            fund_quote_asset=self.fund_quote_asset,
            price_buying=self.price_buying,
            price_current=self.price_current,
            message=message,
        )
        if action == self.Action.BUY:
            log.message = (
                f"Bought {self.fund_base_asset:.3f} of {self.symbol} at "
                f"{self.price_buying:.6f} ("
                f"{self.fund_base_asset * self.price_buying:.6f} "
                f"{self.symbol.quote_asset})"
            )
        if action == self.Action.SELL:
            log.price_selling = self.price_selling
            log.variation = (self.price_selling / self.price_buying - 1) * 100
            log.message = (
                f"BOOM! Sold {self.fund_base_asset:.3f} of {self.symbol} at "
                f"{self.price_selling:.6f} - VAR: {log.variation:.3f}%"
            )
        log.save()
        last_logs = self.others.get("last_logs", [])
        last_logs.append(f"{log.timestamp:%Y-%m-%d %H:%M:%S}| {log.message}")
        self.others["last_logs"] = last_logs[-4:]

    def log_trade(self, cancelled=False):
        trade_history, _ = self.trades.update_or_create(
            bot=self,
            user=self.user,
            timestamp_start=self.timestamp_start,
            defaults={
                "is_dummy": self.is_dummy,
                "symbol": self.symbol,
                "strategy": self.strategy,
                "strategy_params": self.strategy_params,
                "timestamp_start": self.timestamp_start,
                "timestamp_buying": self.timestamp_buying,
                "timestamp_selling": self.timestamp_selling,
                "timestamp_cancelled": (timezone.now() if cancelled else None),
                "fund_base_asset": self.fund_base_asset,
                "price_buying": self.price_buying,
                "price_selling": self.price_selling,
                "receipt_buying": self.receipt_buying,
                "receipt_selling": self.receipt_selling,
            },
        )
        trade_history.save()
        return trade_history

    def get_last_log_message(self):
        last_log = self.logs.last()
        if last_log:
            return last_log.message
        return None

    def get_jumpy_blacklist(self):
        if self.jumpy_blacklist:
            return [s.strip().upper() for s in self.jumpy_blacklist.split(",")]
        return []

    def get_jumpy_whitelist(self):
        if self.jumpy_whitelist:
            return [s.strip().upper() for s in self.jumpy_whitelist.split(",")]
        return []

    def render_html_snippet(self, set_cache=False):
        cache_key = f"bot_{self.pk}_html"
        html = cache.get(cache_key)
        if not html or set_cache:
            html = render_to_string(
                "base/bot_snippet.html",
                {
                    "bot": self,
                },
            )
            cache.set(cache_key, html, settings.TIME_INTERVAL_BOTS * 60 + 9)
        return html

    async def push_to_ws(self):
        bot = f"botzinho-{self.pk}"
        text = self.render_html_snippet()
        await channel_layer.group_send(
            self.others["ws_group_name"],
            {
                "type": "bot.html.message",
                "message": {
                    "type": "bot_update",
                    "bot": bot,
                    "text": text,
                },
            },
        )

    def clean(self):
        super().clean()
        # Check validity of jumpy_whitelist
        jumpy_whitelist = self.get_jumpy_whitelist()
        if jumpy_whitelist:
            if len(Symbol.objects.filter(symbol__in=jumpy_whitelist)) < len(
                jumpy_whitelist
            ):
                raise ValidationError(
                    {"jumpy_whitelist": "Unrecognized Symbols."}
                )
        # Check validity of jumpy_blacklist
        jumpy_blacklist = self.get_jumpy_blacklist()
        if jumpy_blacklist:
            if len(Symbol.objects.filter(symbol__in=jumpy_blacklist)) < len(
                jumpy_blacklist
            ):
                raise ValidationError(
                    {"jumpy_blacklist": "Unrecognized Symbols."}
                )
        # Check either fund_quote_asset or fund_quote_asset_initial are present
        if not (self.fund_quote_asset or self.fund_quote_asset_initial):
            raise ValidationError(
                {
                    "fund_quote_asset_initial": (
                        "Either 'Fund (Quote Asset)' or "
                        "'Initial Fund (Quote Asset)' is required"
                    )
                }
            )
        # Check vailidity of strategy params
        try:
            strategy = self.get_strategy()
        except Exception:
            raise ValidationError(
                {
                    "strategy_params": (
                        "Unrecognized parameters or format for the strategy"
                    )
                }
            )
        for param in self.get_strategy_params():
            if param not in strategy.params:
                raise ValidationError(
                    {
                        "strategy_params": (
                            "Unrecognized parameters or format for the strategy"
                        )
                    }
                )

    @classmethod
    def update_all_bots(cls):
        client = Spot()
        bots = cls.objects.enabled()
        if bots:
            symbols = list(
                Symbol.objects.filter(bots__in=bots)
                .distinct()
                .values_list("symbol", flat=True)
            )
            prices = {
                t["symbol"]: t["price"]
                for t in client.ticker_price(symbols=symbols)
            }
            for bot in bots:
                bot.price_current = Decimal(prices[bot.symbol.symbol])
                bot.decide()
            message = f"{len(bots)} bots updated."
        else:  # pragma: no cover
            message = "No bots were found."
        logger.warning(message)
        return message


class TraderoBotLog(models.Model):
    bot = models.ForeignKey(
        TraderoBot,
        verbose_name="User",
        related_name="logs",
        on_delete=models.PROTECT,
    )
    is_dummy = models.BooleanField("Dummy?")
    timestamp = models.DateTimeField("Timestamp", auto_now_add=True)
    status = models.SmallIntegerField(
        "Bot Status",
        choices=TraderoBot.Status.choices,
    )
    action = models.SmallIntegerField(
        "Bot Action",
        choices=TraderoBot.Action.choices,
    )
    fund_base_asset = models.DecimalField(
        "Fund (Base Asset)",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    fund_quote_asset = models.DecimalField(
        "Fund (Quote Asset)",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    price_buying = models.DecimalField(
        "Buying Price the Base Asset (Net)",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    price_current = models.DecimalField(
        "Current Market Price of the Base Asset",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    price_selling = models.DecimalField(
        "Selling Price of the Base Asset (Net)",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    variation = models.DecimalField(
        "Porcentual Variation between Buying and Selling Price",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    message = models.CharField(
        "Extra message (others)",
        max_length=2048,
        blank=True,
        null=True,
    )

    class Meta:
        verbose_name = "Tradero Bot Log"
        verbose_name_plural = "Tradero Bots Logs"

    def __str__(self):
        return (
            f"{self.bot} {'[[DUMMY]]' if self.is_dummy else ''}| "
            f"{self.timestamp} | [{self.get_action_display()}] "
            f"FBA: {self.fund_base_asset}, "
            f"FQA: {self.fund_quote_asset}, PB(N): {self.price_buying}, "
            f"PC(M): {self.price_current}, PS(N): {self.price_selling}, "
            f"VAR: {self.variation} || {self.message}"
        )

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        logger.warning(str(self))


class TradeHistoryManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().select_related("symbol", "user", "bot")


class TradeHistory(models.Model):
    """
    Highly denormalized on purpose
    """

    user = models.ForeignKey(
        User,
        verbose_name="User",
        related_name="trades",
        on_delete=models.PROTECT,
    )
    bot = models.ForeignKey(
        TraderoBot,
        verbose_name="Bot",
        related_name="trades",
        on_delete=models.PROTECT,
    )
    is_dummy = models.BooleanField("Dummy?")
    symbol = models.ForeignKey(
        Symbol,
        verbose_name="Symbol",
        related_name="trades_history",
        on_delete=models.PROTECT,
    )
    strategy = models.CharField(
        "Strategy",
        max_length=50,
        blank=True,
        null=True,
    )
    strategy_params = models.CharField(
        "Strategy parameters",
        max_length=255,
        blank=True,
        null=True,
    )
    timestamp_start = models.DateTimeField(
        "Timestamp - Start",
        blank=True,
        null=True,
    )
    timestamp_buying = models.DateTimeField(
        "Timestamp - Buying",
        blank=True,
        null=True,
    )
    timestamp_selling = models.DateTimeField(
        "Timestamp - Selling",
        blank=True,
        null=True,
    )
    timestamp_cancelled = models.DateTimeField(
        "Timestamp - Cancelled",
        blank=True,
        null=True,
    )
    fund_base_asset = models.DecimalField(
        "Fund (Base Asset)",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    fund_base_asset_exec = models.DecimalField(
        "Fund (Base Asset) Executed",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    fund_base_asset_unexec = models.DecimalField(
        "Fund (Base Asset) Unexecuted",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    fund_quote_asset = models.DecimalField(
        "Fund (Quote Asset)",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    price_buying = models.DecimalField(
        "Net Buying Price the Base Asset",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    price_selling = models.DecimalField(
        "Net Selling Price of the Base Asset",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    commission_buying = models.DecimalField(
        "Comission (Buying)",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    commission_buying_asset = models.CharField(
        "Comission Asset (Buying)",
        max_length=10,
        blank=True,
        null=True,
    )
    commission_selling = models.DecimalField(
        "Comission (Selling)",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    commission_selling_asset = models.CharField(
        "Comission Asset (Selling)",
        max_length=10,
        blank=True,
        null=True,
    )
    receipt_buying = models.JSONField(
        "Receipt - Buying", default=dict, blank=True, null=True
    )
    receipt_selling = models.JSONField(
        "Receipt - Selling", default=dict, blank=True, null=True
    )
    is_complete = models.BooleanField("Is Trade Complete?", default=False)
    variation = models.DecimalField(
        "Porcentual Variation between Buying and Selling Price",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    variation_quote_asset = models.DecimalField(
        "Variation of the Quote Asset",
        max_digits=40,
        decimal_places=8,
        blank=True,
        null=True,
    )
    duration_seeking = models.DurationField(
        "Elapsed time looking for buy",
        blank=True,
        null=True,
    )
    duration_trade = models.DurationField(
        "Elapsed time between Buying and Selling",
        blank=True,
        null=True,
    )
    duration_total = models.DurationField(
        "Total Elapsed time since start",
        blank=True,
        null=True,
    )

    class Meta:
        verbose_name = "Trade History"
        verbose_name_plural = "Trades History"

    objects = TradeHistoryManager()

    def __str__(self):
        return (
            f"{self.user}: #{self.bot.id} {'[[DUMMY]]' if self.is_dummy else ''} |"
            f" {self.symbol.symbol} | {self.timestamp_start} - "
            f"{self.timestamp_selling or '...'}:: Var: {self.variation}% "
            f"Var FQA: {self.variation_quote_asset}"
        )

    def save(self, *args, **kwargs):
        if self.timestamp_buying and self.timestamp_start:
            self.duration_seeking = (
                self.timestamp_buying - self.timestamp_start
            )
        if (
            self.timestamp_selling
            and self.timestamp_buying
            and self.timestamp_start
        ):
            self.duration_trade = (
                self.timestamp_selling - self.timestamp_buying
            )
            self.duration_total = self.timestamp_selling - self.timestamp_start
        if self.price_buying and self.price_selling:
            self.variation = (self.price_selling / self.price_buying - 1) * 100
        if self.receipt_buying:
            self.fund_quote_asset = Decimal(
                self.receipt_buying["cummulativeQuoteQty"]
            )
            (
                self.commission_buying,
                self.commission_buying_asset,
            ) = get_commission(self.receipt_buying)
        if self.receipt_buying and self.receipt_selling:
            self.is_complete = True
            self.fund_base_asset = Decimal(
                self.receipt_buying["executedQty"]
            ) - (
                self.commission_buying
                if self.commission_buying_asset != "BNB"
                else 0
            )
            self.fund_base_asset_exec = Decimal(
                self.receipt_selling["executedQty"]
            )
            self.fund_base_asset_unexec = self.fund_base_asset - Decimal(
                self.fund_base_asset_exec
            )
            (
                self.commission_selling,
                self.commission_selling_asset,
            ) = get_commission(self.receipt_selling)
            self.variation_quote_asset = (
                Decimal(self.receipt_selling["cummulativeQuoteQty"])
                - (
                    self.commission_selling
                    if self.commission_selling_asset != "BNB"
                    else 0
                )
                - self.fund_quote_asset
            )
        super().save(*args, **kwargs)

    @classmethod
    def summary(cls, trades_qs, checkpoint=None):
        result = {"rows": {}}
        dates = {
            "24h": timezone.now() - timezone.timedelta(days=1),
            "1w": timezone.now() - timezone.timedelta(days=7),
            "checkpoint": checkpoint,
            "alltime": None,
        }
        for label, t in dates.items():
            t_qs = trades_qs.filter(is_complete=True)
            if t:
                t_qs = t_qs.filter(timestamp_selling__gte=t)
            result["rows"][label] = t_qs.aggregate(
                variation_quote_asset_total=models.Sum(
                    "variation_quote_asset"
                ),
                variation_average=models.Avg("variation"),
                trades_quantity=models.Count("pk"),
            )
        result["meta"] = {
            "descriptions": {
                "rows": {
                    "24h": "Last 24 Hours",
                    "1w": "Last Week",
                    "checkpoint": "Checkpoint",
                    "alltime": "All-Time",
                },
                "cols": {
                    "variation_quote_asset_total": "Total Gains (QA)",
                    "variation_average": "Avg. % Var",
                    "trades_quantity": "#Trades",
                },
                "formats": {
                    "variation_quote_asset_total": ".4f",
                    "variation_average": ".3f",
                    "trades_quantity": "i",
                },
            }
        }
        return result

    @classmethod
    def summary_for_object(cls, obj=None):
        if isinstance(obj, TraderoBot):
            qs = cls.objects.filter(bot=obj)
            checkpoint = obj.user.checkpoint
        elif isinstance(obj, User):
            qs = cls.objects.filter(user=obj)
            checkpoint = obj.checkpoint
        else:  # TraderoBotGroup object
            qs = cls.objects.filter(bot__group=obj)
            checkpoint = obj.user.checkpoint
        result = {
            "object": obj,
            "cp": checkpoint,
            "dummy": cls.summary(qs.filter(is_dummy=True), checkpoint),
            "real": cls.summary(qs.filter(is_dummy=False), checkpoint),
        }
        return result
