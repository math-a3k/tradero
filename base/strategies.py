import inspect
import sys
from decimal import Decimal

from django.db.models import Q
from django.utils import timezone


def get_strategies():
    available_strats = {
        i.slug: i
        for j, i in inspect.getmembers(sys.modules[__name__], inspect.isclass)
        if getattr(i, "slug", None)
    }
    return available_strats


class TradingStrategy:
    bot = None
    slug = None

    def evaluate_buy(self):
        raise NotImplementedError

    def evaluate_sell(self):
        raise NotImplementedError

    def evaluate_jump(self):
        raise NotImplementedError

    @property
    def time_safeguard(self):
        return (timezone.now() - self.bot.symbol.last_updated).seconds > 60

    def local_memory_update(self):
        pass

    def has_local_memory(self, symbol):
        pass


class ACMadness(TradingStrategy):
    """
    ACMAdness Trading Strategy

    (requires the STP indicator)
    """

    slug = "acmadness"

    def __init__(self, bot, symbol=None, **kwargs):
        self.bot = bot
        self.symbol = symbol or self.bot.symbol
        self.ac = Decimal(self.bot.symbol.others["stp"]["next_n_sum"])
        self.microgain = Decimal(kwargs.get("microgain", "0.3"))
        self.ac_threshold = self.microgain * Decimal(
            kwargs.get("ac_factor", 3)
        )
        self.keep_going = bool(int(kwargs.get("keep_going", "0")))
        self.outlier_protection = bool(int(kwargs.get("ol_prot", "1")))
        self.max_var_protection = Decimal(kwargs.get("max_var_prot", "0"))
        self.ac_adjusted = Decimal(kwargs.get("ac_adjusted", "1"))

    def evaluate_buy(self):
        if self.time_safeguard:
            return (
                False,
                "Time Safeguard - waiting for next turn...",
            )
        if (
            self.outlier_protection
            and self.bot.symbol.others["outliers"]["o1"]
        ):
            return (
                False,
                "Outlier Protection - waiting for next turn...",
            )
        if (
            self.max_var_protection > 0
            and self.bot.symbol.last_variation > self.max_var_protection
        ):
            return (
                False,
                f"Max Var Protection ({self.bot.symbol.last_variation:.3f} > "
                f"{self.max_var_protection:.3f}) - waiting for next turn...",
            )
        if self.get_ac() > self.ac_threshold:
            return True, None
        return (
            False,
            f"Current AC {'(adj) ' if self.ac_adjusted else '' }"
            f"({self.get_ac():.3f}) is below threshold "
            f"({self.ac_threshold:.3f})",
        )

    def evaluate_sell(self):
        selling_threshold = Decimal(self.bot.price_buying) * Decimal(
            1 + (self.microgain / 100)
        )
        if self.bot.price_current > selling_threshold:
            if self.keep_going:  # and self.cg:
                should_buy, message = self.evaluate_buy()
                if should_buy:
                    return (
                        False,
                        "Kept goin' due to current AC",
                    )
            return True, None
        return (
            False,
            f"Current price ({self.bot.price_current:.8f}) is below threshold "
            f"({selling_threshold:.8f})",
        )

    def evaluate_jump(self):
        if self.time_safeguard:
            return False, None
        symbols_with_siblings = list(
            self.bot.user.bots.filter(
                Q(
                    timestamp_buying__gt=timezone.now()
                    - timezone.timedelta(hours=1)
                )
                | Q(
                    timestamp_start__gt=timezone.now()
                    - timezone.timedelta(hours=1)
                ),
                status__gte=self.bot.Status.INACTIVE,
                strategy=self.bot.strategy,
                strategy_params=self.bot.strategy_params,
            ).values_list("symbol", flat=True)
        )
        symbols = self.bot.symbol._meta.concrete_model.objects.top_symbols()
        symbols = sorted(
            symbols, key=lambda s: s.others["stp"]["next_n_sum"], reverse=True
        )
        symbols_blacklist = self.bot.get_jumpy_blacklist()
        symbols_whitelist = self.bot.get_jumpy_whitelist()
        for symbol in symbols:
            symbol_ac = Decimal(symbol.others["stp"]["next_n_sum"])
            symbol_ac = (
                symbol_ac * symbol.model_score
                if self.ac_adjusted
                else symbol_ac
            )
            if (
                symbol_ac > self.ac_threshold
                and symbol.pk not in symbols_with_siblings
                and symbol.symbol not in symbols_blacklist
            ):
                if (
                    symbols_whitelist
                    and symbol.symbol not in symbols_whitelist
                ):
                    continue
                if self.outlier_protection:
                    if symbol.others["outliers"]["o1"]:
                        continue
                return True, symbol
        return False, None

    def get_ac(self):
        if self.ac_adjusted:
            return self.ac * self.bot.symbol.model_score
        else:
            return self.ac


class CatchTheWave(TradingStrategy):
    """
    Catch The Wave Trading Strategy

    (requires the SCG indicator)
    """

    slug = "catchthewave"

    def __init__(self, bot, symbol=None, **kwargs):
        self.bot = bot
        self.symbol = symbol or self.bot.symbol
        self.local_memory = self.bot.get_local_memory(self.symbol)
        self.scg = self.symbol.others["scg"]
        self.diff_sm = self.scg["line_diff_sm"]  # Short - Middle tendency
        self.s_var = self.scg["line_s_var"]  # Var of the Short tendency
        self.l_var = self.scg["line_l_var"]  # Var of the Long tendency
        #
        self.early_onset = bool(int(kwargs.get("early_onset", "0")))
        self.sell_on_maxima = bool(int(kwargs.get("sell_on_maxima", "1")))
        self.onset_periods = int(kwargs.get("onset_periods", "2"))
        self.maxima_tol = Decimal(kwargs.get("maxima_tol", "0.1"))
        self.sell_safeguard = Decimal(kwargs.get("sell_safeguard", "0.3"))
        self.use_local_memory = bool(int(kwargs.get("use_local_memory", "1")))
        self.use_matrix_time_res = bool(
            int(kwargs.get("use_matrix_time_res", "0"))
        )

    def evaluate_buy(self):
        if self.use_matrix_time_res and self.time_safeguard:
            return (False, "Holding - Using matrix's time resolution...")
        if self.is_on_good_status() and self.is_on_wave_onset():
            return True, None
        return (False, "Symbol is not in good status and ascending...")

    def evaluate_sell(self):
        if self.use_matrix_time_res and self.time_safeguard:
            return (False, "Holding - Using matrix's time resolution")
        if self.bot.price_current > self.get_min_selling_threshold():
            if self.sell_on_maxima and self.is_local_maxima():
                return True, None
            if self.is_on_wave():
                return (False, "Kept goin' due to still being in the wave")
            return True, None
        return (
            False,
            f"Current price ({self.bot.price_current:.8f}) is below min. "
            f"selling threshold ({self.get_min_selling_threshold():.8f})",
        )

    def evaluate_jump(self):
        if self.use_matrix_time_res and self.time_safeguard:
            return False, None
        symbols_with_siblings = self.get_symbols_with_siblings()
        symbols = self.bot.symbol._meta.concrete_model.objects.top_symbols()
        if self.early_onset:
            key = lambda s: s.others["scg"]["seo_index"]
        else:
            key = lambda s: s.others["scg"]["scg_index"]
        symbols = sorted(
            symbols,
            key=key,
            reverse=True,
        )
        symbols_blacklist = self.bot.get_jumpy_blacklist()
        symbols_whitelist = self.bot.get_jumpy_whitelist()
        for symbol in symbols:
            strat_in_symbol = self.bot.get_strategy(symbol)
            should_buy, _ = strat_in_symbol.evaluate_buy()
            if (
                should_buy
                and symbol.pk not in symbols_with_siblings
                and symbol.symbol not in symbols_blacklist
            ):
                if (
                    symbols_whitelist
                    and symbol.symbol not in symbols_whitelist
                ):
                    continue
                return True, symbol
        return False, None

    def get_symbols_with_siblings(self):
        return self.bot.user.bots.filter(
            status__gte=self.bot.Status.INACTIVE,
            strategy=self.bot.strategy,
            strategy_params=self.bot.strategy_params,
        ).values_list("symbol", flat=True)

    def all_positive(self, ts):
        return all([t > 0 for t in ts])

    def is_on_good_status(self):
        if self.early_onset:
            return self.scg["early_onset"] and self.not_decreasing(
                self.l_var[-self.onset_periods :]  # long-term line
            )
        return self.scg["current_good"]

    def is_local_maxima(self):
        if self.local_memory_available():
            return self.local_memory["price_var"][-1] < -self.maxima_tol
        return self.s_var[-1] * 100 < -self.maxima_tol

    def is_on_wave(self):
        return self.diff_sm[-1] > 0

    def is_on_wave_onset(self):
        if self.local_memory_available():
            return self.all_positive(
                self.local_memory["price_var"][-self.onset_periods :]
            )
        return self.all_positive(self.s_var[-self.onset_periods :])

    def not_decreasing(self, line):
        return all([p * 100 > -self.maxima_tol for p in line])

    def get_min_selling_threshold(self):
        return self.bot.price_buying * (1 + self.sell_safeguard / 100)

    def local_memory_update(self):
        if self.use_matrix_time_res and self.time_safeguard:
            return
        lm = self.bot.get_local_memory()
        if not lm:
            lm = {"price": [], "price_var": []}
        price, price_var = lm["price"], lm["price_var"]
        if self.bot.price_current:
            price = price + [float(self.bot.price_current)]
            if len(price) > 1:
                last_var = (Decimal(price[-1]) / Decimal(price[-2]) - 1) * 100
                price_var = price_var + [float(last_var)]
        lm = {
            "price": price[-100:],
            "price_var": price_var[-100:],
        }
        self.local_memory = lm
        self.bot.set_local_memory(self.symbol, lm)

    def has_local_memory(self, symbol=None):
        symbol = symbol or self.symbol
        lm = self.bot.get_local_memory(symbol)
        price = lm.get("price", [])
        if len(price) > 1:
            return True
        return False

    def local_memory_available(self):
        return self.use_local_memory and self.bot.has_local_memory(
            self.symbol, strategy=self
        )
