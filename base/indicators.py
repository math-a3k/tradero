import inspect
import sys

import pandas as pd
from django.conf import settings
from statsmodels.tsa.api import Holt

from .utils import count_periods_from_now


def get_indicators():
    available_inds = {
        i.slug: i
        for j, i in inspect.getmembers(sys.modules[__name__], inspect.isclass)
        if getattr(i, "slug", None)
    }
    if settings.INDICATORS == ["__all__"]:
        return available_inds
    else:
        indicators = {}
        for indicator in settings.INDICATORS:
            indicators[indicator] = available_inds[indicator]
        return indicators


class Indicator:
    slug = None  # slug to be used in the backend
    template = None  # template snippet in the Symbol snippet
    js_slug = None  # slug to be used in UI
    js_sorting = None  # js snippet to obtain the value for sorting in the UI
    symbol = None
    value = None  # the value of the indicator at the moment of calculation

    def __init__(self, symbol):  # pragma: no cover
        self.symbol = symbol

    def calculate(self):  # pragma: no cover
        self.value = None
        return self.value


class MACDCG(Indicator):
    """
    Moving Average Convergence / Divergence - Current Good
    """

    slug = "macdcg"
    template = "base/indicators/_macd.html"
    js_slug = "cg"
    js_sorting = "base/indicators/_macd.js"
    #
    a, b, c = (None, None, None)

    def __init__(
        self,
        symbol,
        a=settings.MACD_CG[0],  # Middle-term tendency
        b=settings.MACD_CG[1],  # Long-term tendency
        c=settings.MACD_CG[2],  # Short-term tendency
    ):
        self.symbol = symbol
        self.a, self.b, self.c = a, b, c

    def calculate(self):
        tds = reversed(
            self.symbol.training_data.all()
            .order_by("-time")[: self.b * 2]
            .values_list("price_close", flat=True)
        )
        tds = pd.Series(tds)

        ts_a = tds.rolling(self.a).mean().dropna()
        ts_b = tds.rolling(self.b).mean().dropna()
        ts_c = tds.rolling(self.c).mean().dropna()
        min_len = min(len(ts_a), len(ts_b), len(ts_c))
        ts_a = ts_a[-min_len:].reset_index(drop=True)
        ts_b = ts_b[-min_len:].reset_index(drop=True)
        ts_c = ts_c[-min_len:].reset_index(drop=True)
        ts_c_var = ts_c.diff().dropna()
        ts_diff_ab = ts_a - ts_b
        ts_diff_ca = ts_c - ts_a

        ts_a_e = pd.Series(Holt(ts_a.tolist()).fit().fittedvalues)
        ts_b_e = pd.Series(Holt(ts_b.tolist()).fit().fittedvalues)
        ts_c_e = pd.Series(Holt(ts_c.tolist()).fit().fittedvalues)
        ts_c_e_var = ts_c_e.diff().dropna()
        ts_diff_e_ab = ts_a_e - ts_b_e
        ts_diff_e_ca = ts_c_e - ts_a_e

        macd_line = ts_a_e - ts_b_e
        macd_signal = (
            macd_line.rolling(self.c).mean().dropna().reset_index(drop=True)
        )
        macd_line = macd_line[(self.c - 1) :].reset_index(drop=True).to_list()
        signal_diff = (macd_line - macd_signal).to_list()
        current_good = (macd_line[-1] > 0) and (signal_diff[-1] > 0)

        self.value = {
            "params": [
                self.a,
                self.b,
                self.c,
            ],
            "smas": {
                "a": ts_a[-min_len:].tolist(),
                "b": ts_b[-min_len:].tolist(),
                "c": ts_c[-min_len:].tolist(),
                "c_var": ts_c_var.tolist(),
                "diff_ab": ts_diff_ab.tolist(),
                "diff_ca": ts_diff_ca.tolist(),
            },
            "emas": {
                "a": ts_a_e[-min_len:].tolist(),
                "b": ts_b_e[-min_len:].tolist(),
                "c": ts_c_e[-min_len:].tolist(),
                "c_var": ts_c_e_var.tolist(),
                "diff_ab": ts_diff_e_ab.tolist(),
                "diff_ca": ts_diff_e_ca.tolist(),
            },
            "macd_line": macd_line[-self.c :],
            "macd_line_last": macd_line[-1],
            "signal": macd_signal[-self.c :].to_list(),
            "signal_diff": signal_diff[-self.c :],
            "current_good": bool(current_good),
        }
        return self.value


class SCG(Indicator):
    """
    Simple Current Good Indicator
    """

    slug = "scg"
    template = "base/indicators/_scg.html"
    js_slug = "scg"
    js_sorting = "base/indicators/_scg.js"
    #
    (
        s,
        m,
        l,
    ) = (None, None, None)

    def __init__(
        self,
        symbol,
        s=settings.SCG[0],  # Short-term tendency
        m=settings.SCG[1],  # Middle-term tendency
        l=settings.SCG[2],  # Long-term tendency
    ):
        self.symbol = symbol
        self.s, self.m, self.l = s, m, l

    def calculate(self):
        tds = reversed(
            self.symbol.training_data.all()
            .order_by("-time")[: self.l * 2]
            .values_list("price_close", flat=True)
        )
        tds = pd.Series(tds)

        ts_s = tds.rolling(self.s).mean().dropna()
        ts_m = tds.rolling(self.m).mean().dropna()
        ts_l = tds.rolling(self.l).mean().dropna()
        min_len = min(len(ts_s), len(ts_m), len(ts_l))
        ts_m = ts_m[-min_len:].reset_index(drop=True)
        ts_l = ts_l[-min_len:].reset_index(drop=True)
        ts_s = ts_s[-min_len:].reset_index(drop=True)
        ts_s_var = ts_s.pct_change().dropna()
        ts_l_var = ts_l.pct_change().dropna()
        ts_diff_ml = ts_m - ts_l
        ts_diff_sm = ts_s - ts_m
        ts_diff_sm_var = ts_diff_ml.pct_change().dropna()
        ts_diff_ml_var = ts_diff_ml.pct_change().dropna()

        cg = (ts_diff_ml > 0) & (ts_diff_sm > 0)
        cg_periods = count_periods_from_now(cg)
        if cg_periods == 0:
            scg_index = 0
        else:
            scg_index = (
                100 - cg_periods + ts_diff_ml_var[len(cg) - 1 - cg_periods]
            )

        eo = ts_diff_sm > 0
        early_onset_periods = count_periods_from_now(eo)
        early_onset = early_onset_periods > 0
        seo_index = (
            100
            - early_onset_periods
            + ts_diff_sm_var[len(eo) - 1 - early_onset_periods]
            if early_onset
            else 0
        )

        self.value = {
            "params": [
                self.s,
                self.m,
                self.l,
            ],
            "line_m": ts_m.to_list(),
            "line_l": ts_l.to_list(),
            "line_s": ts_s.to_list(),
            "line_s_var": ts_s_var.to_list(),
            "line_l_var": ts_l_var.to_list(),
            "line_diff_ml": ts_diff_ml.to_list(),
            "line_diff_sm": ts_diff_sm.to_list(),
            "current_good": bool(cg[len(cg) - 1]),
            "current_good_periods": cg_periods,
            "scg_index": scg_index,
            "early_onset": early_onset,
            "early_onset_periods": early_onset_periods,
            "seo_index": seo_index,
        }
        return self.value


class STP(Indicator):
    """
    Short Term Prediction
    """

    slug = "stp"
    template = "base/indicators/_stp.html"
    js_slug = "ac"
    js_sorting = "base/indicators/_stp.js"

    def __init__(
        self,
        symbol,
        periods=settings.STP,
    ):
        self.symbol = symbol
        self.periods = periods

    def calculate(self):
        stp = {"params": self.periods}
        stp["last_n"] = [
            float(d)
            for d in [
                getattr(self.symbol._last_td, f"variation_{i:02d}")
                for i in range(1, self.periods + 1)
            ]
        ]
        stp["last_n_sum"] = sum(stp["last_n"])
        stp["next_n"] = self.symbol.get_prediction_model().predict_next_times(
            self.periods
        )[0]
        stp["next_n_sum"] = sum(stp["next_n"])
        stp["next_n_value"] = float(self.symbol.last_value) * (
            1 + stp["next_n_sum"] / 100
        )

        self.value = stp
        return self.value


class ATR(Indicator):
    """
    Average True Rate
    """

    slug = "atr"
    template = "base/indicators/_atr.html"
    js_slug = "atr"
    js_sorting = "base/indicators/_atr.js"

    def __init__(
        self,
        symbol,
        periods=settings.ATR,
    ):
        self.symbol = symbol
        self.periods = periods

    def get_data(self):
        return list(
            reversed(
                self.symbol.klines.all()
                .order_by("-time_close")[: self.periods + 1]
                .values(
                    "price_high",
                    "price_low",
                    "price_close",
                )
            )
        )

    def calculate(self):
        atr = {"params": self.periods}
        klines = self.get_data()
        trs = []
        for i in range(1, self.periods + 1):
            tr = max(
                klines[i]["price_high"], klines[i - 1]["price_close"]
            ) - min(klines[i]["price_low"], klines[i - 1]["price_close"])
            trs.append(tr)
        atrs = [sum(trs) / self.periods]
        # No need of updating
        # for i in range(1, self.periods + 1):
        #     atrs.append((atrs[i - 1] * (i - 1) + trs[i]) / i)
        # atr["atrs"] = atrs
        atr["trs"] = [float(tr) for tr in trs]
        atr["current"] = float(atrs[-1])

        self.value = atr
        return self.value


class DC(Indicator):
    """
    Donchian Channel
    """

    slug = "dc"
    template = "base/indicators/_dc.html"
    js_slug = "dc"
    js_sorting = "base/indicators/_dc.js"

    def __init__(
        self,
        symbol,
        periods=settings.DC_PERIODS,
        amount=settings.DC_AMOUNT,
    ):
        self.symbol = symbol
        self.periods = periods
        self.amount = amount

    def get_data(self):
        return list(
            reversed(
                self.symbol.klines.all()
                .order_by("-time_close")[: self.periods + self.amount - 1]
                .values(
                    "price_high",
                    "price_low",
                )
            )
        )

    def calculate(self):
        dc = {
            "params": {
                "periods": self.periods,
                "amount": self.amount,
            }
        }
        klines = self.get_data()
        upper = [
            max([k["price_high"] for k in klines[i : self.periods + i]])
            for i in range(0, self.amount)
        ]
        lower = [
            min([k["price_low"] for k in klines[i : self.periods + i]])
            for i in range(0, self.amount)
        ]
        dc["upper"] = [float(b) for b in upper]
        dc["lower"] = [float(b) for b in lower]
        dc["upper_break"] = dc["upper"][-1] > dc["upper"][-2]
        dc["lower_break"] = dc["lower"][-1] < dc["lower"][-2]

        self.value = dc
        return self.value
