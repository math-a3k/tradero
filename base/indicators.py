import inspect
import sys

import pandas as pd
from django.conf import settings
from statsmodels.tsa.api import Holt


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

        ts_a = tds.rolling(self.a).mean().dropna().to_list()
        ts_b = tds.rolling(self.b).mean().dropna().to_list()
        ts_a_e = Holt(ts_a).fit().fittedvalues
        ts_b_e = Holt(ts_b).fit().fittedvalues

        av_len = min(len(ts_a_e), len(ts_b_e))
        macd_line = pd.Series(ts_a_e[-av_len:] - ts_b_e[-av_len:])
        macd_signal = macd_line.rolling(self.c).mean().dropna()
        macd_line = macd_line[(self.c - 1) :].to_list()
        signal_diff = (macd_line - macd_signal).to_list()
        current_good = (macd_line[-1] > 0) and (signal_diff[-1] > 0)

        self.value = {
            "params": [
                self.a,
                self.b,
                self.c,
            ],
            # "ema_a": ts_a_e.tolist(),   # not needed
            "macd_line": macd_line[-self.c :],
            "signal_diff": signal_diff[-self.c :],
            "current_good": bool(current_good),
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
