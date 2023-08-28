import logging
from decimal import Decimal

import requests
from binance.spot import Spot
from django.conf import settings

from .utils import get_commission

logger = logging.getLogger(__name__)


class TraderoClient(Spot):  # pragma: no cover
    """
    Tradero's Client - a Spot overhauling...

    "tradero_" methods are Tradero's specific
    """

    def __init__(self, user, *args, **kwargs):
        super().__init__(
            *args,
            base_url=settings.EXCHANGE_API_URL,
            api_key=user.api_key,
            api_secret=user.api_secret,
            **kwargs,
        )
        self.session.mount(
            "https://", requests.adapters.HTTPAdapter(pool_maxsize=36)
        )

    def tradero_market_order(
        self,
        side,
        symbol,
        amount,
        dummy=False,
    ):
        """
        SELL Response:
        {
            "symbol": "BETABUSD",
            "orderId": 204665291,
            "orderListId": -1,
            "clientOrderId": "cOcAYDp7dTEu9zd4sPoUs4",
            "transactTime": 1688436028709,
            "price": "0.00000000",
            "origQty": "120.00000000",
            "executedQty": "120.00000000",
            "cummulativeQuoteQty": "10.02360000",
            "status": "FILLED",
            "timeInForce": "GTC",
            "type": "MARKET",
            "side": "SELL",
            "workingTime": 1688436028709,
            "fills":
            [
                {
                    "price": "0.08353000",
                    "qty": "120.00000000",
                    "commission": "0.01002360",
                    "commissionAsset": "BUSD",
                    "tradeId": 10478698
                }
            ],
            "selfTradePreventionMode": "NONE"
        }
        BUY response:
        {
            "symbol": "ONTBUSD",
            "orderId": 227918297,
            "orderListId": -1,
            "clientOrderId": "LUniYVTucWMYAeKc9QpSRt",
            "transactTime": 1689381359633,
                            1693175666.249509
            "price": "0.00000000",
            "origQty": "72.00000000",
            "executedQty": "72.00000000",
            "cummulativeQuoteQty": "14.91840000",
            "status": "FILLED",
            "timeInForce": "GTC",
            "type": "MARKET",
            "side": "BUY",
            "workingTime": 1689381359633,
            "fills":
            [
                {
                    "price": "0.20720000",
                    "qty": "72.00000000",
                    "commission": "0.07200000",
                    "commissionAsset": "ONT",
                    "tradeId": 6242788
                }
            ],
            "selfTradePreventionMode": "NONE"
        }
        """
        try:
            market_price = None
            #
            if side == "BUY":
                # Binance's Client Market's orders are expressed in BASE asset
                market_price = Decimal(
                    self.ticker_price(symbol.symbol)["price"]
                )
                amount = Decimal(amount) / market_price
            #
            step = Decimal(symbol.info["filters"][1]["stepSize"])
            qty = (Decimal(amount) // step) * step
            #
            if dummy:
                # In dummy mode, price is the regular market price
                market_price = market_price or Decimal(
                    self.ticker_price(symbol.symbol)["price"]
                )
                exc_qty = qty
                cum_exc_qty = qty * market_price
                if side == "BUY":
                    commission = exc_qty * settings.EXCHANGE_FEE
                    quantity = exc_qty - commission
                    price = cum_exc_qty / quantity  # Net price
                else:
                    commission = cum_exc_qty * settings.EXCHANGE_FEE
                    quantity = cum_exc_qty - commission
                    price = quantity / exc_qty  # Net price
                #
                quantity = quantity.quantize(Decimal("." + "0" * 8))
                price = price.quantize(Decimal("." + "0" * 8))
                #
                return (
                    True,  # success
                    {  # receipt
                        "orderId": "DUMMY",
                        "side": side,
                        "executedQty": str(exc_qty),
                        "cummulativeQuoteQty": str(cum_exc_qty),
                        "fills": [
                            {
                                "commission": str(commission),
                                "commissionAsset": (
                                    symbol.base_asset
                                    if side == "BUY"
                                    else symbol.quote_asset
                                ),
                            },
                        ],
                    },
                    None,  # message
                )
            else:
                order_params = {
                    "symbol": symbol.symbol,
                    "side": side,
                    "type": "MARKET",
                    "quantity": qty,
                }
                if settings.TRADERO_DEBUG:
                    logger.debug(f"Symbol Info: {symbol.info}")
                    logger.debug(f"Order params: {order_params}")
                    asset = (
                        symbol.quote_asset
                        if side == "BUY"
                        else symbol.base_asset
                    )
                    logger.debug(f"User asset: {self.user_asset(asset=asset)}")
                exc = None
                for i in range(3):
                    try:
                        receipt = self.new_order(**order_params)
                        if settings.TRADERO_DEBUG:
                            logger.debug(f"Binance client result: f{receipt}")
                        #
                        return (
                            True,  # success
                            receipt,
                            None,  # message
                        )
                    except Exception as e:
                        if settings.TRADERO_DEBUG:
                            logger.debug(f"Attempt #{i}: {str(e)}")
                        exc = e
                raise exc
        except Exception as e:
            e_str = str(e)
            if "nginx" in e_str:
                e_str = e_str[1 : e_str.find("{")]
            return (False, None, e_str)

    def tradero_sell(
        self,
        symbol,
        amount,
        dummy=False,
    ):
        return self.tradero_market_order("SELL", symbol, amount, dummy=dummy)

    def tradero_buy(
        self,
        symbol,
        amount,
        dummy=False,
    ):
        return self.tradero_market_order("BUY", symbol, amount, dummy=dummy)

    def parse_receipt(self, receipt):
        cum_exc_qty = Decimal(receipt["cummulativeQuoteQty"])
        exc_qty = Decimal(receipt["executedQty"])
        commission, commission_asset = get_commission(receipt)
        commission_e = commission if commission_asset != "BNB" else 0
        if receipt["side"] == "BUY":
            quantity_exec = cum_exc_qty
            quantity_rec = exc_qty - commission_e
            price = cum_exc_qty / quantity_rec  # Net price
        else:
            quantity_exec = exc_qty
            quantity_rec = cum_exc_qty - commission_e  # Net price
            price = quantity_rec / exc_qty
        r = {
            "quantity_exec": quantity_exec.quantize(Decimal("." + "0" * 8)),
            "quantity_rec": quantity_rec.quantize(Decimal("." + "0" * 8)),
            # Net price (except when comm in BNB)
            "price_net": price.quantize(Decimal("." + "0" * 8)),
            "commission": commission,
            "commission_asset": commission_asset,
        }
        return r
