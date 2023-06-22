from binance.lib.utils import check_required_parameters
from binance.spot import Spot
from django.utils import timezone


class TraderoClient(Spot):  # pragma: no cover
    """
    Tradero's Client - a Spot overhauling...

    "t_" methods are Tradero's specific
    """

    def __init__(self, user, *args, **kwargs):
        super().__init__(
            *args,
            api_key=user.api_key,
            api_secret=user.api_secret,
            **kwargs,
        )

    def get_quote(
        self,
        from_asset,
        to_asset,
        from_amount,
        to_amount=None,
        wallet_type="SPOT",
        valid_time=None,
        recv_window=None,
        timestamp=None,
        **kwargs,
    ):
        check_required_parameters(
            [
                [from_asset, "fromAsset"],
                [to_asset, "toAsset"],
                [from_amount, "fromAmount"],  # TODO: see how to check EITHER
                # [timestamp, "timestamp"],
            ]
        )

        url_path = "/sapi/v1/convert/getQuote"
        # TODO: See if nothing happens if keys have null values and complete
        # the parameters in the payload
        payload = {
            "fromAsset": from_asset,
            "toAsset": to_asset,
            "fromAmount": from_amount,
            "timestamp": timestamp or timezone.now().timestamp() * 1000,
            **kwargs,
        }
        return self.sign_request("POST", url_path, payload)

    def accept_quote(
        self,
        quote_id,
        recv_window=None,
        timestamp=None,
        **kwargs,
    ):
        check_required_parameters(
            [
                [quote_id, "quoteId"],
                # [timestamp, "timestamp"],
            ]
        )

        url_path = "/sapi/v1/convert/acceptQuote"
        # TODO: See if nothing happens if keys have null values and complete
        # the parameters in the payload
        payload = {
            "quoteId": quote_id,
            "timestamp": timestamp or timezone.now().timestamp() * 1000,
            **kwargs,
        }
        return self.sign_request("POST", url_path, payload)

    def t_convert_assets(
        self,
        from_asset,
        to_asset,
        from_amount,
    ):
        """
        TODO: Add error handling
        """
        quote_id = self.get_quote(from_asset, to_asset, from_amount)["quoteId"]
        return self.accept_quote(quote_id)
