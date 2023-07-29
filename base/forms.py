from django import forms

from .models import TraderoBot, User


class TraderoBotForm(forms.ModelForm):
    class Meta:
        model = TraderoBot
        fields = [
            "name",
            "strategy",
            "strategy_params",
            "symbol",
            "fund_quote_asset",
            "fund_quote_asset_initial",
            "fund_base_asset",
            "is_jumpy",
            "should_reinvest",
            "should_stop",
        ]


class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = [
            "first_name",
            "last_name",
            "email",
            "api_key",
            "api_secret",
        ]
