from django import forms

from .models import Symbol, TraderoBot, TraderoBotGroup, User
from .strategies import get_strategies


class TraderoBotForm(forms.ModelForm):
    group = forms.ModelChoiceField(queryset=None, empty_label="(Selecionar)")
    symbol = forms.ModelChoiceField(
        queryset=Symbol.objects.available(), empty_label="(Selecionar)"
    )
    strategy = forms.ChoiceField(
        choices=[(k, v.__name__) for k, v in get_strategies().items()]
    )

    class Meta:
        model = TraderoBot
        fields = [
            "group",
            "name",
            "is_dummy",
            "strategy",
            "strategy_params",
            "symbol",
            "fund_quote_asset",
            "fund_quote_asset_initial",
            "fund_base_asset",
            "is_jumpy",
            "jumpy_whitelist",
            "jumpy_blacklist",
            "should_reinvest",
            "should_stop",
        ]

    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user")
        super().__init__(*args, **kwargs)
        self.fields["group"].queryset = TraderoBotGroup.objects.filter(
            user=user
        )


class TraderoBotGroupForm(forms.ModelForm):
    class Meta:
        model = TraderoBotGroup
        fields = ["name"]


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
