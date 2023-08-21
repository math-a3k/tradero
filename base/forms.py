from decimal import Decimal

from django import forms
from django.db.models import Model

from .models import Symbol, TraderoBot, TraderoBotGroup, User
from .strategies import get_strategies


class TraderoBotForm(forms.ModelForm):
    group = forms.ModelChoiceField(queryset=None, empty_label="(Selecionar)")
    symbol = forms.ModelChoiceField(
        queryset=Symbol.objects.available(), empty_label="(Selecionar)"
    )
    strategy = forms.ChoiceField(
        choices=[(None, "(Selecionar)")]
        + [(k, v.__name__) for k, v in get_strategies().items()]
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

    def __init__(self, *args, for_group=False, **kwargs):
        user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)
        if for_group:
            self.fields.pop("group")
        else:
            self.fields["group"].queryset = TraderoBotGroup.objects.filter(
                user=user
            )


class TraderoBotGroupForm(forms.ModelForm):
    prefix_bot_data = "bot_"

    add_edit_bots = forms.BooleanField(
        label="Add / Edit Botzinhos", required=False
    )
    bots_quantity = forms.IntegerField(
        label="# of Botzinhos", initial=10, required=False
    )

    class Meta:
        model = TraderoBotGroup
        fields = [
            "name",
            "add_edit_bots",
            "bots_quantity",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["add_edit_bots"].label = "Add Botzinhos"
        bot_form_fields = TraderoBotForm(for_group=True).fields
        for field in bot_form_fields:
            new_field = f"{self.prefix_bot_data}{field}"
            self.fields[new_field] = bot_form_fields[field]
            self.fields[new_field].required = False
            self.fields[new_field].initial = None

    def clean(self):
        cleaned_data = super().clean()
        if cleaned_data["add_edit_bots"]:
            bot_data = self.get_bot_data(cleaned_data)
            for k, v in bot_data.items():
                if isinstance(v, Model):
                    bot_data[k] = v.pk
                elif isinstance(v, Decimal):
                    bot_data[k] = str(v)
            bot_form = TraderoBotForm(bot_data, for_group=True)
            if not bot_form.is_valid():
                for field, message in bot_form.errors.items():
                    self.add_error(f"{self.prefix_bot_data}{field}", message)
        return cleaned_data

    def get_bot_data(self, data_dict):
        return {
            k[len(self.prefix_bot_data) :]: v
            for k, v in data_dict.items()
            if k.startswith(self.prefix_bot_data)
        }


class TraderoBotGroupEditForm(TraderoBotGroupForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["add_edit_bots"].label = "Edit Botzinhos"
        self.fields["bots_quantity"].initial = self.instance.bots.all().count()
        self.fields["bots_quantity"].widget.attrs["readonly"] = True
        for field in [
            f for f in self.fields if f.startswith(self.prefix_bot_data)
        ]:
            if isinstance(self.fields[field], forms.fields.BooleanField):
                self.fields[field] = forms.fields.NullBooleanField(
                    label=self.fields[field].label
                )

    def clean(self):
        pass


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


class JumpingForm(forms.Form):
    to_symbol = forms.ModelChoiceField(
        queryset=Symbol.objects.available(), empty_label="(Selecionar)"
    )
