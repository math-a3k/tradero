from decimal import Decimal

from django import forms
from django.core.exceptions import ValidationError
from django.db.models import Model
from django.urls import reverse
from django.utils.safestring import mark_safe

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
    bots_quantity = forms.IntegerField(
        label="# of Botzinhos",
        initial=1,
        required=False,
        help_text="Create several Botzinhos using this as a template",
    )

    class Meta:
        model = TraderoBot
        fields = [
            "bots_quantity",
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

    def __init__(self, *args, for_group=False, for_group_edit=False, **kwargs):
        user = kwargs.pop("user", None)
        for_edit = kwargs.pop("for_edit", None)
        super().__init__(*args, **kwargs)
        if for_edit:
            self.fields.pop("bots_quantity")
        if for_group:
            self.fields.pop("group")
            self.fields.pop("bots_quantity")
            if for_group_edit:
                self.fields.pop("symbol")
                for field in self.fields:
                    self.fields[field].required = False
        else:
            self.fields["group"].queryset = TraderoBotGroup.objects.filter(
                user=user
            ).order_by("name")


class TraderoBotGroupForm(forms.ModelForm):
    prefix_bot_data = "bot_"

    add_edit_bots = forms.BooleanField(
        label="Add / Edit Botzinhos", required=False
    )
    distribute_bots = forms.BooleanField(
        label="Distribute Botzinhos over Symbols", required=False
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
        bot_form_fields = self.get_bot_form().fields
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
            and data_dict[k] not in self.fields[k].empty_values
        }

    def get_bot_form(self):
        return TraderoBotForm(for_group=True)


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
                self.fields[field].widget.choices = [
                    (None, "(Selecionar)"),
                    (True, "Sim"),
                    (False, "Não"),
                ]
                self.fields[field].required = False

    def clean(self):
        cleaned_data = super(TraderoBotGroupForm, self).clean()
        if cleaned_data["add_edit_bots"]:
            bot_data = self.get_bot_data(cleaned_data)
            bots = self.instance.bots.all()
            for bot in bots:
                for field in bot_data:
                    setattr(bot, field, bot_data[field])
                try:
                    bot.full_clean()
                except ValidationError as ve:
                    self.add_error(
                        None,
                        "The following fields produce erros with at least one "
                        "bot ({bot.pk}) of the group:",
                    )
                    for field, message in ve.message_dict.items():
                        self.add_error(
                            f"{self.prefix_bot_data}{field}", message
                        )

                    break
        return cleaned_data

    def get_bot_form(self):
        return TraderoBotForm(for_group=True, for_group_edit=True)


class BotsModelMultipleChoiceField(forms.ModelMultipleChoiceField):
    def label_from_instance(self, obj):  # pragma: no cover
        if obj.status == obj.Status.SELLING:
            status_class = "text-danger"
        elif obj.status == obj.Status.BUYING:
            status_class = "text-success"
        else:
            status_class = "text-muted"
        return mark_safe(
            f"<span class='text-muted'>#{obj.id:03d}</span> "
            f"<a href='{ reverse('base:botzinhos-detail', args=[obj.pk]) }' "
            f"class='name-value'>{ obj.name }</a>: { obj.symbol.base_asset }"
            f"/{ obj.symbol.quote_asset } | <span class='{ status_class }'>"
            f"{ obj.get_status_display() }</span>"
        )


class TraderoBotGroupMoveForm(forms.ModelForm):
    to_group = forms.ModelChoiceField(
        label="To Group", queryset=None, empty_label="(Selecionar)"
    )
    bots = BotsModelMultipleChoiceField(
        queryset=None,
        widget=forms.CheckboxSelectMultiple,
    )

    class Meta:
        model = TraderoBotGroup
        fields = [
            "to_group",
            "bots",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["to_group"].queryset = TraderoBotGroup.objects.filter(
            user=self.instance.user
        ).order_by("name")
        self.fields["bots"].queryset = TraderoBot.objects.filter(
            group=self.instance
        ).order_by("name")


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
