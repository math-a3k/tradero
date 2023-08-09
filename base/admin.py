from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.forms import UserChangeForm
from django_ai.supervised_learning.admin import (
    HGBTreeRegressorAdmin,
    OCSVCAdmin,
)

from .models import (
    DecisionTreeRegressor,
    Kline,
    OutliersSVC,
    Symbol,
    TradeHistory,
    TraderoBot,
    TraderoBotGroup,
    TraderoBotLog,
    TrainingData,
    User,
    WSClient,
)


class TraderoUserChangeForm(UserChangeForm):
    class Meta(UserChangeForm.Meta):
        model = User


@admin.register(User)
class TraderoUserAdmin(UserAdmin):
    form = TraderoUserChangeForm
    list_display = (
        "username",
        "email",
        "first_name",
        "last_name",
        "is_active",
        "trading_active",
        "date_joined",
        "is_staff",
        "get_bots",
    )
    list_display_links = ("username",)
    list_filter = ("trading_active",) + UserAdmin.list_filter
    fieldsets = (
        (
            None,
            {
                "fields": (
                    "trading_active",
                    "api_key",
                    "api_secret",
                    "checkpoint",
                )
            },
        ),
    ) + UserAdmin.fieldsets

    def get_bots(self, obj):
        return obj.bots.count()

    get_bots.short_description = "# Bots"


@admin.register(Symbol)
class SymbolAdmin(admin.ModelAdmin):
    """
    Admin View for Symbol
    """

    list_display = (
        "symbol",
        "status",
        "model_score",
        "volume_quote_asset",
        "last_variation_24h",
        "last_value_time",
        "last_value",
        "last_variation",
        "prediction_time_interval",
    )
    list_filter = (
        "status",
        "is_enabled",
    )
    search_fields = ("symbol",)


@admin.register(Kline)
class KlineAdmin(admin.ModelAdmin):
    """
    Admin View for Kline
    """

    list_display = (
        "symbol",
        "time_open",
        "time_interval",
        "price_open",
        "price_close",
        "variation",
    )
    list_filter = ("symbol",)
    search_fields = ("symbol__symbol",)


@admin.register(TrainingData)
class TrainingDataAdmin(admin.ModelAdmin):
    """
    Admin View for Training Data
    """

    list_display = (
        "symbol",
        "time",
        "time_interval",
        "variation",
        "get_window",
    )
    list_filter = ("time", "symbol")
    search_fields = ("symbol__symbol",)

    def get_window(self, obj):  # pragma: no cover
        return obj.WINDOW

    get_window.short_description = "Window"


@admin.register(DecisionTreeRegressor)
class DecisionTreeAdmin(HGBTreeRegressorAdmin):
    dt_fieldsets = (
        "Symbol",
        {
            "fields": (("symbol",),),
        },
    )

    fieldsets = (dt_fieldsets,) + HGBTreeRegressorAdmin.fieldsets


@admin.register(OutliersSVC)
class OutliersSVCAdmin(OCSVCAdmin):
    dt_fieldsets = (
        "Symbol",
        {
            "fields": (("symbol", "window"),),
        },
    )

    fieldsets = (dt_fieldsets,) + OCSVCAdmin.fieldsets


class IsOpenListFilter(admin.SimpleListFilter):
    # Human-readable title which will be displayed in the
    # right admin sidebar just above the filter options.
    title = "is open"

    # Parameter for the filter that will be used in the URL query.
    parameter_name = "is_open"

    def lookups(self, request, model_admin):
        """
        Returns a list of tuples. The first element in each
        tuple is the coded value for the option that will
        appear in the URL query. The second element is the
        human-readable name for the option that will appear
        in the right sidebar.
        """
        return [
            ("true", "Yes"),
            ("false", "No"),
        ]

    def queryset(self, request, queryset):
        """
        Returns the filtered queryset based on the value
        provided in the query string and retrievable via
        `self.value()`.
        """
        # Compare the requested value (either '80s' or '90s')
        # to decide how to filter the queryset.
        if self.value() == "true":
            return queryset.filter(time_disconnect__isnull=True)
        if self.value() == "false":
            return queryset.filter(time_disconnect__isnull=False)


@admin.register(WSClient)
class WSClientAdmin(admin.ModelAdmin):
    """
    Admin View for WS Client
    """

    list_display = (
        "channel_group",
        "channel_name",
        "get_is_open",
        "time_connect",
        "time_disconnect",
    )
    list_filter = ("channel_group", "time_connect", IsOpenListFilter)
    search_fields = ("channel_name",)

    def get_is_open(self, obj):
        return obj.is_open

    get_is_open.short_description = "Is Open?"


class TraderoBotInline(admin.StackedInline):
    model = TraderoBot
    min_num = 1
    max_num = 20
    extra = 0
    ordering = ["-timestamp_start"]
    readonly_fields = [
        "receipt_buying",
        "receipt_selling",
    ]


@admin.register(TraderoBotGroup)
class TraderoBotGroupAdmin(admin.ModelAdmin):
    """
    Admin View for TraderoBotGroup
    """

    list_display = (
        "id",
        "user",
        "name",
    )
    list_filter = ("user",)
    inlines = [TraderoBotInline]


class TradeHistoryInline(admin.StackedInline):
    model = TradeHistory
    min_num = 3
    max_num = 20
    extra = 0
    ordering = ["-timestamp_start"]
    readonly_fields = [
        "receipt_buying",
        "receipt_selling",
        "user",
        "symbol",
        "timestamp_start",
        "timestamp_buying",
        "timestamp_selling",
    ]


@admin.register(TraderoBot)
class TraderoBotAdmin(admin.ModelAdmin):
    """
    Admin View for TraderoBot
    """

    list_display = (
        "id",
        "user",
        "group",
        "status",
        "symbol",
        "should_reinvest",
        "should_stop",
        "is_dummy",
        "strategy",
        "strategy_params",
        "fund_base_asset",
        "fund_quote_asset",
        "fund_quote_asset_initial",
        "get_last_log_message",
    )
    list_filter = ("status",)
    inlines = [TradeHistoryInline]
    readonly_fields = [
        "receipt_buying",
        "receipt_selling",
        "others",
    ]

    def get_last_log_message(self, obj):
        return obj.get_last_log_message()

    get_last_log_message.short_description = "Last Log Message"


@admin.register(TraderoBotLog)
class TraderoBotLogAdmin(admin.ModelAdmin):
    """
    Admin View for TraderoBot
    """

    list_display = (
        "id",
        "bot",
        "is_dummy",
        "get_action_display",
        "message",
    )
    list_filter = ("action",)

    def get_action_display(self, obj):
        return obj.get_action_display()

    get_action_display.short_description = "Action"


@admin.register(TradeHistory)
class TradeHistoryAdmin(admin.ModelAdmin):
    """
    Admin View for TraderoBot
    """

    list_display = (
        "id",
        "user",
        "bot",
        "is_dummy",
        "is_complete",
        "variation",
        "variation_quote_asset",
        "duration_total",
    )
    list_filter = (
        "user",
        "bot",
    )
