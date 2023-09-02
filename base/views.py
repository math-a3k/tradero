import inspect

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.paginator import Paginator
from django.http import Http404
from django.shortcuts import get_object_or_404
from django.urls import reverse
from django.views.generic import (
    CreateView,
    DetailView,
    FormView,
    ListView,
    RedirectView,
    TemplateView,
    UpdateView,
)

from .forms import (
    JumpingForm,
    TraderoBotForm,
    TraderoBotGroupEditForm,
    TraderoBotGroupForm,
    TraderoBotGroupMoveForm,
    UserForm,
)
from .models import Symbol, TradeHistory, TraderoBot, TraderoBotGroup, User


class HomeView(TemplateView):
    template_name = "base/home.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["symbols"] = Symbol.objects.top_symbols()
        context["time_interval"] = settings.TIME_INTERVAL
        context["ms_threshold"] = settings.MODEL_SCORE_THRESHOLD
        return context


class InstrucoesView(TemplateView):
    template_name = "base/instrucoes.html"


class OwnerMixin:
    def get_object(self):
        obj = super().get_object()
        if (
            obj.user != self.request.user
            and not self.request.user.is_superuser
        ):
            raise Http404("Object not Found")
        return obj


class UsersDetailView(LoginRequiredMixin, DetailView):
    model = User
    template_name = "base/users_detail.html"
    context_object_name = "user"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        trades = self.object.trades.filter(is_complete=True).order_by(
            "-timestamp_selling"
        )
        summary = TradeHistory.summary_for_object(self.object)
        groups = self.object.botgroups.all().order_by("name")
        #
        paginator = Paginator(trades, 10)
        page_number = self.request.GET.get("page", 1)
        page_obj = paginator.get_page(page_number)
        #
        context["page_obj"] = page_obj
        context["page_element"] = "#trades-list"
        context["time_interval"] = settings.TIME_INTERVAL_BOTS
        context["summary"] = summary
        context["groups"] = groups
        return context

    def get_object(self, queryset=None):
        self.object = self.request.user
        return self.object


class UsersUpdateView(LoginRequiredMixin, UpdateView):
    model = User
    form = UserForm
    success_url = "/Usuário"
    template_name = "base/users_form.html"
    fields = [
        "first_name",
        "last_name",
        "email",
        "api_key",
        "api_secret",
    ]

    def get_object(self, queryset=None):
        self.object = self.request.user
        return self.object


class BotzinhosView(LoginRequiredMixin, ListView):
    model = TraderoBot
    template_name = "base/botzinhos.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["groups"] = TraderoBotGroup.objects.filter(
            user=self.request.user
        ).order_by("name")
        context["time_interval"] = settings.TIME_INTERVAL_BOTS
        return context


class BotzinhosDetailView(OwnerMixin, LoginRequiredMixin, DetailView):
    model = TraderoBot
    template_name = "base/botzinhos_detail.html"
    context_object_name = "bot"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        trades = self.object.trades.all().order_by("-id")
        summary = TradeHistory.summary_for_object(self.object)
        #
        paginator = Paginator(trades, 10)
        page_number = self.request.GET.get("page", 1)
        page_obj = paginator.get_page(page_number)
        #
        context["page_obj"] = page_obj
        context["page_element"] = "#trades-list"
        context["time_interval"] = settings.TIME_INTERVAL_BOTS
        context["summary"] = summary
        context["form_jumping"] = JumpingForm()
        return context


class BotzinhosLogsView(OwnerMixin, LoginRequiredMixin, DetailView):
    model = TraderoBot
    template_name = "base/botzinhos_logs.html"
    context_object_name = "bot"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        logs = self.object.logs.all().order_by("-id")
        #
        paginator = Paginator(logs, 30)
        page_number = self.request.GET.get("page", 1)
        page_obj = paginator.get_page(page_number)
        #
        context["page_obj"] = page_obj
        context["time_interval"] = settings.TIME_INTERVAL_BOTS

        return context


class BotzinhosCreateView(LoginRequiredMixin, FormView):
    model = TraderoBot
    form_class = TraderoBotForm
    template_name = "base/botzinhos_form.html"

    def get_initial(self):
        initial = super().get_initial()
        group_pk = self.request.GET.get("group", None)
        if group_pk:
            initial["group"] = get_object_or_404(TraderoBotGroup, pk=group_pk)
        return initial

    def form_valid(self, form):
        bot_kwargs = form.cleaned_data
        bot_kwargs.update({"user": self.request.user})
        bot_name = bot_kwargs.pop("name", None)
        bots_quantity = bot_kwargs.pop("bots_quantity", 1)
        for i in range(bots_quantity):
            if bot_name:
                bot_kwargs.update({"name": f"{bot_name}-{i+1:03d}"})
            bot = TraderoBot(**bot_kwargs)
            bot.save()
        self.group = bot_kwargs.pop("group")
        return super().form_valid(form)

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs.update(user=self.request.user)
        return kwargs

    def get_success_url(self):
        return reverse(
            "base:botzinhos-group-detail", kwargs={"pk": self.group.pk}
        )


class BotzinhosUpdateView(OwnerMixin, LoginRequiredMixin, UpdateView):
    model = TraderoBot
    form_class = TraderoBotForm
    template_name = "base/botzinhos_form.html"

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs.update(user=self.request.user, for_edit=True)
        return kwargs


class ActionView(OwnerMixin, LoginRequiredMixin, RedirectView):
    """
    Runs Action on Objects
    """

    permanent = False
    http_method_names = ["post"]
    #: Available Actions
    ACTIONS = []

    def get_object(self):
        raise NotImplementedError

    def run_action(self):
        try:
            action_method = getattr(self.object, self.action)
            if self.action_params:
                action_method(**self.action_params)
            else:
                action_method()
            messages.success(
                self.request,
                f"SUCCESS at {self.action.upper()}({self.get_params_str()}) "
                f"for [{self.object.pk}] {self.object.name}",
            )
        except Exception as e:
            msg = e.args[0]
            frm = inspect.trace()[-1]
            mod = inspect.getmodule(frm[0])
            modname = mod.__name__ if mod else frm[1]
            messages.error(
                self.request,
                f"ERROR at {self.action.upper()}({self.get_params_str()}) "
                f"for [{self.object.pk}] {self.object.name}: "
                f"[{modname}] {str(msg)}",
            )

    def get_redirect_url(self, *args, **kwargs):
        self.action = kwargs["action"]
        if self.action not in self.ACTIONS:
            raise Http404("Action not Found")
        self.action_params = self.process_params(self.request.POST)
        self.pk = kwargs["pk"]
        self.get_object()
        self.run_action()
        return self.request.META.get("HTTP_REFERER", "/")

    def get_params_str(self):
        if self.action_params:
            return ",".join(
                [f"{k}={v}" for k, v in self.action_params.items()]
            )
        return ""

    def process_params(self, data):
        data = {k: v for k, v in data.items() if k != "csrfmiddlewaretoken"}
        action_params = {}
        action_params_conf = self.ACTIONS[self.action]["params"]
        if action_params_conf:
            for param in action_params_conf:
                if action_params_conf[param]["type"] == "Model":
                    action_params[param] = get_object_or_404(
                        action_params_conf[param]["class"],
                        pk=data[param],
                    )
                # Ignore non-compliant parameters
        return action_params


class BotzinhosActionView(ActionView):
    """
    Runs Actions on Botzinhos
    """

    ACTIONS = {
        "buy": {"params": None},
        "sell": {"params": None},
        "on": {"params": None},
        "off": {"params": None},
        "reset_hard": {"params": None},
        "reset_soft": {"params": None},
        "jump": {"params": {"to_symbol": {"type": "Model", "class": Symbol}}},
    }

    def get_object(self):
        self.object = get_object_or_404(TraderoBot, pk=self.pk)
        return self.object


class BotzinhosGroupDetailView(OwnerMixin, LoginRequiredMixin, DetailView):
    model = TraderoBotGroup
    template_name = "base/botzinhos_group_detail.html"
    context_object_name = "group"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        bots = TraderoBot.objects.filter(group=self.object).order_by("name")
        trades = TradeHistory.objects.filter(
            bot__group=self.object, is_complete=True
        ).order_by("-timestamp_selling")
        summary = TradeHistory.summary_for_object(self.object)
        #
        paginator = Paginator(trades, 10)
        page_number = self.request.GET.get("page", 1)
        page_obj = paginator.get_page(page_number)
        #
        context["page_obj"] = page_obj
        context["page_element"] = "#trades-list"
        context["time_interval"] = settings.TIME_INTERVAL_BOTS
        context["bots"] = bots
        context["summary"] = summary
        context["quote_asset"] = settings.QUOTE_ASSET
        context["form_jumping"] = JumpingForm()
        return context


class BotzinhosGroupCreateView(LoginRequiredMixin, CreateView):
    model = TraderoBotGroup
    form_class = TraderoBotGroupForm
    template_name = "base/botzinhos_group_form.html"

    def form_valid(self, form):
        form.instance.user = self.request.user
        self.object = form.save()
        if form.cleaned_data["add_edit_bots"]:
            bot_kwargs = form.get_bot_data(form.cleaned_data)
            bot_kwargs.update(
                {"user": self.request.user, "group": self.object}
            )
            bot_name = bot_kwargs.get("name", None)
            for i in range(form.cleaned_data["bots_quantity"]):
                if bot_name:
                    bot_kwargs.update({"name": f"{bot_name}-{i+1:03d}"})
                bot = TraderoBot(**bot_kwargs)
                bot.save()
        return super().form_valid(form)


class BotzinhosGroupUpdateView(OwnerMixin, LoginRequiredMixin, UpdateView):
    model = TraderoBotGroup
    form_class = TraderoBotGroupEditForm
    template_name = "base/botzinhos_group_form.html"

    def form_valid(self, form):
        form.instance.user = self.request.user
        self.object = form.save()
        if form.cleaned_data["add_edit_bots"]:
            bot_data = form.get_bot_data(form.cleaned_data)
            bot_name = bot_data.pop("name", None)
            for index, bot in enumerate(self.object.bots.all()):
                if bot_name:
                    bot.name = f"{bot_name}-{index + 1:03d}"
                for field in bot_data:
                    setattr(bot, field, bot_data[field])
                bot.save()
        return super().form_valid(form)


class BotzinhosGroupMoveView(OwnerMixin, LoginRequiredMixin, UpdateView):
    model = TraderoBotGroup
    form_class = TraderoBotGroupMoveForm
    template_name = "base/botzinhos_group_move_form.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["time_interval"] = settings.TIME_INTERVAL_BOTS
        context["group"] = self.object
        return context

    def form_valid(self, form):
        bots = form.cleaned_data["bots"]
        to_group = form.cleaned_data["to_group"]
        TraderoBot.objects.filter(id__in=[b.id for b in bots]).update(
            group=to_group
        )
        self.group = to_group
        return super().form_valid(form)

    def get_success_url(self):
        return reverse(
            "base:botzinhos-group-detail", kwargs={"pk": self.group.pk}
        )


class BotzinhosGroupActionView(ActionView):
    """
    Runs Actions on Botzinhos Group
    """

    ACTIONS = {
        "on": {"params": None},
        "off": {"params": None},
        "liquidate": {"params": None},
        "jump": {"params": {"to_symbol": {"type": "Model", "class": Symbol}}},
        "reset_soft": {"params": None},
    }

    def get_object(self):
        self.object = get_object_or_404(TraderoBotGroup, pk=self.pk)
        return self.object
