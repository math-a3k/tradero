import inspect

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.paginator import Paginator
from django.http import Http404
from django.shortcuts import get_object_or_404
from django.views.generic import (
    CreateView,
    DetailView,
    ListView,
    RedirectView,
    TemplateView,
    UpdateView,
)

from .forms import TraderoBotForm, TraderoBotGroupForm, UserForm
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
        trades = self.object.trades.all().order_by("-id")
        summary = TradeHistory.summary_for_object(self.object)
        #
        paginator = Paginator(trades, 5)
        page_number = self.request.GET.get("page", 1)
        page_obj = paginator.get_page(page_number)
        #
        context["page_obj"] = page_obj
        context["time_interval"] = settings.TIME_INTERVAL_BOTS
        context["summary"] = summary
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
        paginator = Paginator(trades, 5)
        page_number = self.request.GET.get("page", 1)
        page_obj = paginator.get_page(page_number)
        #
        context["page_obj"] = page_obj
        context["time_interval"] = settings.TIME_INTERVAL_BOTS
        context["summary"] = summary
        return context


class BotzinhosCreateView(LoginRequiredMixin, CreateView):
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
        form.instance.user = self.request.user
        return super().form_valid(form)

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs.update(user=self.request.user)
        return kwargs


class BotzinhosUpdateView(OwnerMixin, LoginRequiredMixin, UpdateView):
    model = TraderoBot
    form_class = TraderoBotForm
    template_name = "base/botzinhos_form.html"

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs.update(user=self.request.user)
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

    def run_action(self, action):
        try:
            action_method = getattr(self.object, action)
            action_method()
            messages.success(
                self.request,
                f"SUCCESS at {action.upper()} for {self.object.pk}:"
                f"{self.object.name}",
            )
        except Exception as e:
            msg = e.args[0]
            frm = inspect.trace()[-1]
            mod = inspect.getmodule(frm[0])
            modname = mod.__name__ if mod else frm[1]
            messages.error(
                self.request,
                f"ERROR at {action.upper()} for {self.object.pk}:"
                f"{self.object.name}",
                f"[{modname}] {str(msg)}",
            )

    def get_redirect_url(self, *args, **kwargs):
        if kwargs["action"] not in self.ACTIONS:
            raise Http404("Action not Found")
        self.pk = kwargs["pk"]
        self.get_object()
        self.run_action(kwargs["action"])
        return self.request.META.get("HTTP_REFERER", "/")


class BotzinhosActionView(ActionView):
    """
    Runs Actions on Botzinhos
    """

    ACTIONS = ["buy", "sell", "on", "off", "reset"]

    def get_object(self):
        self.object = get_object_or_404(TraderoBot, pk=self.pk)
        return self.object


class BotzinhosGroupDetailView(OwnerMixin, LoginRequiredMixin, DetailView):
    model = TraderoBotGroup
    template_name = "base/botzinhos_group_detail.html"
    context_object_name = "group"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        bots = TraderoBot.objects.filter(group=self.object)
        trades = TradeHistory.objects.filter(
            bot__group=self.object, is_complete=True
        ).order_by("-id")
        summary = TradeHistory.summary_for_object(self.object)
        #
        paginator = Paginator(trades, 10)
        page_number = self.request.GET.get("page", 1)
        page_obj = paginator.get_page(page_number)
        #
        context["page_obj"] = page_obj
        context["time_interval"] = settings.TIME_INTERVAL_BOTS
        context["bots"] = bots
        context["summary"] = summary
        context["quote_asset"] = settings.QUOTE_ASSET
        return context


class BotzinhosGroupCreateView(LoginRequiredMixin, CreateView):
    model = TraderoBotGroup
    form = TraderoBotGroupForm
    template_name = "base/botzinhos_group_form.html"
    fields = ["name"]

    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)


class BotzinhosGroupUpdateView(OwnerMixin, LoginRequiredMixin, UpdateView):
    model = TraderoBotGroup
    form = TraderoBotGroupForm
    template_name = "base/botzinhos_group_form.html"
    fields = ["name"]


class BotzinhosGroupActionView(ActionView):
    """
    Runs Actions on Botzinhos Group
    """

    ACTIONS = ["on", "off", "liquidate"]

    def get_object(self):
        self.object = get_object_or_404(TraderoBotGroup, pk=self.pk)
        return self.object
