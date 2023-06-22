from django.conf import settings
from django.views.generic import TemplateView

from .models import Symbol


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
