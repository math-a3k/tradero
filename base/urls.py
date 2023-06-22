# urls.py
from django.urls import include, path

from .views import HomeView, InstrucoesView

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path(
        "Instruções",
        InstrucoesView.as_view(),
        name="instrucoes",
    ),
    #
    path("api/v1/", include(("base.api.v1.urls", "base"), namespace="v1")),
]
