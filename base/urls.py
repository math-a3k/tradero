# urls.py
from django.contrib.auth import views as auth_views
from django.urls import include, path
from django_ratelimit.decorators import ratelimit

from base import views as base_views


def rate_tuples(group, request):
    if request.user.is_authenticated:
        return (100, 60)
    return (12, 60)


def default_rate_limit(v):
    return ratelimit(key="ip", rate=rate_tuples)(v)


urlpatterns = [
    path("", base_views.HomeView.as_view(), name="home"),
    path(
        "Instruções",
        default_rate_limit(base_views.InstrucoesView.as_view()),
        name="instrucoes",
    ),
    path(
        "Usuário",
        default_rate_limit(base_views.UsersDetailView.as_view()),
        name="users-detail",
    ),
    path(
        "Usuário/Entrar",
        default_rate_limit(
            auth_views.LoginView.as_view(template_name="base/login.html")
        ),
        name="login",
    ),
    path(
        "Usuário/Sair",
        default_rate_limit(auth_views.LogoutView.as_view()),
        name="logout",
    ),
    path(
        "Usuário/Atualizar",
        default_rate_limit(base_views.UsersUpdateView.as_view()),
        name="users-update",
    ),
    path(
        "Botzinhos",
        default_rate_limit(base_views.BotzinhosView.as_view()),
        name="botzinhos-list",
    ),
    path(
        "Botzinhos/Novo",
        default_rate_limit(base_views.BotzinhosCreateView.as_view()),
        name="botzinhos-create",
    ),
    path(
        "Botzinhos/<pk>/Atualizar",
        default_rate_limit(base_views.BotzinhosUpdateView.as_view()),
        name="botzinhos-update",
    ),
    path(
        "Botzinhos/<pk>/Bitácula",
        default_rate_limit(base_views.BotzinhosLogsView.as_view()),
        name="botzinhos-logs",
    ),
    path(
        "Botzinhos/<pk>",
        default_rate_limit(base_views.BotzinhosDetailView.as_view()),
        name="botzinhos-detail",
    ),
    path(
        "Botzinhos/<pk>/Acção/<path:action>",
        default_rate_limit(base_views.BotzinhosActionView.as_view()),
        name="botzinhos-action",
    ),
    path(
        "Botzinhos/Grupo/Novo",
        default_rate_limit(base_views.BotzinhosGroupCreateView.as_view()),
        name="botzinhos-group-create",
    ),
    path(
        "Botzinhos/Grupo/<pk>",
        default_rate_limit(base_views.BotzinhosGroupDetailView.as_view()),
        name="botzinhos-group-detail",
    ),
    path(
        "Botzinhos/Grupo/<pk>/Atualizar",
        default_rate_limit(base_views.BotzinhosGroupUpdateView.as_view()),
        name="botzinhos-group-update",
    ),
    path(
        "Botzinhos/Grupo/<pk>/Acção/<path:action>",
        default_rate_limit(base_views.BotzinhosGroupActionView.as_view()),
        name="botzinhos-group-action",
    ),
    #
    path("api/v1/", include(("base.api.v1.urls", "base"), namespace="v1")),
]
