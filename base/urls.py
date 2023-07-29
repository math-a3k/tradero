# urls.py
from django.contrib.auth import views as auth_views
from django.urls import include, path

from base import views as base_views

urlpatterns = [
    path("", base_views.HomeView.as_view(), name="home"),
    path(
        "Instruções",
        base_views.InstrucoesView.as_view(),
        name="instrucoes",
    ),
    path(
        "Usuário",
        base_views.UsersDetailView.as_view(),
        name="users-detail",
    ),
    path(
        "Usuário/Entrar",
        auth_views.LoginView.as_view(template_name="base/login.html"),
        name="login",
    ),
    path(
        "Usuário/Sair",
        auth_views.LogoutView.as_view(),
        name="logout",
    ),
    path(
        "Usuário/Atualizar",
        base_views.UsersUpdateView.as_view(),
        name="users-update",
    ),
    path(
        "Botzinhos",
        base_views.BotzinhosView.as_view(),
        name="botzinhos-list",
    ),
    path(
        "Botzinhos/Novo",
        base_views.BotzinhosCreateView.as_view(),
        name="botzinhos-create",
    ),
    path(
        "Botzinhos/<pk>/Atualizar",
        base_views.BotzinhosUpdateView.as_view(),
        name="botzinhos-update",
    ),
    path(
        "Botzinhos/<pk>",
        base_views.BotzinhosDetailView.as_view(),
        name="botzinhos-detail",
    ),
    path(
        "Botzinhos/<pk>/Acção/<path:action>",
        base_views.BotzinhosActionsView.as_view(),
        name="botzinhos-actions",
    ),
    #
    path("api/v1/", include(("base.api.v1.urls", "base"), namespace="v1")),
]
