from django.urls import path
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularRedocView,
    SpectacularSwaggerView,
)
from rest_framework import routers

from .api_views import SymbolViewSet

router = routers.DefaultRouter()
router.register(r"symbols", SymbolViewSet)
urlpatterns = router.urls

urlpatterns = urlpatterns + [
    # "symbols/ws" is handled in base.routing
    path("schema/", SpectacularAPIView.as_view(), name="schema"),
    path(
        "schema/swagger-ui/",
        SpectacularSwaggerView.as_view(url_name="base:v1:schema"),
        name="swagger-ui",
    ),
    path(
        "schema/redoc/",
        SpectacularRedocView.as_view(url_name="base:v1:schema"),
        name="redoc",
    ),
]
