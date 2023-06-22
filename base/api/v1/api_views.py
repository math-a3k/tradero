from rest_framework import viewsets

from base.models import Symbol

from .serializers import SymbolSerializer


class SymbolViewSet(viewsets.ReadOnlyModelViewSet):
    """
    A simple ViewSet for listing or retrieving Symbols.
    """

    queryset = Symbol.objects.available()
    serializer_class = SymbolSerializer
    lookup_field = "symbol"
