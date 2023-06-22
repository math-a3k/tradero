from django.conf import settings
from django.core.cache import cache
from rest_framework import serializers

from base.models import Symbol


class SymbolSerializer(serializers.ModelSerializer):
    class Meta:
        model = Symbol
        exclude = ["id", "is_enabled"]

    def to_representation(self, instance, set_cache=False):
        """
        Caches the dict representation for rendenering
        """
        cache_key = f"{instance.symbol}_dict"
        d = cache.get(cache_key)
        if not d or set_cache:  # pragma: no cover
            d = super().to_representation(instance)
            cache.set(cache_key, d, settings.TIME_INTERVAL * 60 - 1)
        return d
