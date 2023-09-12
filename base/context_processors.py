from django.conf import settings


def add_settings(request):
    return {
        "SOURCE_URL": settings.SOURCE_URL,
        "GIT_VERSION": settings.GIT_VERSION,
    }
