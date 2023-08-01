"""
Django settings for tradero project.

Generated by 'django-admin startproject' using Django 4.2.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.2/ref/settings/
"""

from decimal import Decimal
from multiprocessing import cpu_count
from pathlib import Path

import environ

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env(
    # set casting, default value
    DEBUG=(bool, True)
)
environ.Env.read_env(BASE_DIR / "tradero" / ".env")

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env.str("SECRET_KEY", "django-insecure-3p=+w%*pm%73a")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env("DEBUG")

ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["*"])

# Application definition

INSTALLED_APPS = [
    "daphne",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    #
    "axes",
    "django_celery_beat",
    "django_celery_results",
    "websocketclient",
    "rest_framework",
    "bootstrap5",
    "nested_admin",
    "mathfilters",
    # django-ai apps
    "django_ai",
    "django_ai.ai_base",
    "django_ai.supervised_learning",
    #
    "base",
    #
    "drf_spectacular",
]

AUTHENTICATION_BACKENDS = [
    "axes.backends.AxesStandaloneBackend",
    "django.contrib.auth.backends.ModelBackend",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "axes.middleware.AxesMiddleware",
]

ROOT_URLCONF = "tradero.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

ASGI_APPLICATION = "tradero.asgi.application"

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [
                (
                    env.str("CHANNELS_REDIS_HOST", "127.0.0.1"),
                    env.int("CHANNELS_REDIS_PORT", 6379),
                )
            ],
            "capacity": 2000,
            "channel_capacity": {
                "http.request": 2000,
                "websocket.send*": 2000,
            },
        },
    },
}

# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    "default": env.db_url(
        "DATABASE_URL",
        default=("postgres://tradero@127.0.0.1:5432/tradero"),
    )
}

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": env.str("REDIS_URL", "redis://127.0.0.1:6379/5"),
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            "PICKLE_VERSION": -1,
        },
    }
}

RQ_QUEUES = {
    "default": {
        "USE_REDIS_CACHE": "default",
        "DEFAULT_TIMEOUT": 900,
    },
    "high": {
        "USE_REDIS_CACHE": "default",
        "DEFAULT_TIMEOUT": 900,
    },
    "low": {
        "USE_REDIS_CACHE": "default",
        "DEFAULT_TIMEOUT": 900,
    },
}

RQ_SHOW_ADMIN_LINK = True

# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = False

USE_TZ = False


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = "static/"

STATIC_ROOT = env.str("STATIC_ROOT", "/vol/tradero/static/")


# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Custom User Model
AUTH_USER_MODEL = "base.User"

LOGIN_URL = "/Usuário/Entrar"
LOGIN_REDIRECT_URL = "/Usuário"
LOGOUT_REDIRECT_URL = "/"

# Websocket client

WEBSOCKETCLIENT_HOST = "stream.binance.com:9443"
WEBSOCKETCLIENT_MESSAGE_HANDLER = "base.handlers.message_handler"
WEBSOCKETCLIENT_CONNECT_SECURE = True

SPECTACULAR_SETTINGS = {
    "TITLE": "tradero REST API",
    "DESCRIPTION": "a tool for achieving self-funding via trading",
    "VERSION": "0.0.1",
    "SERVE_INCLUDE_SCHEMA": False,
}

REST_FRAMEWORK = {
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.LimitOffsetPagination",
    "PAGE_SIZE": 25,
}

# django-ai

DJANGO_AI_METRICS = [
    "sklearn.metrics._scorer._SCORERS",
    "base.metrics.METRICS",
]

DJANGO_AI_METRICS_FORMATS = [
    "django_ai.ai_base.metrics_formats.BASE_METRICS_FORMATS",
]

DJANGO_AI_METRICS_FORMATS_MAPPING = [
    "django_ai.ai_base.metrics_formats.BASE_METRICS_FORMATS_MAPPING",
]

# Celery Configuration Options
CELERY_TIMEZONE = "UTC"
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60
CELERY_RESULT_BACKEND = "django-db"
CELERY_CACHE_BACKEND = "default"
CELERY_BEAT_SCHEDULER = "django_celery_beat.schedulers:DatabaseScheduler"
CELERY_BROKER_URL = env.str("REDIS_URL", "redis://127.0.0.1:6379/5")

DJANGO_CELERY_BEAT_TZ_AWARE = False

# tradero specific

# tradero Debug Mode
TRADERO_DEBUG = env.bool("TRADERO_DEBUG", False)

# Exchange API URL (Binance's)
EXCHANGE_API_URL = env.str(
    "TRADERO_EXCHANGE_API_URL", "https://api.binance.com"
)

# Quote Asset to be used
QUOTE_ASSET = env.str("TRADERO_QUOTE_ASSET", "BUSD")

# Exhange fee per transaction (Buy / Sell) (rate) for DUMMY mode
EXCHANGE_FEE = Decimal(env.str("TRADERO_EXCHANGE_FEE", "0.001"))

# Time interval in minutes for prediction and updating symbols
TIME_INTERVAL = env.int("TRADERO_TIME_INTERVAL", 5)

# Time interval in minutes for updating bots
TIME_INTERVAL_BOTS = env.int("TRADERO_TIME_INTERVAL_BOTS", 1)

# Synchronous Execution
SYNC_EXECUTION = env.bool("TRADERO_SYNC_EXECUTION", False)

# Threads to be used in ThreadPoolExecutor
EXECUTOR_THREADS = env.str("TRADERO_EXECUTOR_THREADS", None)

# Symbols quantity to track
SYMBOLS_QUANTITY = env.int("TRADERO_SYMBOLS_QUANTITY", cpu_count() * 4)

# Periods for the warm_and_ready command
WARM_UP_PERIODS = env.int("TRADERO_WARM_UP_PERIODS", 0)

# Minimun Last 24h Volume of Quote Asset
MARKET_SIZE_THRESHOLD = env.int("TRADERO_MARKET_SIZE_THRESHOLD", 180000)

# General Model Score, to be used in base filtering
MODEL_SCORE_THRESHOLD = env.float("TRADERO_MODEL_SCORE_THRESHOLD", 0.3)

BOT_DEFAULT_NAME = "BTZN"

# Preediction Model Class
SERIALIZER_CLASS = env.str(
    "TRADERO_SERIALIZER_CLASS",
    "base.api.v1.serializers.SymbolSerializer",
)

# Enable Prediction for the next period
PREDICTION_ENABLED = env.bool("TRADERO_PREDICTION_ENABLED", True)

# Preediction Model Class
PREDICTION_MODEL_CLASS = env.str(
    "TRADERO_PREDICTION_MODEL_CLASS",
    "base.models.DecisionTreeRegressor",
)

# Enable Outliers detection
OUTLIERS_ENABLED = env.bool("TRADERO_OUTIERS_ENABLED", True)

# Preediction Model Class
OUTLIERS_MODEL_CLASS = env.str(
    "TRADERO_OUTLIERS_MODEL_CLASS",
    "base.models.OutliersSVC",
)

# Proportion to be considered as atypical
OUTLIERS_THRESHOLD = env.float("TRADERO_OUTLIERS_THRESHOLD", 0.05)

# Available Indicators
INDICATORS = env.list("TRADERO_INDICATORS", default=["__all__"])

# Indicators Settings

# MACD Parameters
MACD_CG = (25, 99, 7)

# STP Parameters
STP = 3
