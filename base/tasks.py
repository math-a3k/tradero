from celery import shared_task
from django.core.management import call_command

from .models import Symbol, TraderoBot, TraderoBotGroup


@shared_task
def retrieve_and_update_symbol(symbol_pk, push=False):  # pragma: no cover
    message = Symbol.objects.get(pk=symbol_pk).retrieve_and_update(push=push)
    return message


@shared_task
def update_all_indicators_job(
    all_symbols=False, load_symbols=False
):  # pragma: no cover
    if load_symbols:
        Symbol.load_symbols()
    if (
        Symbol.objects.available().filter(model_score__isnull=True).count() > 0
        or all_symbols
    ):
        # Case of cold start
        message = Symbol.update_all_indicators(push=False)
    message = Symbol.update_all_indicators(only_top=True)
    return message


@shared_task
def update_all_bots_job():  # pragma: no cover
    message = TraderoBot.update_all_bots()
    return message


@shared_task
def update_bots_group_job(group_pk):  # pragma: no cover
    message = TraderoBotGroup.objects.get(pk=group_pk).update_bots()
    return message


@shared_task
def bots_logrotate():  # pragma: no cover
    message = TraderoBot.logrotate()
    return message


@shared_task
def symbols_datarotate():  # pragma: no cover
    Symbol.datarotate()


@shared_task
def dummy_user_reset():  # pragma: no cover
    call_command("dummy_user", reset=True)
