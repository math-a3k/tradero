from celery import shared_task

from .models import Symbol, TraderoBot


@shared_task
def retrieve_and_update_symbol(symbol):  # pragma: no cover
    symbol.retrieve_and_update()


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
