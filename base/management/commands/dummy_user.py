import logging

from django.conf import settings
from django.core.management.base import BaseCommand

from base.models import Symbol, TraderoBot, TraderoBotGroup, User

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Creates / Reset / Removes the Dummy User"

    def add_arguments(self, parser):
        parser.add_argument(
            "--create",
            action="store_true",
            default=True,
            help="Create the Dummy User",
        )
        parser.add_argument(
            "--reset",
            action="store_true",
            default=True,
            help="Resets the Dummy User (user and bots)",
        )
        parser.add_argument(
            "--remove",
            action="store_true",
            default=False,
            help="Remove the Dummy User (and its bots)",
        )

    def handle(self, *args, **options):
        if settings.DUMMY_USER_ENABLED:
            if options["remove"] or options["reset"]:
                du = User.objects.filter(username="dummy").first()
                if du:
                    for bot in du.bots.all():
                        bot.trades.all().delete()
                        bot.logs.all().delete()
                        bot.delete()
                    du.botgroups.all().delete()
                    du.delete(())
            if (options["create"] or options["reset"]) and not options[
                "remove"
            ]:
                du, du_created = User.objects.get_or_create(
                    username="dummy",
                    defaults={
                        "first_name": "Dummy",
                        "last_name": "User",
                        "email": "dummy@tradero.dev",
                        "bot_quota": settings.DUMMY_USER_BOT_QUOTA,
                    },
                )
                du.set_password("dummy")
                du.save()
                if du_created:
                    symbol = Symbol.objects.filter(
                        symbol__startswith=settings.DUMMY_USER_SYMBOL
                    ).first()
                    group = TraderoBotGroup.objects.filter(user=du).first()
                    for i in range(settings.DUMMY_USER_BOTS):
                        bot = TraderoBot(
                            user=du,
                            symbol=symbol,
                            group=group,
                            strategy="catchthewave",
                            is_dummy=True,
                            is_jumpy=True,
                            should_reinvest=True,
                            fund_quote_asset_initial=50,
                        )
                        bot.save()
                        bot.on()
        message = "SUCCESS at managing the Dummy User"
        self.stdout.write(self.style.SUCCESS(message))
