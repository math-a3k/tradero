# Generated by Django 4.2.4 on 2023-08-28 02:42

from django.db import migrations, models


def remove_duplicates(apps, schema_editor):
    """
    Remvos duplicates on the (user, bot, timestart_key) if
    they exists.
    """
    TradeHistory = apps.get_model("base", "TradeHistory")
    dups = (
        TradeHistory.objects.all()
        .order_by()
        .annotate(
            key=models.functions.Concat(
                models.F("user"),
                models.Value("--"),
                models.F("bot"),
                models.Value("--"),
                models.F("timestamp_start"),
                output_field=models.CharField(),
            )
        )
        .values("timestamp_start", "key")
        .annotate(dup_count=models.Count("key"))
    ).filter(dup_count__gt=1)
    for ts in dups:
        records = TradeHistory.objects.filter(
            timestamp_start=ts["timestamp_start"]
        )
        for record in records[1:]:
            record.delete()


class Migration(migrations.Migration):
    dependencies = [
        ("base", "0015_alter_traderobot_fund_quote_asset_initial"),
    ]

    operations = [
        migrations.RunPython(
            code=remove_duplicates,
            reverse_code=migrations.RunPython.noop,
        ),
        migrations.AddConstraint(
            model_name="tradehistory",
            constraint=models.UniqueConstraint(
                fields=("user", "bot", "timestamp_start"),
                name="unique_user_bot_timestamp_start",
            ),
        ),
    ]
