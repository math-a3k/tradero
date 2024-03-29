# Generated by Django 4.2.4 on 2023-09-01 05:36
from decimal import Decimal

from django.db import migrations, models


def get_commission(receipt):
    """
    Snapshot of utils.get_comissiog()
    """
    commission = sum(
        [Decimal(fill["commission"]) for fill in receipt["fills"]]
    )
    return commission, receipt["fills"][0]["commissionAsset"]


def parse_receipt(receipt):
    """
    Snapshot of TraderoClient.parse_receipt()
    """
    cum_exc_qty = Decimal(receipt["cummulativeQuoteQty"])
    exc_qty = Decimal(receipt["executedQty"])
    commission, commission_asset = get_commission(receipt)
    commission_e = commission if commission_asset != "BNB" else 0
    if receipt["side"] == "BUY":
        quantity_exec = cum_exc_qty
        quantity_rec = exc_qty - commission_e
        price = cum_exc_qty / quantity_rec  # Net price
    else:
        quantity_exec = exc_qty
        quantity_rec = cum_exc_qty - commission_e  # Net price
        price = quantity_rec / exc_qty
    r = {
        "quantity_exec": quantity_exec.quantize(Decimal("." + "0" * 8)),
        "quantity_rec": quantity_rec.quantize(Decimal("." + "0" * 8)),
        # Net price (except when comm in BNB)
        "price_net": price.quantize(Decimal("." + "0" * 8)),
        "commission": commission,
        "commission_asset": commission_asset,
    }
    return r


def update_trade_histories(apps, schema_editor):
    """
    Using a snapshot of .save() for updating the records
    """
    TradeHistory = apps.get_model("base", "TradeHistory")
    for ts in TradeHistory.objects.all():
        if ts.timestamp_buying and ts.timestamp_start:
            ts.duration_seeking = ts.timestamp_buying - ts.timestamp_start
        if ts.timestamp_selling and ts.timestamp_buying and ts.timestamp_start:
            ts.duration_trade = ts.timestamp_selling - ts.timestamp_buying
            ts.duration_total = ts.timestamp_selling - ts.timestamp_start
        if ts.receipt_buying:
            rb = parse_receipt(ts.receipt_buying)
            ts.fund_quote_asset_exec = rb["quantity_exec"]
            ts.commission_buying = rb["commission"]
            ts.commission_buying_asset = rb["commission_asset"]
            ts.fund_base_asset = rb["quantity_rec"]
        if ts.receipt_buying and ts.receipt_selling:
            ts.is_complete = True
            rs = parse_receipt(ts.receipt_selling)
            ts.fund_base_asset_exec = rs["quantity_exec"]
            ts.fund_base_asset_unexec = (
                ts.fund_base_asset - ts.fund_base_asset_exec
            )
            ts.commission_selling = rs["commission"]
            ts.commission_selling_asset = rs["commission_asset"]
            ts.fund_quote_asset_return = rs["quantity_rec"]
            ts.gain_quote_asset = (
                ts.fund_quote_asset_return - ts.fund_quote_asset_exec
            )
            ts.variation_quote_asset = (
                ts.fund_quote_asset_return / ts.fund_quote_asset_exec - 1
            ) * 100
            ts.variation_price = (rs["price_net"] / rb["price_net"] - 1) * 100
        ts.save()


class Migration(migrations.Migration):
    dependencies = [
        ("base", "0016_tradehistory_unique_user_bot_timestamp_start"),
    ]

    operations = [
        migrations.RenameField(
            model_name="tradehistory",
            old_name="variation",
            new_name="variation_price",
        ),
        migrations.AddField(
            model_name="tradehistory",
            name="gain_quote_asset",
            field=models.DecimalField(
                blank=True,
                decimal_places=8,
                max_digits=40,
                null=True,
                verbose_name="Gain of Quote Asset",
            ),
        ),
        migrations.RunPython(
            code=update_trade_histories,
            reverse_code=migrations.RunPython.noop,
        ),
    ]
