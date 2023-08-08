from decimal import Decimal

from django.utils import timezone


def datetime_minutes_rounder(t):
    # "Rounds up" by adding one minute and resetting the seconds
    return t.replace(second=0, microsecond=0) + timezone.timedelta(minutes=1)


def get_commission(receipt):
    """
    Gets the amount of the comission from a receipt
    """
    commission = sum(
        [Decimal(fill["commission"]) for fill in receipt["fills"]]
    )
    return commission, receipt["fills"][0]["commissionAsset"]


def count_periods_from_now(list_to_count):
    periods = 0
    i = len(list_to_count) - 1
    flag = False
    while i >= 0 and not flag:
        if list_to_count[i]:
            periods += 1
            i -= 1
        else:
            flag = True
    return periods
