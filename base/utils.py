from django.utils import timezone


def datetime_minutes_rounder(t):
    # "Rounds up" by adding one minute and resetting the seconds
    return t.replace(second=0, microsecond=0) + timezone.timedelta(minutes=1)
