from jdatetime import datetime as jdatetime
from datetime import datetime


def gregorian_to_jalali(year, month):
    """Convert Gregorian date to Jalali"""
    g_date = f"{year}-{month:02d}-01"
    j_date = jdatetime.fromgregorian(date=datetime.strptime(g_date, "%Y-%m-%d").date())
    return j_date.year, j_date.month


def get_jalali_season(j_month):
    """Get season number (1-4) for Jalali month"""
    return (j_month - 1) // 3 + 1
