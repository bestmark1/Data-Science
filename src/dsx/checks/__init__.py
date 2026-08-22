"""Проверки, объявляющие свои предпосылки и блокирующий статус."""

from dsx.checks.base import Check, Context, Report, Signal, Skipped, run_checks
from dsx.checks.contract import CONTRACT_CHECKS
from dsx.checks.data import DATA_CHECKS
from dsx.checks.empirical import EMPIRICAL_CHECKS
from dsx.checks.premise_check import PREMISE_CHECKS
from dsx.checks.split_checks import SPLIT_CHECKS

ALL_CHECKS = [*CONTRACT_CHECKS, *DATA_CHECKS, *EMPIRICAL_CHECKS, *SPLIT_CHECKS, *PREMISE_CHECKS]

__all__ = [
    "ALL_CHECKS",
    "CONTRACT_CHECKS",
    "DATA_CHECKS",
    "EMPIRICAL_CHECKS",
    "PREMISE_CHECKS",
    "SPLIT_CHECKS",
    "Check",
    "Context",
    "Report",
    "Signal",
    "Skipped",
    "run_checks",
]
