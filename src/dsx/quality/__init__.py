"""Проверки, охраняющие инварианты репозитория."""

from dsx.quality.imports import ForbiddenImport, find_forbidden_imports

__all__ = ["ForbiddenImport", "find_forbidden_imports"]
