"""Canonical read-only tool factories."""

from collections.abc import Callable


def default_tools() -> tuple[Callable, ...]:
    """Return the recommended default read-only tools."""
    return ()


def math_tools() -> tuple[Callable, ...]:
    """Return math/calculation tools."""
    return ()


def spatial_tools() -> tuple[Callable, ...]:
    """Return read-only spatial query tools."""
    return ()


def environment_tools() -> tuple[Callable, ...]:
    """Return read-only environment/context tools."""
    return ()


def social_query_tools() -> tuple[Callable, ...]:
    """Return read-only social-context query tools."""
    return ()


def external_tools() -> tuple[Callable, ...]:
    """Return opt-in external tools."""
    return ()


__all__ = [
    "default_tools",
    "environment_tools",
    "external_tools",
    "math_tools",
    "social_query_tools",
    "spatial_tools",
]
