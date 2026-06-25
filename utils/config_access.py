"""Safer access helpers for the global experiment config.

The project still uses ``conf.CONF`` as the source of truth. These helpers keep
that compatibility while making nested reads/writes more explicit and easier to
debug.
"""

from typing import Any

import conf


_MISSING = object()


class ConfigKeyError(KeyError):
    """Raised when a required experiment config path is missing."""


def _root() -> dict:
    if not hasattr(conf, "CONF"):
        raise ConfigKeyError("Experiment config has not been initialized.")
    return conf.CONF


def _parts(path: str) -> list[str]:
    if not path:
        raise ValueError("Config path cannot be empty.")
    return path.split(".")


def get(path: str, default: Any = _MISSING) -> Any:
    """Read a dotted config path, e.g. ``RL.phase_preprocessing``."""
    current = _root()
    for part in _parts(path):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif default is not _MISSING:
            return default
        else:
            raise ConfigKeyError(f"Missing required config value: {path}")
    return current


def has(path: str) -> bool:
    try:
        get(path)
    except ConfigKeyError:
        return False
    return True


def set_value(path: str, value: Any) -> None:
    """Set a dotted config path, creating intermediate dictionaries as needed."""
    current = _root()
    parts = _parts(path)
    for part in parts[:-1]:
        current = current.setdefault(part, {})
        if not isinstance(current, dict):
            raise ConfigKeyError(f"Cannot set nested config value under: {part}")
    current[parts[-1]] = value


def section(path: str) -> dict:
    value = get(path)
    if not isinstance(value, dict):
        raise ConfigKeyError(f"Config value is not a section: {path}")
    return value


def rl(path: str | None = None, default: Any = _MISSING) -> Any:
    return section("RL") if path is None else get(f"RL.{path}", default)


def rl_has(path: str) -> bool:
    return has(f"RL.{path}")


def rl_set(path: str, value: Any) -> None:
    set_value(f"RL.{path}", value)


def rewards() -> dict:
    return section("RL.RewardFnc")


def has_reward(name: str) -> bool:
    return name in rewards()


def reward_weight(name: str, default: Any = _MISSING) -> Any:
    reward_config = rewards()
    if name in reward_config:
        return reward_config[name]
    if default is not _MISSING:
        return default
    raise ConfigKeyError(f"Missing required reward weight: {name}")


def robot_arch(path: str | None = None, default: Any = _MISSING) -> Any:
    return section("robot_arch") if path is None else get(f"robot_arch.{path}", default)


def misc(path: str | None = None, default: Any = _MISSING) -> Any:
    return section("misc") if path is None else get(f"misc.{path}", default)
