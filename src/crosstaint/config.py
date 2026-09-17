from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_INTERP_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _candidate_config_dirs() -> list[Path]:
    configured = os.environ.get("CROSSTAINT_CONFIG_DIR")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend([
        Path.cwd() / "config",
        Path(__file__).resolve().parents[2] / "config",
    ])
    return candidates


def _default_config_dir() -> Path:
    for candidate in _candidate_config_dirs():
        if candidate.exists() and any(candidate.glob("*.yaml")):
            return candidate
    searched = ", ".join(str(path) for path in _candidate_config_dirs())
    raise FileNotFoundError(f"no config directory found; searched: {searched}")


def _env_interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            env_var = m.group(1)
            return os.environ.get(env_var, m.group(0))
        return _ENV_INTERP_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _env_interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_env_interpolate(v) for v in value]
    return value


class Config:
    _instance: "Config | None" = None
    _config_hash: str = ""

    def __init__(self) -> None:
        self._loaded: dict[str, Any] = {}

    @classmethod
    def load(cls, config_dir: Path | None = None) -> "Config":
        if cls._instance is not None:
            return cls._instance

        cfg = cls()
        directory = config_dir or _default_config_dir()
        cfg._loaded = {}

        for yaml_file in sorted(directory.glob("*.yaml")):
            with open(yaml_file, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            if raw is None:
                raw = {}
            name = yaml_file.stem
            if name in raw:
                cfg._loaded[name] = _env_interpolate(raw[name])
            else:
                cfg._loaded[name] = _env_interpolate(raw)

        raw_str = json.dumps(cfg._loaded, sort_keys=True, default=str)
        cfg._config_hash = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:12]

        cls._instance = cfg
        cls._config_hash = cfg._config_hash
        return cfg

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
        cls._config_hash = ""

    @classmethod
    def config_hash(cls) -> str:
        return cls._config_hash

    def __getattr__(self, name: str) -> Any:
        if name in self._loaded:
            return self._loaded[name]
        raise AttributeError(f"No config section '{name}'. Available: {list(self._loaded.keys())}")

    def get(self, section: str, default: Any = None) -> Any:
        return self._loaded.get(section, default)

    @property
    def chains(self) -> dict[str, dict]:
        raw = self._loaded.get("chains")
        if isinstance(raw, list):
            return {str(i): item for i, item in enumerate(raw)}
        if isinstance(raw, dict):
            first_key = next(iter(raw.keys()), None)
            if first_key is not None and isinstance(raw[first_key], list):
                return {str(i): item for i, item in enumerate(raw[first_key])}
        return raw if raw else {}

    @property
    def bridges(self) -> dict[str, dict]:
        raw = self._loaded.get("bridges")
        if isinstance(raw, list):
            return {str(i): item for i, item in enumerate(raw)}
        if isinstance(raw, dict):
            first_key = next(iter(raw.keys()), None)
            if first_key is not None and isinstance(raw[first_key], list):
                return {str(i): item for i, item in enumerate(raw[first_key])}
        return raw if raw else {}

    @property
    def matcher(self) -> dict:
        return self._loaded.get("matcher", {})

    @property
    def pseudonym(self) -> dict:
        return self._loaded.get("pseudonym", {})

    @property
    def propagation(self) -> dict:
        return self._loaded.get("propagation", {})

    @property
    def eval(self) -> dict:
        return self._loaded.get("eval", {})

    @property
    def chain_list(self) -> list[str]:
        return list(self.chains.keys())

    @property
    def bridge_list(self) -> list[str]:
        return list(self.bridges.keys())

    def bridge_spec(self, bridge_id: str) -> dict:
        bridges = self._loaded.get("bridges", {})
        if isinstance(bridges, dict) and bridge_id in bridges:
            spec = dict(bridges[bridge_id])
            spec.setdefault("bridge_id", bridge_id)
            return spec
        for b in self.bridges.values():
            if b.get("bridge_id") == bridge_id:
                return b
        raise KeyError(f"Bridge '{bridge_id}' not found in config")

    def chain_spec(self, chain_id: str) -> dict:
        chains = self._loaded.get("chains", {})
        if isinstance(chains, dict) and chain_id in chains:
            spec = dict(chains[chain_id])
            spec.setdefault("chain_id", chain_id)
            return spec
        for c in self.chains.values():
            if c.get("chain_id") == chain_id:
                return c
        raise KeyError(f"Chain '{chain_id}' not found in config")


def get_config() -> Config:
    return Config.load()
