"""P1-4：辩论 worker 与主线程 ConfigService / Database 完全隔离。

* :func:`build_provider_config_snapshot` 应当从 live ``ConfigService`` 抓取
  所有 worker 后续需要的字段（LLM provider / api_key / app.* / tushare.*）。
* :class:`_FrozenConfigService` 提供 ``get_app_config`` / ``get`` /
  ``get_default_llm_provider``，且写操作硬性 raise。
* 主线程 ``Database`` 强制关闭后，新建 :class:`LLMManager`
  仍能基于 snapshot 解出 provider 信息（不会触发主连接读取）。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deeptrade.core.config import AppConfig, LLMProviderConfig

from limit_up_board.runtime import (
    ProviderConfigSnapshot,
    _FrozenConfigService,
    build_provider_config_snapshot,
)


# ---------------------------------------------------------------------------
# build_provider_config_snapshot
# ---------------------------------------------------------------------------


def _fake_config(providers: dict[str, LLMProviderConfig], api_keys: dict[str, str], default: str | None = None) -> MagicMock:
    """Mock that mimics the ConfigService surface used by build_provider_config_snapshot."""
    app_cfg = AppConfig(
        llm_providers=providers,
        tushare_rps=8.0,
        tushare_timeout=45,
        tushare_max_retries=5,
        app_profile="quality",
    )
    cfg = MagicMock()
    cfg.get_app_config.return_value = app_cfg
    cfg.get_default_llm_provider.return_value = default

    def fake_get(key: str):
        if key.startswith("llm.") and key.endswith(".api_key"):
            name = key[len("llm."): -len(".api_key")]
            return api_keys.get(name, "")
        return None

    cfg.get.side_effect = fake_get
    return cfg


def test_snapshot_captures_providers_and_secrets() -> None:
    providers = {
        "deepseek": LLMProviderConfig(base_url="https://api.deepseek.com", model="deepseek-chat", is_default=True),
        "openai": LLMProviderConfig(base_url="https://api.openai.com", model="gpt-4o", is_default=False),
    }
    api_keys = {"deepseek": "sk-DEEP-1", "openai": "sk-OAI-2"}
    snap = build_provider_config_snapshot(_fake_config(providers, api_keys, default="deepseek"))

    assert isinstance(snap, ProviderConfigSnapshot)
    assert snap.default_provider == "deepseek"
    assert set(snap.providers.keys()) == {"deepseek", "openai"}
    assert snap.providers["deepseek"].api_key == "sk-DEEP-1"
    assert snap.providers["openai"].api_key == "sk-OAI-2"
    # AppConfig fields carried over
    assert snap.app_profile == "quality"
    assert snap.tushare_rps == 8.0


def test_snapshot_empty_api_key_falls_back_to_blank_string() -> None:
    providers = {"x": LLMProviderConfig(base_url="https://x", model="m")}
    snap = build_provider_config_snapshot(_fake_config(providers, api_keys={"x": ""}, default=None))
    assert snap.providers["x"].api_key == ""


# ---------------------------------------------------------------------------
# _FrozenConfigService
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot() -> ProviderConfigSnapshot:
    providers = {
        "deepseek": LLMProviderConfig(base_url="https://api.deepseek.com", model="deepseek-chat", is_default=True),
    }
    return build_provider_config_snapshot(_fake_config(providers, {"deepseek": "sk-XYZ"}, default="deepseek"))


def test_frozen_config_get_app_config_recreates_app_config(snapshot: ProviderConfigSnapshot) -> None:
    frozen = _FrozenConfigService(snapshot)
    app_cfg = frozen.get_app_config()
    assert isinstance(app_cfg, AppConfig)
    assert app_cfg.app_profile == snapshot.app_profile
    assert "deepseek" in app_cfg.llm_providers


def test_frozen_config_get_api_key(snapshot: ProviderConfigSnapshot) -> None:
    frozen = _FrozenConfigService(snapshot)
    assert frozen.get("llm.deepseek.api_key") == "sk-XYZ"
    assert frozen.get("llm.missing.api_key") is None


def test_frozen_config_get_default(snapshot: ProviderConfigSnapshot) -> None:
    frozen = _FrozenConfigService(snapshot)
    assert frozen.get_default_llm_provider() == "deepseek"


def test_frozen_config_writes_raise(snapshot: ProviderConfigSnapshot) -> None:
    frozen = _FrozenConfigService(snapshot)
    with pytest.raises(RuntimeError, match="read-only"):
        frozen.set("app.profile", "fast")
    with pytest.raises(RuntimeError, match="read-only"):
        frozen.delete("llm.deepseek.api_key")
    with pytest.raises(RuntimeError, match="read-only"):
        frozen.set_llm_provider("x", LLMProviderConfig(base_url="u", model="m"))


# ---------------------------------------------------------------------------
# LLMManager interoperation: snapshot + frozen config drives the manager
# without ever touching the main thread's DB / SecretStore.
# ---------------------------------------------------------------------------


def test_llm_manager_resolves_against_frozen_config(snapshot: ProviderConfigSnapshot) -> None:
    """LLMManager only calls ``get_app_config`` / ``get(llm.<n>.api_key)`` /
    ``get_default_llm_provider``; the frozen surrogate must satisfy all three."""
    from deeptrade.core.llm_manager import LLMManager

    frozen = _FrozenConfigService(snapshot)
    db = MagicMock()  # LLMManager only uses db at client-construction time
    mgr = LLMManager(db, frozen)  # type: ignore[arg-type]

    # list_providers reads from frozen (which uses the snapshot, not the DB)
    assert mgr.list_providers() == ["deepseek"]
    info = mgr.get_provider_info("deepseek")
    assert info.model == "deepseek-chat"
