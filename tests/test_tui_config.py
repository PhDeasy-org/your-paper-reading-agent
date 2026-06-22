from ppagent.config import AppConfig
from ppagent.tui import (
    get_menu_definition,
    set_config_value,
    _switch_vendor,
)
from ppagent.providers import (
    detect_provider as detect_vendor,
    PROVIDERS,
    get_provider,
)

def test_detect_vendor():
    # Test standard vendors
    assert detect_vendor("https://api.openai.com/v1") == "openai"
    assert detect_vendor("https://api.deepseek.com") == "deepseek"
    assert detect_vendor("https://api.deepseek.com/v1") == "deepseek"
    assert detect_vendor("https://api.mistral.ai/v1") == "mistral"
    assert (
        detect_vendor("https://generativelanguage.googleapis.com/v1beta/openai")
        == "gemini"
    )
    assert detect_vendor("https://api.anthropic.com/v1") == "anthropic"
    assert detect_vendor("https://dashscope.aliyuncs.com/compatible-mode/v1") == "qwen"
    assert detect_vendor("https://api.moonshot.ai/v1") == "kimi_ai"
    assert detect_vendor("https://api.moonshot.cn/v1") == "kimi_cn"
    assert detect_vendor("https://open.bigmodel.cn/api/paas/v4") == "glm"
    assert detect_vendor("https://api.x.ai/v1") == "grok"
    assert detect_vendor("https://api.stepfun.ai/v1") == "stepfun"
    assert detect_vendor("https://api.minimax.io/v1") == "minimax"
    assert detect_vendor("https://api.xiaomimimo.com/v1") == "mimo"
    assert detect_vendor("https://ark.cn-beijing.volces.com/api/v3") == "doubao"
    assert detect_vendor("https://api.hunyuan.cloud.tencent.com") == "tencent"

    # Test custom vendor cases
    assert detect_vendor("https://mycustomapi.com/v1") == "custom"
    assert detect_vendor(None) == "custom"
    assert detect_vendor("") == "custom"


def test_get_menu_definition_vendor_list():
    cfg = AppConfig()
    # Set text LLM base URL to deepseek
    cfg.llms.text.base_url = "https://api.deepseek.com"

    menu_items = get_menu_definition("llm_text_vendor", cfg)
    assert len(menu_items) == len(PROVIDERS) + 1  # PROVIDERS plus back menu item

    # Check that DeepSeek has the active label
    deepseek_item = next(
        item for item in menu_items if "deepseek" in (item.target or "")
    )
    assert "Active" in deepseek_item.label

    # Check that OpenAI is not active
    openai_item = next(item for item in menu_items if "openai" in (item.target or ""))
    assert "Active" not in openai_item.label


def test_get_menu_definition_vendor_list_searcher():
    cfg = AppConfig()
    menu_items = get_menu_definition("llm_searcher_vendor", cfg)

    # 4 providers (openai, qwen, doubao, grok) + 1 back item = 5 items
    assert len(menu_items) == 5

    # Verify exact provider targets are present
    targets = {item.target for item in menu_items if item.target != "back"}
    assert targets == {
        "llm_searcher_openai",
        "llm_searcher_qwen",
        "llm_searcher_doubao",
        "llm_searcher_grok",
    }


def test_get_menu_definition_vendor_setting():
    cfg = AppConfig()

    # Check menu items for DeepSeek - API Base URL should not be in the fields
    deepseek_settings = get_menu_definition("llm_text_deepseek", cfg)
    assert not any(item.label == "API Base URL" for item in deepseek_settings)
    assert any(item.key == "llms.text.api_key" for item in deepseek_settings)

    # Check menu items for Custom OpenAI Compatible - API Base URL should be in the fields
    custom_settings = get_menu_definition("llm_text_custom", cfg)
    assert any(item.label == "API Base URL" for item in custom_settings)


def test_switch_vendor_preserves_each_provider_values():
    """Switching providers should keep each provider's own api_key/model.

    This is the regression test for the bug where values entered for one
    provider leaked into every other provider's settings page.
    """
    cfg = AppConfig()
    # Start: text role is OpenAI (the default base_url).
    assert detect_vendor(cfg.llms.text.base_url) == "openai"

    # Enter DeepSeek and fill in its credentials/model.
    _switch_vendor(cfg, "text", "deepseek")
    cfg.llms.text.api_key = "deepseek-key"
    cfg.llms.text.model = "deepseek-chat"

    # Switch to Qwen and fill in different credentials.
    _switch_vendor(cfg, "text", "qwen")
    cfg.llms.text.api_key = "qwen-key"
    cfg.llms.text.model = "qwen-plus"

    # The live config now reflects Qwen.
    assert cfg.llms.text.api_key == "qwen-key"
    assert cfg.llms.text.model == "qwen-plus"
    assert detect_vendor(cfg.llms.text.base_url) == "qwen"

    # Switching back to DeepSeek restores DeepSeek's values (NOT qwen's).
    _switch_vendor(cfg, "text", "deepseek")
    assert cfg.llms.text.api_key == "deepseek-key"
    assert cfg.llms.text.model == "deepseek-chat"
    assert detect_vendor(cfg.llms.text.base_url) == "deepseek"

    # And the live DeepSeek edit didn't overwrite the stored Qwen snapshot.
    _switch_vendor(cfg, "text", "qwen")
    assert cfg.llms.text.api_key == "qwen-key"
    assert cfg.llms.text.model == "qwen-plus"


def test_switch_vendor_seeds_fresh_config_on_first_visit():
    """A never-before-used vendor gets a fresh config with its base_url/model."""
    cfg = AppConfig()  # text defaults to openai
    _switch_vendor(cfg, "text", "glm")
    assert detect_vendor(cfg.llms.text.base_url) == "glm"
    assert cfg.llms.text.api_key == ""  # blank creds, not the openai default
    # Model is seeded from the vendor's default_model.
    assert cfg.llms.text.model == get_provider("glm").default_model


def test_switch_vendor_same_vendor_is_noop():
    """Re-entering the already-active vendor must not wipe current edits."""
    cfg = AppConfig()
    _switch_vendor(cfg, "text", "deepseek")
    cfg.llms.text.api_key = "deepseek-key"
    cfg.llms.text.model = "deepseek-chat"
    # Re-select the same active vendor — values must survive.
    _switch_vendor(cfg, "text", "deepseek")
    assert cfg.llms.text.api_key == "deepseek-key"
    assert cfg.llms.text.model == "deepseek-chat"


def test_snapshot_active_vendor_then_save_persists_active_edits(tmp_path):
    """save_config snapshots the active vendor for each role before writing."""
    from ppagent.tui import save_config

    cfg = AppConfig()
    cfg.root = tmp_path
    _switch_vendor(cfg, "text", "deepseek")
    cfg.llms.text.api_key = "deepseek-key"
    cfg.llms.text.model = "deepseek-chat"

    config_file = tmp_path / "settings.toml"
    save_config(cfg, config_file)

    # The active DeepSeek config was snapshotted under its vendor key.
    assert cfg.llms.saved_vendors["text"]["deepseek"].api_key == "deepseek-key"
    assert cfg.llms.saved_vendors["text"]["deepseek"].model == "deepseek-chat"

    # Reload and verify the TOML round-trips saved_vendors correctly.
    import tomllib

    with open(config_file, "rb") as f:
        raw = tomllib.load(f)
    assert raw["llms"]["saved_vendors"]["text"]["deepseek"]["api_key"] == "deepseek-key"
    assert raw["llms"]["text"]["api_key"] == "deepseek-key"


def test_vision_and_searcher_roles_independent():
    """Switching vendors in one role must not bleed into the other roles."""
    cfg = AppConfig()
    _switch_vendor(cfg, "vision", "glm")
    cfg.llms.vision.api_key = "glm-vision-key"
    _switch_vendor(cfg, "searcher", "deepseek")
    cfg.llms.searcher.api_key = "ds-searcher-key"

    # Text role is untouched by the above switches.
    assert detect_vendor(cfg.llms.text.base_url) == "openai"
    assert cfg.llms.text.api_key == ""

    # Each role retains its own active provider.
    assert cfg.llms.vision.api_key == "glm-vision-key"
    assert cfg.llms.searcher.api_key == "ds-searcher-key"


# ---------------------------------------------------------------------------
# "-latest" / predefined model picker
# ---------------------------------------------------------------------------


def test_latest_models_nav_item_shown_for_every_provider_with_aliases():
    """Every provider with a non-empty latest_models shows 'Latest Models'."""
    cfg = AppConfig()
    expected_with_picker = {
        spec.key for spec in PROVIDERS if spec.latest_models
    }
    expected_without = {spec.key for spec in PROVIDERS if not spec.latest_models}

    for vendor_key in expected_with_picker:
        settings = get_menu_definition(f"llm_text_{vendor_key}", cfg)
        assert any(item.label == "Latest Models" for item in settings), (
            f"{vendor_key} should show Latest Models"
        )

    # Providers without a picker must not show the item.
    for vendor_key in expected_without:
        settings = get_menu_definition(f"llm_text_{vendor_key}", cfg)
        assert not any(item.label == "Latest Models" for item in settings), (
            f"{vendor_key} should NOT show Latest Models"
        )


def test_latest_models_nav_target_is_correct():
    grok_settings = get_menu_definition("llm_text_grok", AppConfig())
    nav = next(item for item in grok_settings if item.label == "Latest Models")
    assert nav.target == "llm_text_grok_latest"

    gemini_settings = get_menu_definition("llm_vision_gemini", AppConfig())
    nav = next(item for item in gemini_settings if item.label == "Latest Models")
    assert nav.target == "llm_vision_gemini_latest"


def test_latest_models_picker_lists_vendor_aliases():
    """The picker lists the vendor's latest_models, then a Custom Model entry."""
    cfg = AppConfig()
    grok_spec = get_provider("grok")
    assert grok_spec is not None

    picker = get_menu_definition("llm_text_grok_latest", cfg)
    assert picker[0].target == "back"

    # Predefined options carry set_value; the trailing Custom Model entry does not.
    predefined = [o for o in picker[1:] if o.set_value is not None]
    assert [o.set_value for o in predefined] == list(grok_spec.latest_models)

    for opt in predefined:
        assert opt.key == "llms.text.model"


def test_picker_always_ends_with_custom_model_entry():
    """Every picker menu ends with a 'Custom Model (type your own)' entry.

    It carries the role's model key (so the typed value lands in the right
    place) but uses a target (not set_value) so it routes through the
    free-text edit path instead of the picker write-and-pop path.
    """
    cfg = AppConfig()
    for vendor_key in ("grok", "openai", "minimax", "qwen"):
        picker = get_menu_definition(f"llm_text_{vendor_key}_latest", cfg)
        last = picker[-1]
        assert last.label == "Custom Model (type your own)..."
        assert last.key == "llms.text.model"
        assert last.set_value is None
        assert last.target == f"llm_text_{vendor_key}_custom_model"


def test_latest_models_picker_marks_current_model_active():
    cfg = AppConfig()
    cfg.llms.text.model = "grok-4-latest"

    picker = get_menu_definition("llm_text_grok_latest", cfg)
    # Only consider the predefined options for the Active marker; the Custom
    # Model entry never carries it.
    predefined = [o for o in picker if o.set_value is not None]
    active = [o for o in predefined if "Active" in o.label]
    assert len(active) == 1
    assert active[0].set_value == "grok-4-latest"

    inactive = [o for o in predefined if o.set_value != "grok-4-latest"]
    assert all("Active" not in o.label for o in inactive)

    # The Custom Model entry is never marked active.
    custom = picker[-1]
    assert "Active" not in custom.label


def test_picker_set_value_branch_writes_model_and_is_idempotent():
    """Selecting a picker option must write the model to the role's config.

    This mirrors the run_config_tui handler: ``set_config_value`` with the
    item's ``set_value``.  Exercised directly (rather than via the raw-keypress
    loop) because the loop needs a real TTY.
    """
    cfg = AppConfig()
    picker = get_menu_definition("llm_text_grok_latest", cfg)
    target = next(o for o in picker if o.set_value == "grok-4-latest")

    set_config_value(cfg, target.key, target.set_value)
    assert cfg.llms.text.model == "grok-4-latest"


def test_latest_picker_menu_id_not_misrouted_to_vendor_settings():
    """The '_latest' suffix must not be captured as a vendor key.

    Regression guard: the generic vendor-settings regex
    ``^llm_(text|vision|searcher)_([a-z0-9_]+)$`` would otherwise swallow
    "grok_latest" from "llm_text_grok_latest". The picker menu definition
    resolves it instead.
    """
    cfg = AppConfig()
    # If misrouted to _llm_submenu_items, the result would contain "API Key"
    # and "Model Name". A picker menu contains "back" + alias options only.
    picker = get_menu_definition("llm_text_grok_latest", cfg)
    assert not any(item.key == "llms.text.api_key" for item in picker)
    assert not any(item.label == "Model Name" for item in picker)


# ---------------------------------------------------------------------------
# Shared api_key across roles (per-provider, not per-role)
# ---------------------------------------------------------------------------


def test_api_key_change_propagates_to_sibling_roles_on_same_provider():
    """Editing a key in one role mirrors it into every sibling on that provider.

    OpenAI is the default base_url for text/vision/searcher, so all three are
    on the same provider. Entering the key once via "vision" should make it
    appear in "text" and "searcher" immediately.
    """
    cfg = AppConfig()
    assert detect_vendor(cfg.llms.text.base_url) == "openai"
    assert detect_vendor(cfg.llms.vision.base_url) == "openai"
    assert detect_vendor(cfg.llms.searcher.base_url) == "openai"

    set_config_value(cfg, "llms.vision.api_key", "shared-openai-key")

    assert cfg.llms.text.api_key == "shared-openai-key"
    assert cfg.llms.searcher.api_key == "shared-openai-key"
    assert cfg.llms.vision.api_key == "shared-openai-key"


def test_api_key_change_does_not_touch_different_provider():
    """A key change must not leak into roles on a different provider."""
    cfg = AppConfig()
    # Move searcher to deepseek so it's no longer on the openai provider.
    _switch_vendor(cfg, "searcher", "deepseek")

    set_config_value(cfg, "llms.vision.api_key", "openai-key")

    assert cfg.llms.text.api_key == "openai-key"  # same provider (openai)
    assert cfg.llms.vision.api_key == "openai-key"
    # searcher is on deepseek — must stay blank, not inherit the openai key.
    assert cfg.llms.searcher.api_key == ""


def test_switch_vendor_inherits_existing_key_for_same_provider():
    """Visiting a provider that a sibling already configured reuses its key.

    Sets up text=openai with a key, then switches vision to a *different*
    provider and back... instead we test the direct path: switch vision to
    openai (its default) after text already holds an openai key. Since vision
    is already on openai by default, use searcher instead.
    """
    cfg = AppConfig()
    cfg.llms.text.api_key = "openai-key-from-text"
    # searcher also defaults to openai but has no key yet.
    assert cfg.llms.searcher.api_key == ""

    # Switch searcher away to deepseek, then back to openai. On the way back,
    # it should pick up the key text already holds for openai rather than
    # present a blank field.
    _switch_vendor(cfg, "searcher", "deepseek")
    assert detect_vendor(cfg.llms.searcher.base_url) == "deepseek"
    _switch_vendor(cfg, "searcher", "openai")
    assert cfg.llms.searcher.api_key == "openai-key-from-text"


def test_save_config_persists_propagated_keys(tmp_path):
    """save_config round-trips keys that were propagated across roles."""
    from ppagent.tui import save_config
    import tomllib

    cfg = AppConfig()
    cfg.root = tmp_path
    set_config_value(cfg, "llms.vision.api_key", "shared-openai-key")

    config_file = tmp_path / "settings.toml"
    save_config(cfg, config_file)

    with open(config_file, "rb") as f:
        raw = tomllib.load(f)
    # All three roles were on openai, so all three persist the shared key.
    assert raw["llms"]["text"]["api_key"] == "shared-openai-key"
    assert raw["llms"]["vision"]["api_key"] == "shared-openai-key"
    assert raw["llms"]["searcher"]["api_key"] == "shared-openai-key"
