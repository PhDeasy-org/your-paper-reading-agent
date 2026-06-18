from ppagent.config import AppConfig
from ppagent.tui import (
    get_menu_definition,
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
    assert detect_vendor("https://generativelanguage.googleapis.com/v1beta/openai") == "gemini"
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
    deepseek_item = next(item for item in menu_items if "deepseek" in (item.target or ""))
    assert "Active" in deepseek_item.label

    # Check that OpenAI is not active
    openai_item = next(item for item in menu_items if "openai" in (item.target or ""))
    assert "Active" not in openai_item.label

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

