import pytest
from ppagent.config import AppConfig
from ppagent.tui import detect_vendor, get_menu_definition, VENDORS

def test_detect_vendor():
    # Test standard vendors
    assert detect_vendor("https://api.openai.com/v1") == "openai"
    assert detect_vendor("https://api.deepseek.com") == "deepseek"
    assert detect_vendor("https://api.deepseek.com/v1") == "deepseek"
    assert detect_vendor("https://api.mistral.ai/v1") == "mistral"
    assert detect_vendor("https://generativelanguage.googleapis.com/v1beta/openai") == "gemini"
    assert detect_vendor("https://api.anthropic.com/v1") == "anthropic"
    assert detect_vendor("https://dashscope.aliyuncs.com/compatible-mode/v1") == "qwen"
    assert detect_vendor("https://api.moonshot.ai/v1") == "kimi"
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
    assert len(menu_items) == len(VENDORS) + 1  # VENDORS plus back menu item
    
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
