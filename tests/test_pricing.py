import pytest

from ppagent.models import AgentResult
from ppagent.agents.assembler import Assembler, calculate_cost
from ppagent.storage import Storage


def test_calculate_cost():
    # Test match with DeepSeek (flagship)
    usage = {
        "prompt_tokens": 100000,
        "completion_tokens": 50000,
        "total_tokens": 150000,
    }
    cost = calculate_cost("deepseek-chat", usage)
    assert cost is not None
    assert cost["provider"] == "DeepSeek"
    assert cost["input_price"] == 0.435
    assert cost["output_price"] == 0.87
    # 100,000 / 1,000,000 * 0.435 = 0.0435
    # 50,000 / 1,000,000 * 0.87 = 0.0435
    assert pytest.approx(cost["input_cost"]) == 0.0435
    assert pytest.approx(cost["output_cost"]) == 0.0435
    assert pytest.approx(cost["total_cost"]) == 0.087

    # Test match with Qwen
    cost = calculate_cost("qwen-plus", usage)
    assert cost is not None
    assert cost["provider"] == "Qwen"
    assert cost["input_price"] == 0.40
    assert cost["output_price"] == 1.20

    # Test match with specific submodels (OpenAI mini)
    cost = calculate_cost("gpt-4o-mini", usage)
    assert cost is not None
    assert cost["provider"] == "OpenAI"
    assert cost["input_price"] == 0.15
    assert cost["output_price"] == 0.60
    assert pytest.approx(cost["total_cost"]) == (100000 / 1e6 * 0.15) + (
        50000 / 1e6 * 0.60
    )

    # Test match with Google Flash
    cost = calculate_cost("gemini-1.5-flash", usage)
    assert cost is not None
    assert cost["provider"] == "Google"
    assert cost["input_price"] == 1.50
    assert cost["output_price"] == 9.00

    # Test match with unknown model -> should return None (skip cost report)
    cost = calculate_cost("some-unknown-model", usage)
    assert cost is None


def test_assembler_cost_report(tmp_path, sample_paper, template_dir):
    storage = Storage(tmp_path)

    # 1. Test when model is matched
    assembler = Assembler(
        template_dir=template_dir, storage=storage, model_used="deepseek-chat"
    )

    writer_result = AgentResult(
        agent_name="writer",
        success=True,
        data={"keywords": ["NLP"], "affiliations": ["A"]},
        usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
    )
    finder_result = AgentResult(
        agent_name="finder",
        success=True,
        data={"narrative": "Related work info", "related_works": []},
        usage={"prompt_tokens": 2000, "completion_tokens": 1000, "total_tokens": 3000},
    )
    criticizer_result = AgentResult(
        agent_name="criticizer",
        success=True,
        data={"critique": "A critique"},
        usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
    )

    report, md_content, html_content = assembler.assemble(
        paper=sample_paper,
        writer_result=writer_result,
        finder_result=finder_result,
        criticizer_result=criticizer_result,
    )

    # Check that aggregated usage is correct
    # prompt: 1000 + 2000 + 1000 = 4000
    # completion: 500 + 1000 + 500 = 2000
    # total: 1500 + 3000 + 1500 = 6000
    assert report.usage["prompt_tokens"] == 4000
    assert report.usage["completion_tokens"] == 2000
    assert report.usage["total_tokens"] == 6000

    # Check that cost report exists
    assert report.cost_report is not None
    assert report.cost_report["models"][0]["provider"] == "DeepSeek"
    assert report.cost_report["models"][0]["model"] == "deepseek-chat"
    assert pytest.approx(report.cost_report["total_cost"]) == (4000 / 1e6 * 0.435) + (
        2000 / 1e6 * 0.87
    )

    # Check that Markdown contains the cost breakdown (price table no longer rendered)
    assert "Generation Cost" in md_content
    assert "DeepSeek" in md_content
    assert "Total Report Generation Cost" in md_content
    assert "Official LLM Price Table" not in md_content

    # Check HTML content
    assert "Generation Cost" in html_content
    assert "Generation Cost Breakdown" in html_content
    assert "Total Cost" in html_content
    assert "Official LLM Price Table" not in html_content

    # 2. Test when model is NOT matched -> should skip cost breakdown entirely
    assembler_unmatched = Assembler(
        template_dir=template_dir, storage=storage, model_used="my-custom-llm"
    )
    report_unmatched, md_unmatched, html_unmatched = assembler_unmatched.assemble(
        paper=sample_paper,
        writer_result=writer_result,
        finder_result=finder_result,
        criticizer_result=criticizer_result,
    )

    assert report_unmatched.cost_report is None
    # With no cost report, the entire Generation Cost section is omitted
    assert "Generation Cost" not in md_unmatched
    assert "Official LLM Price Table" not in md_unmatched
    assert "Generation Cost Breakdown" not in md_unmatched

    assert "Generation Cost" not in html_unmatched
    assert "Official LLM Price Table" not in html_unmatched
    assert "Generation Cost Breakdown" not in html_unmatched
