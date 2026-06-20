from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner

from ppagent.cli import app
from ppagent.config import AppConfig
from ppagent.models import Paper, PaperReport, ReportSection


@pytest.fixture
def mock_config(tmp_path):
    cfg = AppConfig()
    cfg.root = tmp_path
    cfg.report.output_dir = str(tmp_path / "output")
    return cfg


def test_cli_report_multiple_papers(mock_config):
    runner = CliRunner()

    paper_1 = Paper(id="1111.1111", title="Paper One")
    paper_2 = Paper(id="2222.2222", title="Paper Two")

    report_1 = PaperReport(
        paper=paper_1,
        metadata=ReportSection(name="metadata", content="meta 1"),
        tldr=ReportSection(name="tldr", content="tldr 1"),
        method=ReportSection(name="method", content="method 1"),
        evaluation=ReportSection(name="evaluation", content="eval 1"),
        critique=ReportSection(name="critique", content="critique 1"),
        benchmarks=ReportSection(name="benchmarks", content="bench 1"),
        previous_works=ReportSection(name="previous_works", content="prev 1"),
        related_works=[],
    )

    report_2 = PaperReport(
        paper=paper_2,
        metadata=ReportSection(name="metadata", content="meta 2"),
        tldr=ReportSection(name="tldr", content="tldr 2"),
        method=ReportSection(name="method", content="method 2"),
        evaluation=ReportSection(name="evaluation", content="eval 2"),
        critique=ReportSection(name="critique", content="critique 2"),
        benchmarks=ReportSection(name="benchmarks", content="bench 2"),
        previous_works=ReportSection(name="previous_works", content="prev 2"),
        related_works=[],
    )

    with (
        patch("ppagent.cli._load", return_value=mock_config),
        patch("ppagent.pipeline.PaperPipeline") as mock_pipeline_cls,
        patch("ppagent.hf.paper_info"),
        patch("webbrowser.open") as mock_webbrowser_open,
    ):
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_pipeline.storage.report_exists.return_value = False

        def mock_report(paper_id):
            if paper_id == "1111.1111":
                report_dir = Path(mock_config.report.output_dir) / "paper_one"
                report_dir.mkdir(parents=True, exist_ok=True)
                (report_dir / "report.html").touch()
                return report_1
            elif paper_id == "2222.2222":
                report_dir = Path(mock_config.report.output_dir) / "paper_two"
                report_dir.mkdir(parents=True, exist_ok=True)
                (report_dir / "report.html").touch()
                return report_2
            raise ValueError("Unexpected paper_id")

        mock_pipeline.report.side_effect = mock_report

        result = runner.invoke(app, ["report", "1111.1111", "2222.2222", "--no-open"])

        assert result.exit_code == 0
        assert "Generating report for paper: 1111.1111" in result.stdout
        assert "Generating report for paper: 2222.2222" in result.stdout
        assert mock_pipeline.report.call_count == 2
        mock_webbrowser_open.assert_not_called()


def test_cli_report_one_failure_continues(mock_config):
    runner = CliRunner()

    paper_2 = Paper(id="2222.2222", title="Paper Two")
    report_2 = PaperReport(
        paper=paper_2,
        metadata=ReportSection(name="metadata", content="meta 2"),
        tldr=ReportSection(name="tldr", content="tldr 2"),
        method=ReportSection(name="method", content="method 2"),
        evaluation=ReportSection(name="evaluation", content="eval 2"),
        critique=ReportSection(name="critique", content="critique 2"),
        benchmarks=ReportSection(name="benchmarks", content="bench 2"),
        previous_works=ReportSection(name="previous_works", content="prev 2"),
        related_works=[],
    )

    with (
        patch("ppagent.cli._load", return_value=mock_config),
        patch("ppagent.pipeline.PaperPipeline") as mock_pipeline_cls,
        patch("ppagent.hf.paper_info"),
        patch("webbrowser.open"),
    ):
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_pipeline.storage.report_exists.return_value = False

        def mock_report(paper_id):
            if paper_id == "1111.1111":
                raise ValueError("HF Paper not found")
            elif paper_id == "2222.2222":
                report_dir = Path(mock_config.report.output_dir) / "paper_two"
                report_dir.mkdir(parents=True, exist_ok=True)
                (report_dir / "report.html").touch()
                return report_2
            raise ValueError("Unexpected paper_id")

        mock_pipeline.report.side_effect = mock_report

        result = runner.invoke(app, ["report", "1111.1111", "2222.2222", "--no-open"])

        assert result.exit_code == 1
        assert "Generating report for paper: 1111.1111" in result.stdout
        assert "Report generation failed for 1111.1111" in result.stdout
        assert "Generating report for paper: 2222.2222" in result.stdout
        assert "Report generated!" in result.stdout
        assert mock_pipeline.report.call_count == 2


def test_cli_config_show(mock_config):
    runner = CliRunner()
    with patch("ppagent.cli._load", return_value=mock_config):
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "Text LLM" in result.stdout
        assert "Vision LLM" in result.stdout
        assert "Searcher LLM" in result.stdout


def test_cli_config_init(tmp_path):
    runner = CliRunner()
    # Redirect the persistent backup too — config_init now seeds it, and we must
    # not clobber the developer's real ~/.config/ppagent/settings.toml.
    backup = tmp_path / "backup" / "settings.toml"
    with patch("ppagent.cli.PROJECT_ROOT", tmp_path), patch(
        "ppagent.cli._BACKUP_CONFIG_PATH", backup
    ), patch("ppagent.config._BACKUP_CONFIG_PATH", backup):
        result = runner.invoke(app, ["config", "init"])
        assert result.exit_code == 0
        assert (tmp_path / "config" / "settings.toml").exists()
        # A fresh init also seeds the persistent backup so reinstalls can restore.
        assert backup.exists()

        # Calling it again should print that it already exists
        result_again = runner.invoke(app, ["config", "init"])
        assert "Config already exists" in result_again.stdout


def test_paper_published_at_parsing():
    from datetime import datetime

    p1 = Paper(id="2604.12345", title="Test Paper 1")
    assert p1.published_at == datetime(2026, 4, 1)

    p2 = Paper(id="1706.03762", title="Test Paper 2")
    assert p2.published_at == datetime(2017, 6, 1)

    p3 = Paper(id="hep-th/9703012", title="Test Paper 3")
    assert p3.published_at == datetime(1997, 3, 1)

    # If already set, should not override
    p4 = Paper(
        id="2604.12345", title="Test Paper 4", published_at=datetime(2026, 4, 15)
    )
    assert p4.published_at == datetime(2026, 4, 15)


def test_fetch_arxiv_info_success():
    from ppagent.hf import fetch_arxiv_info

    mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Test Arxiv Paper</title>
    <published>2026-04-15T10:00:00Z</published>
    <author><name>Author One</name></author>
    <author><name>Author Two</name></author>
    <summary>This is a summary.</summary>
  </entry>
</feed>"""

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = mock_xml.encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        paper = fetch_arxiv_info("2604.12345")
        assert paper is not None
        assert paper.title == "Test Arxiv Paper"
        assert paper.id == "2604.12345"
        assert paper.authors == ["Author One", "Author Two"]
        assert paper.summary == "This is a summary."


def test_pipeline_arxiv_fallback_integration(mock_config):
    from ppagent.pipeline import PaperPipeline
    from ppagent.hf import HfCliError

    mock_config.llms.text.api_key = "dummy"
    mock_config.llms.vision.api_key = "dummy"
    mock_config.llms.searcher.api_key = "dummy"

    paper_id = "2606.01075"

    with (
        patch("ppagent.hf.paper_info", side_effect=HfCliError("not found")),
        patch("ppagent.hf.fetch_arxiv_info") as mock_fetch_arxiv,
        patch("ppagent.hf.paper_read", return_value="some markdown content"),
        patch("ppagent.pdf.download_pdf"),
        patch("ppagent.pdf.extract_text"),
        patch("ppagent.agents.classifier.ClassifierAgent.run") as mock_classifier,
        patch("ppagent.agents.writer.WriterAgent.run") as mock_writer,
        patch("ppagent.agents.finder.FinderAgent.run") as mock_finder,
        patch("ppagent.agents.criticizer.CriticizerAgent.run") as mock_criticizer,
    ):
        from ppagent.models import AgentResult, Paper

        mock_paper = Paper(id=paper_id, title="Attention Is All You Need")
        mock_fetch_arxiv.return_value = mock_paper

        # Mock agents
        mock_classifier.return_value = AgentResult(
            agent_name="classifier",
            success=True,
            data={"paper_type": "method", "confidence": 0.95},
        )
        mock_writer.return_value = AgentResult(
            agent_name="writer", success=True, data={}
        )
        mock_finder.return_value = AgentResult(
            agent_name="finder", success=True, data={}
        )
        mock_criticizer.return_value = AgentResult(
            agent_name="criticizer", success=True, data={}
        )

        pipeline = PaperPipeline(mock_config)
        report = pipeline.report(paper_id)

        assert report.paper.title == "Attention Is All You Need"
        assert report.paper.published_at is not None
        mock_fetch_arxiv.assert_called_once_with(paper_id)
