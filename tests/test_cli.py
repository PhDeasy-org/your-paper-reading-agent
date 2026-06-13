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

    with patch("ppagent.cli._load", return_value=mock_config), \
         patch("ppagent.pipeline.PaperPipeline") as mock_pipeline_cls, \
         patch("ppagent.hf.paper_info"), \
         patch("webbrowser.open") as mock_webbrowser_open:
         
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

    with patch("ppagent.cli._load", return_value=mock_config), \
         patch("ppagent.pipeline.PaperPipeline") as mock_pipeline_cls, \
         patch("ppagent.hf.paper_info"), \
         patch("webbrowser.open"):
         
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
    with patch("ppagent.cli.PROJECT_ROOT", tmp_path):
        result = runner.invoke(app, ["config", "init"])
        assert result.exit_code == 0
        assert (tmp_path / "config" / "settings.toml").exists()
        
        # Calling it again should print that it already exists
        result_again = runner.invoke(app, ["config", "init"])
        assert "Config already exists" in result_again.stdout

