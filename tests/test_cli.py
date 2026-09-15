"""Tests for analysis CLI input source selection."""

import sys

import pandas as pd
import pytest

from litreview.cli import run_analysis


class FakeReport:
    def export_csv(self, path):
        self.csv_path = path

    def generate_plots(self, path):
        self.plot_path = path

    def summary(self):
        return "summary"


def write_config(path):
    path.write_text(
        "bertopic: {compute: false}\n"
        "zeroshot:\n"
        "  models: [test-model]\n"
        "  candidate_labels: {A: alpha}\n"
    )


def test_cli_reads_local_csv(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    input_path = tmp_path / "papers.csv"
    write_config(config_path)
    pd.DataFrame({"Title": ["paper"], "Abstract Note": ["abstract"]}).to_csv(
        input_path, index=False
    )
    captured = {}

    class FakePipeline:
        def __init__(self, config, skip_bertopic=False):
            captured["skip_bertopic"] = skip_bertopic

        def run(self, df=None):
            captured["df"] = df
            return FakeReport()

    monkeypatch.setattr(run_analysis, "ReviewPipeline", FakePipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "litreview-analysis",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--skip-bertopic",
        ],
    )

    run_analysis.main()

    assert captured["df"]["Title"].tolist() == ["paper"]
    assert captured["skip_bertopic"] is True


def test_cli_fetches_when_zotero_source_selected(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    captured = {}

    class FakePipeline:
        def __init__(self, config, skip_bertopic=False):
            pass

        def run(self, df=None):
            captured["df"] = df
            return FakeReport()

    monkeypatch.setattr(run_analysis, "ReviewPipeline", FakePipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "litreview-analysis",
            "--config",
            str(config_path),
            "--fetch-zotero",
        ],
    )

    run_analysis.main()

    assert captured["df"] is None


def test_cli_rejects_multiple_input_sources(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    input_path = tmp_path / "papers.csv"
    write_config(config_path)
    input_path.write_text("Title,Abstract Note\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "litreview-analysis",
            "--config",
            str(config_path),
            "--fetch-zotero",
            "--input",
            str(input_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_analysis.main()

    assert exc_info.value.code == 2
