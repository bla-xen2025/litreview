"""Tests for the review pipeline and report exports."""

import json

import pandas as pd
import pytest

from litreview.pipeline import Report, ReviewPipeline


def make_report(sample_df, sample_config) -> Report:
    return Report(
        df=sample_df,
        corpus_stats={"total_papers": len(sample_df), "year_range": [2021, 2023]},
        topic_coverage={"topic_sizes": {0: 2, 1: 2}},
        gap_analysis={"gaps": [], "num_gaps": 0},
        cross_analysis={},
        bertopic_results={"topic_sizes": {0: 2, 1: 2}},
        zeroshot_results={"label_counts": {"alpha": 3, "beta": 2}},
        config=sample_config,
    )


def test_report_summary_and_exports(sample_df, sample_config, tmp_path):
    report = make_report(sample_df, sample_config)
    csv_path = tmp_path / "report.csv"
    json_path = tmp_path / "report.json"

    report.export_csv(str(csv_path))
    report.export_json(str(json_path))

    assert "Total papers: 5" in report.summary()
    assert len(pd.read_csv(csv_path)) == 5
    assert json.loads(json_path.read_text())["corpus_stats"]["total_papers"] == 5


def test_clean_preserves_text_and_deduplicates_by_doi_then_title():
    source = pd.DataFrame(
        {
            "Title": ["Original!", "Different", "Same Title", "same-title"],
            "Abstract Note": ["First; abstract.", "Second", "Third", "Fourth"],
            "DOI": ["https://doi.org/10.1/ABC", "doi:10.1/abc.", None, None],
        }
    )

    cleaned = ReviewPipeline._clean(source)

    assert cleaned["Title"].tolist() == ["Original!", "Same Title"]
    assert cleaned["Abstract Note"].tolist() == ["First; abstract.", "Third"]
    assert cleaned["Title Normalized"].tolist() == ["original", "same title"]
    assert "Title Normalized" not in source


def test_clean_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="Abstract Note"):
        ReviewPipeline._clean(pd.DataFrame({"Title": ["paper"]}))


def test_clean_keeps_untitled_rows_when_input_indices_repeat():
    source = pd.DataFrame(
        {"Title": [None, ""], "Abstract Note": ["First", "Second"]},
        index=[0, 0],
    )

    cleaned = ReviewPipeline._clean(source)

    assert cleaned["Abstract Note"].tolist() == ["First", "Second"]


def test_pipeline_uses_supplied_dataframe_without_fetching(
    sample_df, sample_config, monkeypatch
):
    pipeline = ReviewPipeline(sample_config, skip_bertopic=True)
    monkeypatch.setattr(
        pipeline.fetcher,
        "fetch",
        lambda *_: pytest.fail("Zotero must not be called for supplied data"),
    )
    monkeypatch.setattr(
        pipeline.zeroshot_analyzer, "_device_options", lambda: (-1, None)
    )
    monkeypatch.setattr(
        pipeline.zeroshot_analyzer,
        "_classify_with_model",
        lambda model_name, texts, labels, device, dtype: [
            {label: 0.9 if label == labels[0] else 0.1 for label in labels}
            for _ in texts
        ],
    )

    report = pipeline.run(df=sample_df)

    assert isinstance(report, Report)
    assert report.corpus_stats["total_papers"] == len(sample_df)
    assert report.df["Title"].tolist() == sample_df["Title"].tolist()
    assert report.df["classified"].all()


def test_pipeline_handles_empty_corpus(sample_config):
    pipeline = ReviewPipeline(sample_config, skip_bertopic=True)

    report = pipeline.run(df=pd.DataFrame(columns=["Title", "Abstract Note"]))

    assert report.df.empty
    assert report.corpus_stats["total_papers"] == 0
    assert "0.0%" in report.summary()
