"""Tests for corpus and cross-analysis statistics."""

import pandas as pd

from litreview.statistics import (
    compute_corpus_stats,
    compute_cross_analysis,
    compute_gap_analysis,
    compute_topic_coverage,
)


def validation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": ["alpha", "alpha", "unknown", "beta", "unknown"],
            "score": [0.9, 0.8, 0.4, 0.7, 0.2],
            "classified": [True, True, False, True, False],
            "labels": [["alpha", "beta"], ["alpha"], [], ["beta"], []],
        }
    )


def test_compute_corpus_stats_counts_only_nonempty_abstracts(sample_df):
    sample_df.loc[0, "Abstract Note"] = ""
    sample_df.loc[1, "Abstract Note"] = None

    stats = compute_corpus_stats(sample_df)

    assert stats["total_papers"] == 5
    assert stats["papers_with_abstracts"] == 3
    assert stats["year_range"] == [2021, 2023]


def test_compute_corpus_stats_accepts_empty_minimal_frame():
    stats = compute_corpus_stats(pd.DataFrame(columns=["Title", "Abstract Note"]))

    assert stats["total_papers"] == 0
    assert stats["year_range"] == [None, None]


def test_compute_topic_coverage():
    topics = pd.DataFrame({"topic": [0, 0, 1, 1, -1]})

    coverage = compute_topic_coverage(topics, validation_frame())

    assert coverage["topic_sizes"] == {0: 2, 1: 2}
    assert coverage["outlier_count"] == 1
    assert coverage["total_classified"] == 3


def test_gap_analysis_uses_only_threshold_qualified_labels():
    topics = pd.DataFrame({"topic": [0, 0, 1, 1, -1]})

    gaps = compute_gap_analysis(
        topics, validation_frame(), seed_topics=[["alpha"], ["missing"]]
    )

    gaps_by_seed = {gap.get("seed"): gap for gap in gaps["gaps"] if "seed" in gap}
    assert gaps_by_seed["alpha"]["papers"] == 2
    assert gaps_by_seed["missing"]["papers"] == 0


def test_compute_cross_analysis():
    topics = pd.DataFrame({"topic": [0, 0, 1, -1, -1]})

    cross = compute_cross_analysis(topics, validation_frame())

    assert cross == {
        "bertopic_classified": 3,
        "zeroshot_classified": 3,
        "both_classified": 2,
        "neither_classified": 1,
    }
