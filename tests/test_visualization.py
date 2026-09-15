"""Tests for visualization inputs and generated files."""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from litreview.visualization import (
    plot_colabel_matrix,
    plot_confidence_distribution,
    plot_gap_analysis,
    plot_source_distribution,
    plot_topic_coverage,
    plot_year_distribution,
)


def test_plot_year_distribution(sample_df, tmp_path):
    path = tmp_path / "years.png"

    plot_year_distribution(sample_df, str(path))

    assert path.is_file()


def test_source_distribution_counts_every_paper(monkeypatch, tmp_path):
    source_df = pd.DataFrame({"Source": ["A", "A", "A", "B", "B"]})
    widths = []
    original_barh = Axes.barh

    def capture_barh(axis, y, width, *args, **kwargs):
        widths.extend(np.asarray(width).tolist())
        return original_barh(axis, y, width, *args, **kwargs)

    monkeypatch.setattr(Axes, "barh", capture_barh)

    plot_source_distribution(source_df, str(tmp_path / "sources.png"))

    assert sorted(widths) == [2, 3]


def test_plot_topic_coverage(tmp_path):
    path = tmp_path / "coverage.png"

    plot_topic_coverage({"topic_sizes": {0: 10, 1: 5}}, str(path))

    assert path.is_file()


def test_plot_gap_analysis(tmp_path):
    path = tmp_path / "gaps.png"
    gaps = {
        "gaps": [{"seed": "missing", "severity": "high", "papers": 0}],
        "num_gaps": 1,
    }

    plot_gap_analysis(gaps, str(path))

    assert path.is_file()


def test_plot_confidence_distribution(tmp_path):
    path = tmp_path / "confidence.png"
    results = {"classifications": pd.DataFrame({"score": [0.9, 0.6, 0.2]})}

    plot_confidence_distribution(results, str(path))

    assert path.is_file()


def test_colabel_matrix_contains_off_diagonal_cooccurrence(monkeypatch, tmp_path):
    matrices = []
    original_imshow = Axes.imshow

    def capture_imshow(axis, values, *args, **kwargs):
        matrices.append(np.asarray(values).copy())
        return original_imshow(axis, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", capture_imshow)
    results = {
        "classifications": pd.DataFrame(
            {
                "labels": [["alpha", "beta"], ["alpha"], ["beta"]],
                "label_scores": [
                    {"alpha": 0.9, "beta": 0.8},
                    {"alpha": 0.7},
                    {"beta": 0.6},
                ],
            }
        )
    }

    plot_colabel_matrix(results, str(tmp_path / "colabel.png"))

    assert matrices[0].tolist() == [[0, 1], [1, 0]]
