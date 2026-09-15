"""Tests for analyzer contracts and result alignment."""

import numpy as np
import pandas as pd
import pytest

from litreview.analyzers.base import Analyzer
from litreview.analyzers.bertopic._utils import normalize_topic_sizes
from litreview.analyzers.bertopic.distribution import TopicDistributionAnalyzer
from litreview.analyzers.bertopic.representatives import TopicRepresentativeDocs
from litreview.analyzers.zeroshot import ZeroShotAnalyzer
from litreview.config import BERTopicConfig, ZeroShotConfig


def test_analyzer_is_abstract():
    with pytest.raises(TypeError):
        Analyzer()


def test_topic_frequency_dataframe_is_normalized():
    frequencies = pd.DataFrame({"Topic": [-1, 0, 1], "Count": [2, 5, 3]})

    assert normalize_topic_sizes(frequencies) == {-1: 2, 0: 5, 1: 3}


def test_representatives_use_public_bertopic_api():
    class TopicModel:
        def get_topic_freq(self):
            return pd.DataFrame({"Topic": [-1, 0], "Count": [1, 2]})

        def get_representative_docs(self, topic):
            assert topic == 0
            return ["first", "second", "third"]

    results = TopicRepresentativeDocs(TopicModel(), n_documents=2).results

    assert results == {"topic_representatives": {0: ["first", "second"]}}


def test_distribution_preserves_positions_for_missing_texts():
    class TopicModel:
        def approximate_distribution(self, texts, **kwargs):
            assert texts == ["first", "third"]
            return np.array([[0.9, 0.1], [0.2, 0.8]]), None

    texts = pd.Series(["first", None, "third"], index=[10, 11, 12])
    analyzer = TopicDistributionAnalyzer(TopicModel(), BERTopicConfig(), top_n_topics=1)

    result = analyzer.fit_transform(texts)

    assert result["dominant_topic"].to_dict() == {10: 0, 11: -1, 12: 1}


def test_zeroshot_preserves_all_threshold_scores_and_averages_models(monkeypatch):
    config = ZeroShotConfig(
        models=["model-a", "model-b"],
        threshold=0.5,
        candidate_labels={"A": "alpha", "B": "beta"},
    )
    analyzer = ZeroShotAnalyzer(config)

    model_results = {
        "model-a": [
            {"alpha": 0.9, "beta": 0.6},
            {"alpha": 0.4, "beta": 0.3},
        ],
        "model-b": [
            {"alpha": 0.7, "beta": 0.8},
            {"alpha": 0.8, "beta": 0.2},
        ],
    }

    monkeypatch.setattr(analyzer, "_device_options", lambda: (-1, None))
    monkeypatch.setattr(
        analyzer,
        "_classify_with_model",
        lambda model_name, texts, labels, device, dtype: model_results[model_name],
    )

    result = analyzer.fit_transform(pd.Series(["one", "two"]))

    assert result.loc[0, "labels"] == ["alpha", "beta"]
    assert result.loc[0, "label_scores"] == pytest.approx({"alpha": 0.8, "beta": 0.7})
    assert result.loc[1, "labels"] == ["alpha"]
    assert analyzer.results["label_counts"] == {"alpha": 2, "beta": 1}


def test_zeroshot_handles_empty_and_missing_text_without_loading_model(monkeypatch):
    analyzer = ZeroShotAnalyzer(
        ZeroShotConfig(models=["model"], candidate_labels={"A": "alpha"})
    )
    monkeypatch.setattr(
        analyzer,
        "_device_options",
        lambda: pytest.fail("empty input must not initialize a model"),
    )

    result = analyzer.fit_transform(pd.Series([None, ""], index=[3, 4]))

    assert list(result.index) == [3, 4]
    assert result["classified"].tolist() == [False, False]
    assert result["labels"].tolist() == [[], []]
