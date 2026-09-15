"""Internal helpers shared by BERTopic result consumers."""

from collections.abc import Mapping

import pandas as pd


def normalize_topic_sizes(topic_sizes) -> dict[int, int]:
    """Convert BERTopic frequency output into a topic-to-count mapping."""
    if topic_sizes is None:
        return {}
    if isinstance(topic_sizes, pd.DataFrame):
        if topic_sizes.empty:
            return {}
        if {"Topic", "Count"}.issubset(topic_sizes.columns):
            return {
                int(topic): int(count)
                for topic, count in zip(topic_sizes["Topic"], topic_sizes["Count"])
            }
        if topic_sizes.shape[1] == 1:
            series = topic_sizes.iloc[:, 0]
            return {int(topic): int(count) for topic, count in series.items()}
        raise ValueError(
            "Topic frequency DataFrame must contain Topic and Count columns"
        )
    if isinstance(topic_sizes, pd.Series):
        return {int(topic): int(count) for topic, count in topic_sizes.items()}
    if isinstance(topic_sizes, Mapping):
        return {int(topic): int(count) for topic, count in topic_sizes.items()}
    raise TypeError("Unsupported topic frequency result type")
