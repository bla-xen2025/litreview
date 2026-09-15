"""Corpus characterization statistics.

Provides metrics for understanding the corpus, topic coverage,
research gaps, and cross-method agreement.
"""

import pandas as pd


def compute_corpus_stats(df: pd.DataFrame) -> dict:
    """Compute corpus-level statistics.

    Args:
        df: DataFrame with columns: Publication Year, Source, Item Type,
            Abstract Note, Title.

    Returns:
        Dict with total_papers, papers_with_abstracts, year_range,
        sources, item_types.
    """
    years = pd.to_numeric(
        df.get("Publication Year", pd.Series(index=df.index, dtype=float)),
        errors="coerce",
    ).dropna()
    abstracts = df.get("Abstract Note", pd.Series(index=df.index, dtype=object))
    has_abstract = abstracts.fillna("").astype(str).str.strip().ne("")
    sources = df.get("Source", pd.Series(index=df.index, dtype=object))
    item_types = df.get("Item Type", pd.Series(index=df.index, dtype=object))
    return {
        "total_papers": len(df),
        "papers_with_abstracts": int(has_abstract.sum()),
        "year_range": [int(years.min()), int(years.max())]
        if len(years) > 0
        else [None, None],
        "sources": sources.value_counts().to_dict(),
        "item_types": item_types.value_counts().to_dict(),
    }


def compute_topic_coverage(
    topic_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    seed_topics: list[list[str]] | None = None,
) -> dict:
    """Compute how many papers discuss each topic.

    Args:
        topic_df: DataFrame from BERTopicAnalyzer with 'topic' column.
        validation_df: DataFrame from ZeroShotAnalyzer with 'score', 'label',
            'classified' columns.
        seed_topics: Optional list of seed topic groups.

    Returns:
        Dict with topic_sizes, outlier_count, num_topics, total_classified,
        mean_confidence, median_confidence.
    """
    topics = topic_df.get("topic", pd.Series(index=topic_df.index, dtype=int))
    topic_sizes = topics.value_counts().to_dict()
    outlier_count = topic_sizes.get(-1, 0)
    topic_sizes_clean = {k: v for k, v in topic_sizes.items() if k != -1}

    scores = validation_df["score"]
    classified = validation_df["classified"].sum()

    return {
        "topic_sizes": topic_sizes_clean,
        "outlier_count": outlier_count,
        "num_topics": len(topic_sizes_clean),
        "total_classified": int(classified),
        "mean_confidence": float(scores.mean()),
        "median_confidence": float(scores.median()),
    }


def compute_gap_analysis(
    topic_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    seed_topics: list[list[str]] | None = None,
) -> dict:
    """Identify gaps: topics with few/no papers, seeds with no matches.

    Args:
        topic_df: DataFrame from BERTopicAnalyzer with 'topic' column.
        validation_df: DataFrame from ZeroShotAnalyzer with 'label' column.
        seed_topics: Optional list of seed topic groups.

    Returns:
        Dict with gaps list and num_gaps count.
    """
    gaps = []

    # Check seed topics against zero-shot results
    if seed_topics:
        if "labels" in validation_df:
            selected_labels = validation_df["labels"].explode().dropna()
        else:
            classified = validation_df.get(
                "classified", pd.Series(True, index=validation_df.index)
            )
            selected_labels = validation_df.loc[classified, "label"]
        labels = selected_labels.value_counts()
        for seed_group in seed_topics:
            seed_label = seed_group[0] if isinstance(seed_group, list) else seed_group
            count = labels.get(seed_label, 0)
            if count == 0:
                gaps.append(
                    {
                        "type": "seed_no_matches",
                        "seed": seed_label,
                        "papers": 0,
                        "severity": "high",
                    }
                )
            elif count < 5:
                gaps.append(
                    {
                        "type": "seed_few_matches",
                        "seed": seed_label,
                        "papers": int(count),
                        "severity": "medium",
                    }
                )

    # Check for large outlier groups
    topics = topic_df.get("topic", pd.Series(index=topic_df.index, dtype=int))
    outlier_count = int((topics == -1).sum())
    total = len(topic_df)
    if total > 0 and outlier_count > total * 0.3:
        gaps.append(
            {
                "type": "large_outliers",
                "description": (
                    f"{outlier_count} papers ({outlier_count / total * 100:.0f}%) "
                    "don't match any topic"
                ),
                "severity": "medium",
            }
        )

    return {
        "gaps": gaps,
        "num_gaps": len(gaps),
    }


def compute_cross_analysis(
    topic_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> dict:
    """Compare BERTopic and zero-shot results for agreement.

    Args:
        topic_df: DataFrame from BERTopicAnalyzer with 'topic' column.
        validation_df: DataFrame from ZeroShotAnalyzer with 'classified' column.

    Returns:
        Dict with bertopic_classified, zeroshot_classified,
        both_classified, neither_classified.
    """
    topics = topic_df.get("topic", pd.Series(-1, index=validation_df.index))
    if len(topics) != len(validation_df):
        raise ValueError("Topic and validation results must have equal length")
    topic_classified = topics.to_numpy() != -1
    zeroshot_classified = validation_df["classified"].to_numpy(dtype=bool)

    return {
        "bertopic_classified": int(topic_classified.sum()),
        "zeroshot_classified": int(zeroshot_classified.sum()),
        "both_classified": int((topic_classified & zeroshot_classified).sum()),
        "neither_classified": int((~topic_classified & ~zeroshot_classified).sum()),
    }
