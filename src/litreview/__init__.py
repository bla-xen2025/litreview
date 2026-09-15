"""LitReview - Corpus characterization through topic discovery and validation."""

__version__ = "0.2.0"

from litreview.analyzers.bertopic_wrapper import BERTopicAnalyzer
from litreview.analyzers.zeroshot import ZeroShotAnalyzer
from litreview.config import PipelineConfig, load_config
from litreview.fetchers.zotero import ZoteroFetcher
from litreview.pipeline import Report, ReviewPipeline
from litreview.statistics import (
    compute_corpus_stats,
    compute_cross_analysis,
    compute_gap_analysis,
    compute_topic_coverage,
)
from litreview.visualization import (
    plot_bertopic_representative_docs,
    plot_bertopic_sizes,
    plot_bertopic_topic_words,
    plot_colabel_matrix,
    plot_confidence_distribution,
    plot_gap_analysis,
    plot_method_overlap,
    plot_topic_confidence_scatter,
    plot_topic_coverage,
    plot_topic_distribution_by_year,
    plot_topic_label_distribution,
    plot_topic_label_heatmap,
    plot_topic_network,
    plot_year_distribution,
    plot_zeroshot_confidence_by_label,
    plot_zeroshot_label_counts,
)

__all__ = [
    "BERTopicAnalyzer",
    "PipelineConfig",
    "Report",
    "ReviewPipeline",
    "ZeroShotAnalyzer",
    "ZoteroFetcher",
    "__version__",
    "compute_corpus_stats",
    "compute_cross_analysis",
    "compute_gap_analysis",
    "compute_topic_coverage",
    "load_config",
    "plot_bertopic_representative_docs",
    "plot_bertopic_sizes",
    "plot_bertopic_topic_words",
    "plot_colabel_matrix",
    "plot_confidence_distribution",
    "plot_gap_analysis",
    "plot_method_overlap",
    "plot_topic_confidence_scatter",
    "plot_topic_coverage",
    "plot_topic_distribution_by_year",
    "plot_topic_label_distribution",
    "plot_topic_label_heatmap",
    "plot_topic_network",
    "plot_year_distribution",
    "plot_zeroshot_confidence_by_label",
    "plot_zeroshot_label_counts",
]
