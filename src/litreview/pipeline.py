"""ReviewPipeline and Report classes.

Orchestrates the full literature review workflow:
fetch -> clean -> BERTopic discovery -> zero-shot validation -> statistics -> report
"""

import json
import os
import re

import pandas as pd

from litreview.analyzers.bertopic import (
    BERTopicFitter,
    TopicDistributionAnalyzer,
    TopicRepresentativeDocs,
    TopicWordExtractor,
)
from litreview.analyzers.bertopic._utils import normalize_topic_sizes
from litreview.analyzers.zeroshot import ZeroShotAnalyzer
from litreview.config import PipelineConfig
from litreview.fetchers.zotero import ZoteroFetcher
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


class ReviewPipeline:
    """Orchestrates the full literature review workflow.

    Usage:
        config = load_config("config.yaml")
        pipeline = ReviewPipeline(config)
        report = pipeline.run()
        report.export_csv("data/processed/classified.csv")
        report.generate_plots("results/plots/")
        print(report.summary())
    """

    def __init__(self, config: PipelineConfig, skip_bertopic: bool = False):
        self.config = config
        self.skip_bertopic = skip_bertopic
        self.fetcher = ZoteroFetcher(
            config.zotero.library_id,
            config.zotero.api_key,
            config.zotero.library_type,
        )
        self.zeroshot_analyzer = ZeroShotAnalyzer(config.zeroshot)

    def run(self, df: pd.DataFrame | None = None) -> "Report":
        """Execute the full pipeline and return a Report.

        Args:
            df: Optional input corpus. When omitted, papers are fetched from Zotero.

        Returns:
            Report with DataFrame, statistics, and analysis results.
        """
        # 1. Fetch unless a caller supplied an input corpus.
        if df is None:
            df = self.fetcher.fetch(self.config.zotero.collection_names)

        # 2. Clean
        df = self._clean(df)

        abstracts = df["Abstract Note"]
        abstract_mask = abstracts.fillna("").astype(str).str.strip().ne("")
        valid_abstracts = abstracts[abstract_mask]

        # 3. BERTopic — fit model, then run configured analyses
        bertopic_results = {}
        topic_results = pd.DataFrame(index=df.index)
        bertopic_model = None
        bertopic_ran = False

        if (
            not self.skip_bertopic
            and self.config.bertopic.compute
            and len(valid_abstracts) > 0
        ):
            # BERTopic needs a minimum number of documents to produce
            # meaningful clusters (UMAP needs n_neighbors >= 2, HDBSCAN
            # needs at least a few points per cluster).
            if len(valid_abstracts) < 10:
                total_papers = len(df)
                non_empty = len(valid_abstracts)
                raise ValueError(
                    f"Only {non_empty} of {total_papers} papers have abstracts. "
                    "BERTopic requires at least 10 documents with non-empty abstracts. "
                    "Options:\n"
                    "  1. Import abstracts into Zotero (item details → Notes → Abstract)\n"
                    "  2. Use --skip-bertopic to run zero-shot classification only\n"
                    "  3. Set bertopic.compute: false in config to skip entirely"
                )
            # Phase 1: Fit the model (always runs when compute=True)
            fitter = BERTopicFitter(self.config.bertopic)
            fitter.fit(valid_abstracts)
            bertopic_model = fitter.topic_model
            bertopic_results = {
                k: v for k, v in fitter.results.items() if k != "topic_model"
            }
            topic_results = pd.DataFrame({"topic": -1}, index=df.index)
            topic_results.loc[valid_abstracts.index, "topic"] = fitter.topic_assignments
            bertopic_results["topic_assignments"] = topic_results["topic"].to_numpy()
            bertopic_ran = True

            # Phase 2: Run configured analyses, each receiving the fitted model
            analysis_map = {
                "distribution": TopicDistributionAnalyzer,
                "words": TopicWordExtractor,
                "representatives": TopicRepresentativeDocs,
            }
            for analysis_name in self.config.bertopic.analyses:
                analyzer_cls = analysis_map.get(analysis_name)
                if analyzer_cls is None:
                    continue
                if analysis_name == "distribution":
                    analyzer = analyzer_cls(
                        bertopic_model,
                        self.config.bertopic,
                        top_n_topics=self.config.bertopic.top_n_topics,
                        window=self.config.bertopic.distribution_window,
                        stride=self.config.bertopic.distribution_stride,
                    )
                else:
                    analyzer = analyzer_cls(bertopic_model)
                analysis_texts = (
                    abstracts if analysis_name == "distribution" else valid_abstracts
                )
                analyzer.fit(analysis_texts)
                analysis_df = analyzer.transform(analysis_texts)
                topic_results = pd.concat([topic_results, analysis_df], axis=1)
                bertopic_results.update(analyzer.results)

        df = pd.concat([df, topic_results], axis=1)

        # 4. Validate topics (Zero-shot)
        self.zeroshot_analyzer.fit(abstracts)
        validation_results = self.zeroshot_analyzer.transform(abstracts)
        df = pd.concat([df, validation_results], axis=1)

        # 5. Compute statistics
        corpus_stats = compute_corpus_stats(df)
        if not bertopic_ran:
            topic_coverage = {"coverage": {}, "num_topics": 0}
            gap_analysis = {"gaps": [], "num_gaps": 0}
            cross_analysis = {"agreement": 0.0}
        else:
            topic_coverage = compute_topic_coverage(
                topic_results, validation_results, self.config.bertopic.seed_topics
            )
            gap_analysis = compute_gap_analysis(
                topic_results, validation_results, self.config.bertopic.seed_topics
            )
            cross_analysis = compute_cross_analysis(topic_results, validation_results)

        # 6. Return report
        return Report(
            df=df,
            corpus_stats=corpus_stats,
            topic_coverage=topic_coverage,
            gap_analysis=gap_analysis,
            cross_analysis=cross_analysis,
            bertopic_results=bertopic_results,
            zeroshot_results=self.zeroshot_analyzer.results,
            config=self.config,
        )

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Validate the corpus and remove duplicates without altering source text."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Pipeline input must be a pandas DataFrame")

        required_columns = {"Title", "Abstract Note"}
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            raise ValueError(
                "Input data is missing required columns: " + ", ".join(missing_columns)
            )

        df = df.copy()
        optional_defaults = {
            "Publication Year": pd.NA,
            "DOI": pd.NA,
            "Source": "CSV",
            "Item Type": "unknown",
        }
        for column, default in optional_defaults.items():
            if column not in df:
                df[column] = default

        df["Title Normalized"] = df["Title"].apply(ReviewPipeline._normalize_text)
        normalized_doi = df["DOI"].apply(ReviewPipeline._normalize_doi)
        dedupe_key = normalized_doi.where(
            normalized_doi.ne(""), "title:" + df["Title Normalized"]
        )
        empty_key = dedupe_key.eq("title:")
        dedupe_key.loc[empty_key] = [
            f"untitled-row:{position}" for position in range(int(empty_key.sum()))
        ]
        df = df.loc[~dedupe_key.duplicated(keep="first")]

        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def _normalize_text(text):
        """Normalize text: remove non-word chars, lowercase, strip."""
        if pd.isna(text):
            return ""
        return re.sub(r"\W+", " ", str(text)).strip().lower()

    @staticmethod
    def _normalize_doi(doi) -> str:
        """Normalize common DOI representations for identity matching."""
        if pd.isna(doi):
            return ""
        value = str(doi).strip().lower()
        value = re.sub(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)", "", value)
        return value.rstrip(".,; ")


class Report:
    """Container for pipeline results with export methods."""

    def __init__(
        self,
        df: pd.DataFrame,
        corpus_stats: dict,
        topic_coverage: dict,
        gap_analysis: dict,
        cross_analysis: dict,
        bertopic_results: dict,
        zeroshot_results: dict,
        config: PipelineConfig,
    ):
        self.df = df
        self.corpus_stats = corpus_stats
        self.topic_coverage = topic_coverage
        self.gap_analysis = gap_analysis
        self.cross_analysis = cross_analysis
        self.bertopic_results = bertopic_results
        self.zeroshot_results = zeroshot_results
        self.config = config

    def export_csv(self, path: str) -> None:
        """Export full DataFrame to CSV."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.df.to_csv(path, index=False)

    def export_json(self, path: str) -> None:
        """Export statistics summary to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        summary = {
            "corpus_stats": self.corpus_stats,
            "topic_coverage": self.topic_coverage,
            "gap_analysis": self.gap_analysis,
            "cross_analysis": self.cross_analysis,
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def generate_plots(self, path: str) -> None:
        """Generate all plots to directory."""
        os.makedirs(path, exist_ok=True)

        # Corpus overview
        plot_year_distribution(self.df, os.path.join(path, "year_distribution.png"))

        # Zero-shot results
        plot_zeroshot_label_counts(
            self.zeroshot_results, os.path.join(path, "zeroshot_label_counts.png")
        )
        plot_zeroshot_confidence_by_label(
            self.zeroshot_results,
            os.path.join(path, "zeroshot_confidence_by_label.png"),
        )
        plot_confidence_distribution(
            self.zeroshot_results, os.path.join(path, "confidence_distribution.png")
        )

        # BERTopic results
        plot_bertopic_sizes(
            self.bertopic_results, os.path.join(path, "bertopic_sizes.png")
        )
        plot_bertopic_topic_words(
            self.bertopic_results, os.path.join(path, "bertopic_topic_words.png")
        )
        plot_bertopic_representative_docs(
            self.bertopic_results,
            os.path.join(path, "bertopic_representative_docs.png"),
        )

        # Cross-analysis
        plot_topic_coverage(
            self.topic_coverage, os.path.join(path, "topic_coverage.png")
        )
        plot_gap_analysis(self.gap_analysis, os.path.join(path, "gap_analysis.png"))

        # Topic × Label alignment
        plot_topic_label_heatmap(
            self.bertopic_results,
            self.zeroshot_results,
            os.path.join(path, "topic_label_heatmap.png"),
        )

        # Label co-occurrence
        plot_colabel_matrix(
            self.zeroshot_results, os.path.join(path, "colabel_matrix.png")
        )

        # New: Topic size vs confidence
        plot_topic_confidence_scatter(
            self.bertopic_results,
            self.zeroshot_results,
            os.path.join(path, "topic_confidence_scatter.png"),
        )

        # New: Method overlap
        plot_method_overlap(
            self.cross_analysis, os.path.join(path, "method_overlap.png")
        )

        # New: Topic network
        plot_topic_network(
            self.bertopic_results,
            self.zeroshot_results,
            os.path.join(path, "topic_network.png"),
        )

        # New: Per-topic label distribution
        plot_topic_label_distribution(
            self.bertopic_results,
            self.zeroshot_results,
            os.path.join(path, "topic_label_distribution.png"),
        )

        # Topic distribution by year (soft assignments)
        plot_topic_distribution_by_year(
            self.bertopic_results,
            self.df,
            os.path.join(path, "topic_distribution_by_year.png"),
        )

    def summary(self) -> str:
        """Return human-readable summary string."""
        lines = [
            "=== LitReview Report ===",
            f"Total papers: {self.corpus_stats['total_papers']}",
            f"Year range: {self.corpus_stats['year_range']}",
            "",
            "--- BERTopic ---",
            f"Topics discovered: {self.bertopic_results.get('num_topics', 0)}",
            f"Outlier papers: {self.bertopic_results.get('outlier_count', 0)}",
        ]

        topic_sizes = normalize_topic_sizes(
            self.bertopic_results.get("topic_sizes", {})
        )

        if topic_sizes:
            non_outlier = {k: v for k, v in topic_sizes.items() if k != -1}
            if non_outlier:
                largest = max(non_outlier, key=non_outlier.get)
                smallest = min(non_outlier, key=non_outlier.get)
                lines.append(
                    f"Largest topic: T{largest} ({non_outlier[largest]} papers)"
                )
                lines.append(
                    f"Smallest topic: T{smallest} ({non_outlier[smallest]} papers)"
                )

        lines.append("")
        lines.append("--- Zero-Shot Classification ---")
        label_counts = self.zeroshot_results.get("label_counts", {})
        if label_counts:
            sorted_labels = sorted(
                label_counts.items(), key=lambda x: x[1], reverse=True
            )
            total_papers = self.corpus_stats["total_papers"]
            for label, count in sorted_labels:
                pct = count / total_papers * 100 if total_papers else 0.0
                lines.append(f"  {label}: {count} papers ({pct:.1f}%)")

        if self.gap_analysis.get("gaps"):
            lines.append("")
            lines.append(f"--- Gaps ({self.gap_analysis['num_gaps']}) ---")
            for gap in self.gap_analysis["gaps"]:
                severity = gap.get("severity", "unknown")
                desc = gap.get("description", gap.get("seed", "unknown"))
                lines.append(f"  [{severity.upper()}] {desc}")

        lines.append("")
        lines.append("=== End Report ===")
        return "\n".join(lines)
