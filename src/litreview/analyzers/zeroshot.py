"""Zero-shot classification analyzer for paper topic validation."""

from collections.abc import Iterable
from typing import ClassVar

import pandas as pd
from tqdm import tqdm

from litreview.analyzers.base import Analyzer
from litreview.config import ZeroShotConfig


class ZeroShotAnalyzer(Analyzer):
    """Classify papers into all configured labels that meet a score threshold.

    Scores from multiple configured models are averaged. The legacy ``label``,
    ``score``, and ``classified`` columns are retained alongside the complete
    ``labels``, ``scores``, and ``label_scores`` results.
    """

    _RESULT_COLUMNS: ClassVar[list[str]] = [
        "label",
        "score",
        "classified",
        "labels",
        "scores",
        "label_scores",
        "model_scores",
    ]

    def __init__(self, config: ZeroShotConfig):
        self.config = config
        self._is_fitted = False
        self._results: pd.DataFrame | None = None

    def fit(self, texts: pd.Series) -> "ZeroShotAnalyzer":
        """Validate configuration and prepare the analyzer."""
        if not self.config.models:
            raise ValueError("At least one zero-shot model must be configured")
        if len(self.config.models) != len(set(self.config.models)):
            raise ValueError("Zero-shot model names must be unique")
        labels = list(self.config.candidate_labels.values())
        if not labels:
            raise ValueError(
                "No candidate labels configured. "
                "Add labels under zeroshot.candidate_labels."
            )
        if len(labels) != len(set(labels)):
            raise ValueError("Zero-shot candidate label descriptions must be unique")
        self._is_fitted = True
        return self

    @staticmethod
    def _device_options() -> tuple[int, object]:
        """Select a supported inference device and numeric type."""
        import torch

        if torch.cuda.is_available():
            use_bf16 = torch.cuda.is_bf16_supported()
            return 0, torch.bfloat16 if use_bf16 else torch.float32
        return -1, torch.float32

    @staticmethod
    def _create_pipeline(model_name: str, device: int, dtype):
        from transformers import pipeline

        return pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _score_map(item: dict, labels: list[str]) -> dict[str, float]:
        returned_scores = {
            label: float(score) for label, score in zip(item["labels"], item["scores"])
        }
        missing = [label for label in labels if label not in returned_scores]
        if missing:
            raise RuntimeError(
                "Zero-shot model omitted candidate labels: " + ", ".join(missing)
            )
        return {label: returned_scores[label] for label in labels}

    def _classify_with_model(
        self,
        model_name: str,
        texts: list[str],
        labels: list[str],
        device: int,
        dtype,
    ) -> list[dict[str, float]]:
        classifier = self._create_pipeline(model_name, device, dtype)
        outputs: Iterable[dict] = classifier(
            iter(texts),
            candidate_labels=labels,
            multi_label=True,
            batch_size=self.config.batch_size,
            truncation=True,
            max_length=256,
        )
        scores = [
            self._score_map(item, labels)
            for item in tqdm(
                outputs,
                total=len(texts),
                desc=f"Zero-shot: {model_name}",
            )
        ]
        if len(scores) != len(texts):
            raise RuntimeError(
                f"Model '{model_name}' returned {len(scores)} results for "
                f"{len(texts)} texts"
            )
        del classifier
        if device >= 0:
            import torch

            torch.cuda.empty_cache()
        return scores

    def transform(self, texts: pd.Series) -> pd.DataFrame:
        """Classify texts and retain every label score above the threshold."""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        labels = list(self.config.candidate_labels.values())
        empty_mask = texts.isna() | texts.fillna("").astype(str).str.strip().eq("")
        non_empty_positions = [
            position
            for position, is_empty in enumerate(empty_mask.tolist())
            if not is_empty
        ]
        non_empty = texts.iloc[non_empty_positions].astype(str).tolist()

        per_model: dict[str, list[dict[str, float]]] = {}
        if non_empty:
            device, dtype = self._device_options()
            for model_name in self.config.models:
                per_model[model_name] = self._classify_with_model(
                    model_name, non_empty, labels, device, dtype
                )

        rows = [self._empty_result() for _ in range(len(texts))]
        for result_position, text_position in enumerate(non_empty_positions):
            model_scores = {
                model_name: scores[result_position]
                for model_name, scores in per_model.items()
            }
            averaged_scores = {
                label: sum(scores[label] for scores in model_scores.values())
                / len(model_scores)
                for label in labels
            }
            ranked_scores = sorted(
                averaged_scores.items(), key=lambda item: item[1], reverse=True
            )
            label_scores = {
                label: score
                for label, score in ranked_scores
                if score >= self.config.threshold
            }
            best_label, best_score = ranked_scores[0]
            rows[text_position] = {
                "label": best_label if label_scores else "unknown",
                "score": best_score,
                "classified": bool(label_scores),
                "labels": list(label_scores),
                "scores": averaged_scores,
                "label_scores": label_scores,
                "model_scores": model_scores,
            }

        self._results = pd.DataFrame(
            rows, index=texts.index, columns=self._RESULT_COLUMNS
        )
        return self._results

    @staticmethod
    def _empty_result() -> dict:
        return {
            "label": "unknown",
            "score": 0.0,
            "classified": False,
            "labels": [],
            "scores": {},
            "label_scores": {},
            "model_scores": {},
        }

    @property
    def results(self) -> dict:
        """Return classifications, multi-label counts, and the threshold."""
        label_counts = {label: 0 for label in self.config.candidate_labels.values()}
        if self._results is None:
            return {"label_counts": label_counts, "threshold": self.config.threshold}

        exploded = self._results["labels"].explode().dropna()
        label_counts.update(exploded.value_counts().to_dict())
        return {
            "classifications": self._results,
            "label_counts": label_counts,
            "threshold": self.config.threshold,
        }
