"""CLI entry point for the full analysis pipeline.

Usage:
    litreview-analysis --config config.yaml --fetch-zotero
    litreview-analysis --config config.yaml --fetch-zotero --output results/classified.csv --plots results/plots/
"""

import argparse
from pathlib import Path

import pandas as pd

from litreview import ReviewPipeline, load_config


def main():
    parser = argparse.ArgumentParser(
        description="Run literature review pipeline: fetch -> discover -> validate -> report"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config.yaml"), help="Config file path"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fetch-zotero", action="store_true", help="Fetch from Zotero")
    source.add_argument("--input", type=Path, help="Read papers from a local CSV file")
    parser.add_argument(
        "--output",
        default="data/processed/classified.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--plots",
        default="results/plots",
        help="Plots output directory",
    )
    parser.add_argument(
        "--skip-bertopic",
        action="store_true",
        help="Skip BERTopic discovery, run zero-shot only",
    )
    args = parser.parse_args()

    # Load config
    if not args.config.is_file():
        parser.error(f"Config file '{args.config}' not found")
    if args.input is not None and not args.input.is_file():
        parser.error(f"Input file '{args.input}' not found")

    config = load_config(args.config)
    input_df = pd.read_csv(args.input) if args.input is not None else None

    # Run pipeline
    pipeline = ReviewPipeline(config, skip_bertopic=args.skip_bertopic)
    report = pipeline.run(df=input_df)

    # Export results
    report.export_csv(args.output)
    report.generate_plots(args.plots)
    print(report.summary())


if __name__ == "__main__":
    main()
