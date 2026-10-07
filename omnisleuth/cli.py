"""Command-line interface (CLI) for running OmniSleuth investigations directly from the terminal."""

import argparse
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode/Emojis
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from omnisleuth.agents.orchestrator import MultiAgentOrchestrator
from omnisleuth.discovery.scanner import DatasetDiscoveryScanner
from omnisleuth.evaluation.dataset_generator import generate_benchmark_sales_dataset
from omnisleuth.ingestion.csv_loader import CSVLoader
from omnisleuth.persistence.repository import Repository
from omnisleuth.profiling.profiler import DatasetProfiler


def main():
    parser = argparse.ArgumentParser(
        prog="omnisleuth",
        description="OmniSleuth: Evidence-Grounded Multi-Agent Data Investigation System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Investigate command
    inv_parser = subparsers.add_parser("investigate", help="Run multi-agent investigation on a question")
    inv_parser.add_argument("question", type=str, help="Business question to investigate")
    inv_parser.add_argument("--csv", type=str, default=None, help="Path to CSV dataset (defaults to benchmark sample)")

    # 2. Discover command
    disc_parser = subparsers.add_parser("discover", help="Run automated statistical discovery scan on dataset")
    disc_parser.add_argument("--csv", type=str, default=None, help="Path to CSV dataset")

    # 3. Profile command
    prof_parser = subparsers.add_parser("profile", help="Generate data quality and distribution scorecard")
    prof_parser.add_argument("--csv", type=str, default=None, help="Path to CSV dataset")

    # 4. Generate dataset command
    gen_parser = subparsers.add_parser("generate-sample", help="Generate synthetic benchmark sales dataset")
    gen_parser.add_argument("--rows", type=int, default=1000, help="Number of rows")
    gen_parser.add_argument("--output", type=str, default="data/uploads/sample_sales_2024.csv", help="Output path")

    args = parser.parse_args()

    # Default to benchmark dataset if not specified
    csv_path = args.csv if hasattr(args, "csv") and args.csv else "data/uploads/sample_sales_2024.csv"
    path_obj = Path(csv_path)

    if not path_obj.exists():
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        print(f"Generating benchmark sample sales dataset at: {path_obj}")
        generate_benchmark_sales_dataset(n_rows=1000, random_seed=42, output_path=path_obj)

    loader = CSVLoader()
    df, metadata = loader.load_csv(path_obj)
    repo = Repository()
    repo.save_dataset(metadata)

    if args.command == "investigate" or (len(sys.argv) > 1 and not args.command):
        question = args.question if hasattr(args, "question") else sys.argv[1]
        print(f"\n🔍 Investigating: \"{question}\" on dataset '{metadata.name}' ({metadata.row_count:,} rows)\n")
        orchestrator = MultiAgentOrchestrator()
        result = orchestrator.run_investigation(question, df, metadata)
        print(result["final_answer"])

    elif args.command == "discover":
        print(f"\n⚡ Running Automated Discovery on '{metadata.name}'...\n")
        scanner = DatasetDiscoveryScanner()
        findings = scanner.scan(df, metadata)
        print(f"Discovered {len(findings)} noteworthy findings:\n")
        for i, f in enumerate(findings, 1):
            print(f"{i}. [{f.category.value.upper()}] {f.title} (Rank Score: {f.composite_rank_score:.2f})")
            print(f"   Recommended Question: \"{f.recommended_question}\"\n")

    elif args.command == "profile":
        print(f"\n📊 Profiling Dataset '{metadata.name}'...\n")
        profiler = DatasetProfiler()
        report = profiler.generate_report(df, metadata)
        print(f"Completeness Score: {report.completeness_score:.1f}%")
        print(f"Duplicate Rows: {report.duplicate_rows} ({report.duplicate_percentage:.1f}%)")
        print(f"Quality Warnings: {len(report.warnings)}")
        for w in report.warnings:
            print(f"  - [{w.severity.value.upper()}] {w.message}")

    elif args.command == "generate-sample":
        out = Path(args.output)
        generate_benchmark_sales_dataset(n_rows=args.rows, random_seed=42, output_path=out)
        print(f"Generated {args.rows} rows at: {out}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
