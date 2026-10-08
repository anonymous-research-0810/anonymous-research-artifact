"""Main for the paper option-hedging pipeline."""

from __future__ import annotations
import argparse
from .adapter import ensure_sx5e_data_root


def main() -> int:
    """Parse command-line arguments, run the requested operation, and report failures."""
    parser = argparse.ArgumentParser(
        description="Convert SX5E Parquet files to the shared option-data schema"
    )
    parser.add_argument("--data-root", default="data_sx5e", help="Raw SX5E data directory")
    parser.add_argument("--output-root", default=None, help="Canonical output directory")
    parser.add_argument(
        "--force", action="store_true", help="Ignore the cached manifest and convert again"
    )
    args = parser.parse_args()
    output = ensure_sx5e_data_root(
        args.data_root, output_root=args.output_root, is_force=args.force
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
