"""Plot a selected Stage 1 result for either market."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from _segmentation_plot import make_figure
from _segmentation_statistics_plot import make_statistics_figure


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    market = payload["data_source"]
    if market not in {"spx", "sx5e"}:
        raise ValueError("The result must specify spx or sx5e")
    path = str(args.result.resolve())
    print(make_figure(market, path, output_dir=args.output_dir))
    print(make_statistics_figure(market, path, output_dir=args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
