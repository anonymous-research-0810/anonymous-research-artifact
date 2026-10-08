"""Test result loading for the paper option-hedging pipeline."""

from __future__ import annotations
import json
from market_segmentation import get_seg_periods_from_result, resolve_seg_result_path
from market_simulation import resolve_simulation_result_directory


def _write_seg_result(path, *, start: str, end: str, status: str = "completed"):
    """Write seg result."""
    path.write_text(
        json.dumps(
            {"schema_version": 1, "status": status, "num_segment": 1, "seg_periods": [[start, end]]}
        ),
        encoding="utf-8",
    )


def test_latest_completed_segmentation_result_uses_filename_timestamp(tmp_path) -> None:
    """Verify latest completed segmentation result uses filename timestamp."""
    older = tmp_path / "seg_20200101T000000Z.json"
    latest = tmp_path / "seg_20200103T000000Z.json"
    failed = tmp_path / "seg_20200104T000000Z.json"
    _write_seg_result(older, start="2020-01-01", end="2020-01-02")
    _write_seg_result(latest, start="2020-01-03", end="2020-01-04")
    _write_seg_result(failed, start="2020-01-05", end="2020-01-06", status="failed")
    assert resolve_seg_result_path(result_root=tmp_path) == latest
    assert get_seg_periods_from_result(result_root=tmp_path) == [("2020-01-03", "2020-01-04")]
    assert get_seg_periods_from_result(older.name, result_root=tmp_path) == [
        ("2020-01-01", "2020-01-02")
    ]


def test_simulation_result_directory_accepts_name_under_result_root(tmp_path) -> None:
    """Verify simulation result directory accepts name under result root."""
    result_root = tmp_path / "sim_results"
    result_directory = result_root / "sim_20210101T000000Z"
    result_directory.mkdir(parents=True)
    (result_directory / "sim_result.json").write_text(
        json.dumps({"schema_version": 1, "status": "completed", "segment_artifacts": []}),
        encoding="utf-8",
    )
    assert (
        resolve_simulation_result_directory(result_directory.name, result_root=result_root)
        == result_directory
    )
