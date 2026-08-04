"""Compare two authenticated Rapfi reports on identical paired openings."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

try:
    from .rapfi_distill import (
        BLACK,
        WHITE,
        _record_from_dict,
        canonical_sha256,
        validate_record,
    )
except ImportError:
    from rapfi_distill import (  # type: ignore[no-redef]
        BLACK,
        WHITE,
        _record_from_dict,
        canonical_sha256,
        validate_record,
    )


def _load(path: Path) -> tuple[dict[str, object], list[object]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    recorded_hash = report.pop("report_sha256", None)
    if recorded_hash != canonical_sha256(report):
        raise ValueError(f"{path}: report hash mismatch")
    if report.get("report_type") != "rapfi_student_distillation":
        raise ValueError(f"{path}: wrong report type")
    if not report.get("complete"):
        raise ValueError(f"{path}: report is incomplete")
    records = [_record_from_dict(raw) for raw in report.get("games", [])]
    for record in records:
        validate_record(record)
        if record.error is not None or record.student_result is None:
            raise ValueError(f"{path}: report contains an unscored game")
    return report, records


def _comparable_signature(signature: dict[str, object]) -> dict[str, object]:
    ignored = {"checkpoint", "checkpoint_sha256"}
    return {key: value for key, value in signature.items() if key not in ignored}


def _index(records: list[object]) -> dict[tuple[int, int], object]:
    indexed = {(record.pair_index, record.student_color): record for record in records}
    if len(indexed) != len(records):
        raise ValueError("report contains duplicate pair/color games")
    return indexed


def _by_color(records: list[object]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for color, name in ((BLACK, "black"), (WHITE, "white")):
        games = [record for record in records if record.student_color == color]
        wins = sum(record.student_result == 1.0 for record in games)
        draws = sum(record.student_result == 0.5 for record in games)
        losses = sum(record.student_result == 0.0 for record in games)
        result[name] = {
            "games": len(games),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "score": (wins + 0.5 * draws) / len(games) if games else 0.0,
        }
    return result


def exact_two_sided_sign_p(gains: int, losses: int) -> float:
    trials = gains + losses
    if trials == 0:
        return 1.0
    lower = min(gains, losses)
    tail = sum(math.comb(trials, count) for count in range(lower + 1)) / 2**trials
    return min(1.0, 2.0 * tail)


def compare(parent_path: Path, candidate_path: Path) -> dict[str, object]:
    parent_report, parent_records = _load(parent_path)
    candidate_report, candidate_records = _load(candidate_path)
    parent_signature = parent_report.get("signature")
    candidate_signature = candidate_report.get("signature")
    if not isinstance(parent_signature, dict) or not isinstance(candidate_signature, dict):
        raise ValueError("report signature must be an object")
    if _comparable_signature(parent_signature) != _comparable_signature(candidate_signature):
        raise ValueError("reports do not use the same engine, openings, and budgets")

    parent_index = _index(parent_records)
    candidate_index = _index(candidate_records)
    if set(parent_index) != set(candidate_index):
        raise ValueError("reports do not contain the same pair/color keys")
    for key in parent_index:
        if parent_index[key].opening != candidate_index[key].opening:
            raise ValueError(f"opening mismatch at pair/color {key}")

    pair_ids = sorted({pair for pair, _ in parent_index})
    parent_pair_scores: dict[int, float] = {}
    candidate_pair_scores: dict[int, float] = {}
    for pair in pair_ids:
        parent_pair_scores[pair] = sum(
            float(parent_index[pair, color].student_result)
            for color in (BLACK, WHITE)
        ) / 2.0
        candidate_pair_scores[pair] = sum(
            float(candidate_index[pair, color].student_result)
            for color in (BLACK, WHITE)
        ) / 2.0
    deltas = [candidate_pair_scores[pair] - parent_pair_scores[pair] for pair in pair_ids]
    gains = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    unchanged = sum(delta == 0 for delta in deltas)
    sign_p = exact_two_sided_sign_p(gains, losses)

    parent_summary = parent_report["summary"]
    candidate_summary = candidate_report["summary"]
    parent_score = float(parent_summary["student_score"])
    candidate_score = float(candidate_summary["student_score"])
    if candidate_score < parent_score:
        recommendation = "reject_candidate"
    elif candidate_score > parent_score and sign_p < 0.05:
        recommendation = "eligible_for_non_regression_gates"
    else:
        recommendation = "continue_independent_evaluation"

    payload: dict[str, object] = {
        "format_version": 1,
        "parent_report": str(parent_path.resolve()),
        "candidate_report": str(candidate_path.resolve()),
        "parent_checkpoint_sha256": parent_signature["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate_signature["checkpoint_sha256"],
        "pairs": len(pair_ids),
        "parent_score": parent_score,
        "candidate_score": candidate_score,
        "score_delta": candidate_score - parent_score,
        "paired_opening_gains": gains,
        "paired_opening_losses": losses,
        "paired_opening_unchanged": unchanged,
        "two_sided_exact_sign_p": sign_p,
        "statistically_significant_at_0_05": sign_p < 0.05,
        "parent_by_color": _by_color(parent_records),
        "candidate_by_color": _by_color(candidate_records),
        "parent_student_teacher_agreement_rate": parent_summary.get(
            "student_teacher_agreement_rate"
        ),
        "candidate_student_teacher_agreement_rate": candidate_summary.get(
            "student_teacher_agreement_rate"
        ),
        "recommendation": recommendation,
    }
    payload["comparison_sha256"] = canonical_sha256(payload)
    return payload


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(args.parent_report, args.candidate_report)
    _atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
