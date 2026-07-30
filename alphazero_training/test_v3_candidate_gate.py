from __future__ import annotations

import copy
from dataclasses import asdict
import json
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from alphazero_training.ddqk_replay_export import (
    BLACK,
    BOARD_SIZE,
    EMPTY,
    WHITE,
    is_win,
    rebuild_benchmark_openings,
)
from alphazero_training.train_alphazero import Config, PolicyValueNet
from alphazero_training.v3_candidate_gate import (
    GateError,
    freeze_candidate,
    parse_args,
    promote_candidate,
    sha256_file,
)


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _lower95(values: list[float]) -> float:
    return max(
        0.0,
        sum(values) / len(values) - math.sqrt(math.log(20.0) / (2.0 * len(values))),
    )


def _exact_lower95(successes: int, trials: int) -> float:
    if successes == 0 or trials == 0:
        return 0.0
    if successes == trials:
        return 0.05 ** (1.0 / trials)

    def upper_tail(probability: float) -> float:
        return sum(
            math.comb(trials, outcome)
            * probability**outcome
            * (1.0 - probability) ** (trials - outcome)
            for outcome in range(successes, trials + 1)
        )

    low, high = 0.0, successes / trials
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if upper_tail(midpoint) < 0.05:
            low = midpoint
        else:
            high = midpoint
    return low


def _legal_winning_game(
    *,
    pair_index: int,
    model_color: int,
    opening: list[list[int]],
    model_result: float,
) -> dict[str, object]:
    """Build a short, fully legal 19x19 record with the requested winner."""

    if model_result not in (0.0, 1.0):
        raise ValueError("test histories support decisive results only")
    desired_winner = model_color if model_result == 1.0 else 3 - model_color
    opponent = 3 - desired_winner
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    moves: list[list[int]] = []
    for ply_index, raw_move in enumerate(opening):
        x, y = raw_move
        player = BLACK if ply_index % 2 == 0 else WHITE
        board[y, x] = player
        moves.append([x, y, player])

    target: list[tuple[int, int]] | None = None
    for y in range(BOARD_SIZE):
        for start_x in range(BOARD_SIZE - 4):
            cells = [(start_x + offset, y) for offset in range(5)]
            if all(int(board[cy, cx]) != opponent for cx, cy in cells):
                target = cells
                break
        if target is not None:
            break
    if target is None:  # pragma: no cover - impossible after a six-ply opening.
        raise AssertionError("could not find an unblocked five-cell line")

    player = BLACK if len(moves) % 2 == 0 else WHITE
    while True:
        if player == desired_winner:
            choices = [(x, y) for x, y in target if board[y, x] == EMPTY]
            if not choices:  # pragma: no cover - five target stones must win.
                raise AssertionError("winning target filled without a win")
            x, y = choices[0]
        else:
            x = y = -1
            for candidate_y in range(BOARD_SIZE - 1, -1, -1):
                for candidate_x in range(BOARD_SIZE - 1, -1, -1):
                    if (
                        board[candidate_y, candidate_x] != EMPTY
                        or (candidate_x, candidate_y) in target
                    ):
                        continue
                    board[candidate_y, candidate_x] = player
                    unsafe = is_win(board, candidate_x, candidate_y, player)
                    board[candidate_y, candidate_x] = EMPTY
                    if not unsafe:
                        x, y = candidate_x, candidate_y
                        break
                if x >= 0:
                    break
            if x < 0:  # pragma: no cover - board is nearly empty here.
                raise AssertionError("could not find harmless filler move")
        board[y, x] = player
        moves.append([x, y, player])
        if is_win(board, x, y, player):
            if player != desired_winner:  # pragma: no cover - fillers are checked.
                raise AssertionError("filler player won unexpectedly")
            break
        player = 3 - player

    post_opening = moves[len(opening) :]
    model_moves = sum(move[2] == model_color for move in post_opening)
    ddqk_moves = len(post_opening) - model_moves
    return {
        "pair_index": pair_index,
        "model_color": model_color,
        "opening": opening,
        "moves": moves,
        "winner": desired_winner,
        "model_result": model_result,
        "plies": len(moves),
        "model_seconds": 0.0,
        "ddqk_seconds": 0.0,
        "model_moves": model_moves,
        "ddqk_moves": ddqk_moves,
        "termination": "win",
        "error": None,
        "model_decision_reasons": ["mcts"] * model_moves,
    }


class V3CandidateGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = Config(
            board_size=5,
            win_length=5,
            channels=4,
            residual_blocks=1,
            simulations=2,
        )
        approved_model = PolicyValueNet(5, 4, 1)
        candidate_model = PolicyValueNet(5, 4, 1)
        with torch.no_grad():
            for parameter in candidate_model.parameters():
                parameter.add_(0.125)
        self.approved_state = _clone_state(approved_model.state_dict())
        self.candidate_state = _clone_state(candidate_model.state_dict())
        self.source = self.root / "latest.pt"
        self.source_payload = {
            "format_version": 3,
            "v3_stage": "selfplay",
            "iteration": 17,
            "global_step": 340,
            "config": asdict(self.config),
            "model_spec": {
                "board_size": 5,
                "channels": 4,
                "residual_blocks": 1,
                "input_planes": 4,
            },
            "train_model": _clone_state(self.candidate_state),
            "candidate_model": _clone_state(self.candidate_state),
            "best_model": _clone_state(self.approved_state),
            "approved_model": _clone_state(self.approved_state),
            "approved_checkpoint_sha256": "a" * 64,
            "parent_checkpoint_sha256": "b" * 64,
        }
        torch.save(self.source_payload, self.source)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _freeze(self) -> Path:
        candidate = self.root / "candidate_i17_eval.pt"
        result = freeze_candidate(
            self.source,
            candidate,
            expected_source_sha256=sha256_file(self.source),
        )
        self.assertEqual(result["status"], "frozen_not_approved")
        return candidate

    def _warmstart_source(
        self,
        *,
        parent_sha256: str = "d" * 64,
        stage: str = "tactical_expert_warmstart",
    ) -> tuple[Path, dict[str, object]]:
        path = self.root / "v3f_supervised.pt"
        payload: dict[str, object] = {
            "format_version": 3,
            "v3_stage": stage,
            "iteration": 17,
            "global_step": 1000,
            "config": asdict(self.config),
            "model_spec": {
                "board_size": 5,
                "channels": 4,
                "residual_blocks": 1,
                "input_planes": 4,
            },
            # Deliberately keep best_model different.  A valid V3F freeze must
            # select train_model and must never silently fall back to this key.
            "train_model": _clone_state(self.candidate_state),
            "best_model": _clone_state(self.approved_state),
            "parent_checkpoint_sha256": parent_sha256,
        }
        torch.save(payload, path)
        return path, payload

    def _freeze_warmstart(self) -> Path:
        source, _ = self._warmstart_source()
        candidate = self.root / "v3f_candidate_eval.pt"
        result = freeze_candidate(
            source,
            candidate,
            expected_source_sha256=sha256_file(source),
            expected_parent_sha256="d" * 64,
        )
        self.assertEqual(result["status"], "frozen_not_approved")
        self.assertEqual(result["source_stage"], "tactical_expert_warmstart")
        self.assertEqual(result["source_model_key"], "train_model")
        return candidate

    def _reports(
        self,
        candidate: Path,
        *,
        tactical_top1: float = 1.0,
        ddqk_score: float = 1.0,
        black_score: float = 1.0,
        white_score: float = 1.0,
        checkpoint_sha: str | None = None,
        certification_mode: str = "development",
        pairs: int = 50,
        current_schema: bool | None = None,
        pair_sweep_successes: int | None = None,
    ) -> tuple[Path, Path]:
        if current_schema is None:
            current_schema = certification_mode == "final-certification"
        digest = checkpoint_sha or sha256_file(candidate)
        tactical = {
            "checkpoint_sha256": digest,
            "checkpoint_model_key": "best_model",
            "split": "eval",
            "samples": 64,
            "dataset_sha256": "c" * 64,
            "raw_network": {
                "top1": tactical_top1,
                "family_macro_top1": tactical_top1,
            },
        }
        def results_for(score: float) -> list[float]:
            wins = round(score * pairs)
            return [1.0] * wins + [0.0] * (pairs - wins)

        if pair_sweep_successes is None:
            black_results = results_for(black_score)
            white_results = results_for(white_score)
        else:
            if not 0 <= pair_sweep_successes <= pairs:
                raise ValueError("invalid pair_sweep_successes")
            black_results = [1.0] * pairs
            white_results = [1.0] * pair_sweep_successes + [0.0] * (
                pairs - pair_sweep_successes
            )
        pair_scores = [
            (black_results[index] + white_results[index]) / 2.0
            for index in range(pairs)
        ]
        actual_score = sum(pair_scores) / pairs
        opening_plies = 6
        seed = 20260722
        openings = rebuild_benchmark_openings(
            seed=seed,
            pairs=pairs,
            opening_plies=opening_plies,
        )
        file_hashes = {
            "play_agent.py": "1" * 64,
            "v3_search.py": "2" * 64,
            "tactical_solver.py": "3" * 64,
            "benchmark_ddqk.py": "4" * 64,
        }
        if current_schema:
            file_hashes["train_alphazero.py"] = "a" * 64
            file_hashes["ddqk_adapter.py"] = "b" * 64
        canonical = lambda value: json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        code_bundle_sha = hashlib.sha256(canonical(file_hashes)).hexdigest()
        asset_hashes = {
            "dll.so": "6" * 64,
            "guess_data.txt": "7" * 64,
            "black_calculated_value_19.txt": "8" * 64,
            "white_calculated_value_19.txt": "9" * 64,
        }
        asset_bundle_sha = hashlib.sha256(canonical(asset_hashes)).hexdigest()
        opening_sha = hashlib.sha256(canonical(openings)).hexdigest()
        games = []
        for index in range(pairs):
            for color, result in ((1, black_results[index]), (2, white_results[index])):
                games.append(
                    _legal_winning_game(
                        pair_index=index,
                        model_color=color,
                        opening=openings[index],
                        model_result=result,
                    )
                )
        signature = {
            "checkpoint_sha256": digest,
            "ddqk_source": "DDQK.py",
            "ddqk_source_sha256": "5" * 64,
            "ddqk_dll_sha256": "6" * 64,
            "ddqk_depth": 4,
            "simulations": 384,
            "opening_plies": opening_plies,
            "max_moves": 361,
            "seed": seed,
            "pairs": pairs,
            "certification_mode": certification_mode,
            "opening_manifest_sha256": opening_sha,
            "evaluation_code": {
                "files": file_hashes,
                "bundle_sha256": code_bundle_sha,
            },
        }
        if current_schema:
            signature["ddqk_assets"] = {
                "files": asset_hashes,
                "bundle_sha256": asset_bundle_sha,
            }
        sweep_successes = sum(
            black_results[index] == 1.0 and white_results[index] == 1.0
            for index in range(pairs)
        )
        exact_lower = _exact_lower95(sweep_successes, pairs)
        final_passed = (
            certification_mode == "final-certification"
            and pairs >= 600
            and actual_score >= 0.995
            and sum(black_results) / pairs >= 0.99
            and sum(white_results) / pairs >= 0.99
            and exact_lower >= 0.995
        )
        certification: dict[str, object] = {
            "mode": certification_mode,
            "final_certified": final_passed,
        }
        if current_schema:
            certification.update(
                {
                    "status": (
                        "benchmark_final_requirements_passed"
                        if final_passed
                        else "not_final_certified"
                    ),
                    "requirements": {
                        "minimum_independent_paired_openings": 600,
                        "minimum_observed_score": 0.995,
                        "minimum_observed_black_score": 0.99,
                        "minimum_observed_white_score": 0.99,
                        "minimum_exact_pair_sweep_one_sided_95_lower_bound": 0.995,
                        "pair_sweep_success_definition": (
                            "model_wins_both_color_swapped_games"
                        ),
                        "requires_zero_errors": True,
                        "requires_zero_truncated_games": True,
                    },
                }
            )
        summary: dict[str, object] = {
            "requested_pairs": pairs,
            "complete_pairs": pairs,
            "incomplete_pairs": 0,
            "errors": 0,
            "truncated": 0,
            "completed_games": 2 * pairs,
            "scored_games": 2 * pairs,
            "score": actual_score,
            "paired_bootstrap_ci95": [actual_score, 1.0],
            "one_sided_95_lower_bound": _lower95(pair_scores),
            "one_sided_95_lower_bound_method": {
                "name": "hoeffding_bounded_mean",
                "alpha": 0.05,
                "independent_unit": "paired_opening",
                "sample_size": pairs,
            },
            "by_color": {
                "black": {"games": pairs, "score": sum(black_results) / pairs},
                "white": {"games": pairs, "score": sum(white_results) / pairs},
            },
        }
        if current_schema:
            summary.update(
                {
                    "observed_score": actual_score,
                    "hoeffding_bounded_pair_score_lower95": _lower95(pair_scores),
                    "pair_sweep_successes": sweep_successes,
                    "pair_sweep_trials": pairs,
                    "observed_pair_sweep_rate": sweep_successes / pairs,
                    "exact_pair_sweep_lower95": exact_lower,
                    "exact_pair_sweep_lower95_method": {
                        "name": "clopper_pearson_exact_binomial",
                        "alpha": 0.05,
                        "success_definition": "model_wins_both_color_swapped_games",
                        "independent_unit": "paired_opening",
                        "successes": sweep_successes,
                        "trials": pairs,
                    },
                }
            )
        ddqk = {
            "format_version": 3,
            "signature": signature,
            "openings": openings,
            "games": games,
            "certification": certification,
            "summary": summary,
        }
        tactical_path = self.root / "tactical.json"
        ddqk_path = self.root / "ddqk.json"
        tactical_path.write_text(json.dumps(tactical), encoding="utf-8")
        ddqk_path.write_text(json.dumps(ddqk), encoding="utf-8")
        return tactical_path, ddqk_path

    def _expected_ddqk(self, ddqk_path: Path) -> dict[str, object]:
        report = json.loads(ddqk_path.read_text(encoding="utf-8"))
        signature = report["signature"]
        return {
            "expected_ddqk_source_sha256": signature["ddqk_source_sha256"],
            "expected_ddqk_dll_sha256": signature["ddqk_dll_sha256"],
            "expected_ddqk_assets_bundle_sha256": signature["ddqk_assets"]["bundle_sha256"],
            "expected_ddqk_depth": signature["ddqk_depth"],
            "expected_simulations": signature["simulations"],
            "expected_opening_plies": signature["opening_plies"],
            "expected_max_moves": signature["max_moves"],
            "expected_seed": signature["seed"],
            "expected_evaluation_bundle_sha256": signature["evaluation_code"]["bundle_sha256"],
        }

    def test_freeze_uses_candidate_as_best_but_marks_not_approved(self) -> None:
        source_before = sha256_file(self.source)
        candidate = self._freeze()
        frozen = torch.load(candidate, map_location="cpu", weights_only=False)

        self.assertEqual(frozen["v3_stage"], "candidate_eval")
        self.assertEqual(frozen["approval_status"], "not_approved")
        self.assertIs(frozen["is_approved"], False)
        self.assertEqual(frozen["source_checkpoint_sha256"], source_before)
        for name in self.candidate_state:
            torch.testing.assert_close(frozen["best_model"][name], self.candidate_state[name])
            torch.testing.assert_close(frozen["candidate_model"][name], self.candidate_state[name])
        # Freezing is read-only with respect to training latest.
        self.assertEqual(sha256_file(self.source), source_before)

    def test_freeze_rejects_source_sha_mismatch(self) -> None:
        with self.assertRaisesRegex(GateError, "SHA256 mismatch"):
            freeze_candidate(
                self.source,
                self.root / "candidate.pt",
                expected_source_sha256="0" * 64,
            )

    def test_freeze_rejects_missing_candidate_model_even_if_best_exists(self) -> None:
        payload = dict(self.source_payload)
        payload.pop("candidate_model")
        bad = self.root / "bad_source.pt"
        torch.save(payload, bad)
        with self.assertRaisesRegex(GateError, "missing required candidate_model"):
            freeze_candidate(
                bad,
                self.root / "candidate.pt",
                expected_source_sha256=sha256_file(bad),
            )

    def test_freeze_warmstart_uses_only_train_model_and_binds_both_hashes(self) -> None:
        source, _ = self._warmstart_source()
        source_sha = sha256_file(source)
        candidate = self.root / "v3f_candidate_eval.pt"
        result = freeze_candidate(
            source,
            candidate,
            expected_source_sha256=source_sha,
            expected_parent_sha256="d" * 64,
        )
        frozen = torch.load(candidate, map_location="cpu", weights_only=False)

        self.assertEqual(result["source_sha256"], source_sha)
        self.assertEqual(result["source_parent_sha256"], "d" * 64)
        self.assertEqual(frozen["v3_stage"], "candidate_eval")
        self.assertEqual(frozen["approval_status"], "not_approved")
        self.assertIs(frozen["is_approved"], False)
        self.assertEqual(frozen["source_checkpoint_sha256"], source_sha)
        self.assertEqual(frozen["source_parent_checkpoint_sha256"], "d" * 64)
        self.assertEqual(frozen["source_checkpoint_stage"], "tactical_expert_warmstart")
        self.assertEqual(frozen["source_model_key"], "train_model")
        for name, expected in self.candidate_state.items():
            torch.testing.assert_close(frozen["best_model"][name], expected)
            torch.testing.assert_close(frozen["train_model"][name], expected)
            torch.testing.assert_close(frozen["candidate_model"][name], expected)

        # The three mappings are independent snapshots, not aliases.
        parameter_name = next(iter(frozen["best_model"]))
        original_train = frozen["train_model"][parameter_name].clone()
        frozen["best_model"][parameter_name].view(-1)[0] += 1.0
        torch.testing.assert_close(frozen["train_model"][parameter_name], original_train)
        self.assertEqual(sha256_file(source), source_sha)

    def test_freeze_warmstart_requires_out_of_band_parent_sha(self) -> None:
        source, _ = self._warmstart_source()
        with self.assertRaisesRegex(GateError, "requires expected_parent_sha256"):
            freeze_candidate(
                source,
                self.root / "candidate.pt",
                expected_source_sha256=sha256_file(source),
            )

        with self.assertRaisesRegex(GateError, "parent checkpoint SHA256 mismatch"):
            freeze_candidate(
                source,
                self.root / "candidate.pt",
                expected_source_sha256=sha256_file(source),
                expected_parent_sha256="e" * 64,
            )

    def test_freeze_warmstart_rejects_missing_or_malformed_train_model(self) -> None:
        source, payload = self._warmstart_source()
        payload.pop("train_model")
        torch.save(payload, source)
        with self.assertRaisesRegex(GateError, "missing required train_model"):
            freeze_candidate(
                source,
                self.root / "missing.pt",
                expected_source_sha256=sha256_file(source),
                expected_parent_sha256="d" * 64,
            )

        source, payload = self._warmstart_source()
        payload["train_model"] = {"not_a_model": torch.zeros(1)}
        torch.save(payload, source)
        with self.assertRaisesRegex(GateError, "incompatible with config"):
            freeze_candidate(
                source,
                self.root / "malformed.pt",
                expected_source_sha256=sha256_file(source),
                expected_parent_sha256="d" * 64,
            )

    def test_freeze_warmstart_rejects_bad_parent_stage_and_overwrite(self) -> None:
        source, _ = self._warmstart_source(parent_sha256="not-a-sha")
        with self.assertRaisesRegex(GateError, "64-character SHA256"):
            freeze_candidate(
                source,
                self.root / "bad_parent.pt",
                expected_source_sha256=sha256_file(source),
                expected_parent_sha256="d" * 64,
            )

        source, _ = self._warmstart_source(stage="tactical_expert_warmstart_typo")
        with self.assertRaisesRegex(GateError, "source v3_stage must be"):
            freeze_candidate(
                source,
                self.root / "wrong_stage.pt",
                expected_source_sha256=sha256_file(source),
                expected_parent_sha256="d" * 64,
            )

        source, _ = self._warmstart_source()
        output = self.root / "immutable.pt"
        freeze_candidate(
            source,
            output,
            expected_source_sha256=sha256_file(source),
            expected_parent_sha256="d" * 64,
        )
        with self.assertRaisesRegex(GateError, "refusing to overwrite immutable output"):
            freeze_candidate(
                source,
                output,
                expected_source_sha256=sha256_file(source),
                expected_parent_sha256="d" * 64,
            )

    def test_warmstart_candidate_continues_through_existing_promotion_chain(self) -> None:
        candidate = self._freeze_warmstart()
        tactical, ddqk = self._reports(candidate)
        champion = self.root / "v3f_development_screen.pt"
        promote_candidate(
            candidate,
            tactical,
            ddqk,
            champion,
            expected_candidate_sha256=sha256_file(candidate),
        )
        screened = torch.load(champion, map_location="cpu", weights_only=False)
        provenance = screened["provenance"]
        self.assertEqual(provenance["source_checkpoint_stage"], "tactical_expert_warmstart")
        self.assertEqual(provenance["source_model_key"], "train_model")
        self.assertEqual(provenance["source_parent_checkpoint_sha256"], "d" * 64)

    def test_freeze_cli_accepts_required_warmstart_parent_hash(self) -> None:
        args = parse_args(
            [
                "freeze",
                "--source",
                "run_v3f/latest.pt",
                "--expected-source-sha256",
                "a" * 64,
                "--expected-parent-sha256",
                "b" * 64,
                "--output",
                "candidates/v3f_candidate_eval.pt",
            ]
        )
        self.assertEqual(args.command, "freeze")
        self.assertEqual(args.expected_parent_sha256, "b" * 64)

    def test_promote_creates_permanent_champion_with_provenance(self) -> None:
        candidate = self._freeze()
        candidate_sha = sha256_file(candidate)
        tactical, ddqk = self._reports(candidate)
        champion = self.root / "champion_i17_score1000.pt"
        result = promote_candidate(
            candidate,
            tactical,
            ddqk,
            champion,
            expected_candidate_sha256=candidate_sha,
        )

        self.assertEqual(result["status"], "development_screen_passed_not_final_certified")
        approved = torch.load(champion, map_location="cpu", weights_only=False)
        self.assertEqual(approved["v3_stage"], "development_screened")
        self.assertEqual(approved["approval_status"], "not_final_certified")
        self.assertIs(approved["is_approved"], False)
        self.assertEqual(
            approved["provenance"]["candidate_evaluation_checkpoint_sha256"],
            candidate_sha,
        )
        self.assertEqual(
            approved["external_evaluation"]["tactical"]["report_sha256"],
            sha256_file(tactical),
        )
        self.assertEqual(
            approved["external_evaluation"]["ddqk"]["report_sha256"],
            sha256_file(ddqk),
        )
        for name in self.candidate_state:
            torch.testing.assert_close(approved["best_model"][name], self.candidate_state[name])
            torch.testing.assert_close(approved["approved_model"][name], self.candidate_state[name])

    def test_promote_rejects_below_tactical_gate(self) -> None:
        candidate = self._freeze()
        tactical, ddqk = self._reports(candidate, tactical_top1=0.98)
        output = self.root / "champion.pt"
        with self.assertRaisesRegex(GateError, "raw tactical top1"):
            promote_candidate(
                candidate,
                tactical,
                ddqk,
                output,
                expected_candidate_sha256=sha256_file(candidate),
            )
        self.assertFalse(output.exists())

    def test_promote_rejects_below_ddqk_color_gate(self) -> None:
        candidate = self._freeze()
        tactical, ddqk = self._reports(candidate, white_score=0.98)
        output = self.root / "champion.pt"
        with self.assertRaisesRegex(GateError, "white"):
            promote_candidate(
                candidate,
                tactical,
                ddqk,
                output,
                expected_candidate_sha256=sha256_file(candidate),
            )
        self.assertFalse(output.exists())

    def test_promote_rejects_report_bound_to_another_checkpoint(self) -> None:
        candidate = self._freeze()
        tactical, ddqk = self._reports(candidate, checkpoint_sha="d" * 64)
        with self.assertRaisesRegex(GateError, "not produced from this candidate"):
            promote_candidate(
                candidate,
                tactical,
                ddqk,
                self.root / "champion.pt",
                expected_candidate_sha256=sha256_file(candidate),
            )

    def test_promote_rejects_tactical_report_for_wrong_model_key(self) -> None:
        candidate = self._freeze()
        tactical, ddqk = self._reports(candidate)
        report = json.loads(tactical.read_text(encoding="utf-8"))
        report["checkpoint_model_key"] = "candidate_model"
        tactical.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(GateError, "frozen best_model key"):
            promote_candidate(
                candidate,
                tactical,
                ddqk,
                self.root / "champion.pt",
                expected_candidate_sha256=sha256_file(candidate),
            )

    def test_promote_rejects_candidate_tampered_after_hash_was_recorded(self) -> None:
        candidate = self._freeze()
        recorded_sha = sha256_file(candidate)
        tactical, ddqk = self._reports(candidate)
        with candidate.open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(GateError, "SHA256 mismatch"):
            promote_candidate(
                candidate,
                tactical,
                ddqk,
                self.root / "champion.pt",
                expected_candidate_sha256=recorded_sha,
            )

    def test_promote_rejects_wrong_model_key_weights(self) -> None:
        candidate = self._freeze()
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        key = next(
            name
            for name, tensor in payload["best_model"].items()
            if tensor.dtype.is_floating_point
        )
        payload["best_model"][key] = payload["best_model"][key].clone()
        payload["best_model"][key].view(-1)[0] += 1.0
        wrong = self.root / "candidate_wrong_key.pt"
        torch.save(payload, wrong)
        tactical, ddqk = self._reports(wrong)
        with self.assertRaisesRegex(GateError, "model keys do not contain identical weights"):
            promote_candidate(
                wrong,
                tactical,
                ddqk,
                self.root / "champion.pt",
                expected_candidate_sha256=sha256_file(wrong),
            )

    def test_promote_refuses_latest_name_and_existing_output(self) -> None:
        candidate = self._freeze()
        tactical, ddqk = self._reports(candidate)
        with self.assertRaisesRegex(GateError, "permanent name"):
            promote_candidate(
                candidate,
                tactical,
                ddqk,
                self.root / "latest.pt",
                expected_candidate_sha256=sha256_file(candidate),
            )
        champion = self.root / "champion.pt"
        champion.write_bytes(b"keep me")
        with self.assertRaisesRegex(GateError, "overwrite immutable output"):
            promote_candidate(
                candidate,
                tactical,
                ddqk,
                champion,
                expected_candidate_sha256=sha256_file(candidate),
            )
        self.assertEqual(champion.read_bytes(), b"keep me")

    def test_final_certification_requires_600_pairs_and_all_cli_expectations(self) -> None:
        candidate = self._freeze()
        tactical, ddqk = self._reports(
            candidate, certification_mode="final-certification", pairs=600
        )
        champion = self.root / "champion_i17_final.pt"
        result = promote_candidate(
            candidate,
            tactical,
            ddqk,
            champion,
            expected_candidate_sha256=sha256_file(candidate),
            certification_mode="final-certification",
            **self._expected_ddqk(ddqk),
        )
        self.assertEqual(result["status"], "approved_champion_created")
        self.assertIs(result["final_certified"], True)
        self.assertGreaterEqual(
            result["metrics"]["ddqk"]["one_sided_95_lower_bound"], 0.95
        )
        self.assertLess(
            result["metrics"]["ddqk"]["one_sided_95_lower_bound"], 1.0
        )
        self.assertEqual(result["metrics"]["ddqk"]["pair_sweep_successes"], 600)
        self.assertGreaterEqual(
            result["metrics"]["ddqk"]["exact_pair_sweep_lower95"],
            0.995,
        )
        payload = torch.load(champion, map_location="cpu", weights_only=False)
        self.assertEqual(payload["v3_stage"], "external_champion")
        self.assertIs(payload["is_approved"], True)

    def test_final_certification_rejects_missing_or_drifted_expectation(self) -> None:
        candidate = self._freeze()
        tactical, ddqk = self._reports(
            candidate, certification_mode="final-certification", pairs=600
        )
        with self.assertRaisesRegex(GateError, "requires CLI expectations"):
            promote_candidate(
                candidate,
                tactical,
                ddqk,
                self.root / "missing.pt",
                expected_candidate_sha256=sha256_file(candidate),
                certification_mode="final-certification",
            )
        expected = self._expected_ddqk(ddqk)
        expected["expected_simulations"] = 385
        with self.assertRaisesRegex(GateError, "simulations drift"):
            promote_candidate(
                candidate,
                tactical,
                ddqk,
                self.root / "drift.pt",
                expected_candidate_sha256=sha256_file(candidate),
                certification_mode="final-certification",
                **expected,
            )

    def test_final_certification_never_weakens_600_pair_floor(self) -> None:
        candidate = self._freeze()
        tactical, ddqk = self._reports(
            candidate, certification_mode="final-certification", pairs=599
        )
        with self.assertRaisesRegex(GateError, "below required 600"):
            promote_candidate(
                candidate,
                tactical,
                ddqk,
                self.root / "too_small.pt",
                expected_candidate_sha256=sha256_file(candidate),
                certification_mode="final-certification",
                min_ddqk_pairs=1,
                **self._expected_ddqk(ddqk),
            )

    def test_final_certification_rejects_599_sweeps_out_of_600_pairs(self) -> None:
        candidate = self._freeze()
        tactical, ddqk = self._reports(
            candidate,
            certification_mode="final-certification",
            pairs=600,
            pair_sweep_successes=599,
        )
        with self.assertRaisesRegex(GateError, "exact pair-sweep gate failed"):
            promote_candidate(
                candidate,
                tactical,
                ddqk,
                self.root / "one_pair_not_swept.pt",
                expected_candidate_sha256=sha256_file(candidate),
                certification_mode="final-certification",
                **self._expected_ddqk(ddqk),
            )

    def test_current_schema_rejects_tampered_exact_and_provenance_fields(self) -> None:
        candidate = self._freeze()

        def wrong_successes(report: dict[str, object]) -> None:
            report["summary"]["pair_sweep_successes"] = 49

        def wrong_method(report: dict[str, object]) -> None:
            report["summary"]["exact_pair_sweep_lower95_method"]["name"] = "bootstrap"

        def wrong_asset_bundle(report: dict[str, object]) -> None:
            report["signature"]["ddqk_assets"]["bundle_sha256"] = "0" * 64

        def missing_asset_bundle(report: dict[str, object]) -> None:
            report["signature"].pop("ddqk_assets")

        def incomplete_code_bundle(report: dict[str, object]) -> None:
            files = report["signature"]["evaluation_code"]["files"]
            files.pop("train_alphazero.py")
            canonical = json.dumps(
                files,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            report["signature"]["evaluation_code"]["bundle_sha256"] = (
                hashlib.sha256(canonical).hexdigest()
            )

        cases = (
            ("successes", wrong_successes, "pair-sweep successes"),
            ("method", wrong_method, "confidence method metadata"),
            ("assets", wrong_asset_bundle, "asset bundle hash"),
            ("missing assets", missing_asset_bundle, "no complete ddqk_assets"),
            ("code", incomplete_code_bundle, "six decision files"),
        )
        for index, (name, mutate, message) in enumerate(cases):
            with self.subTest(field=name):
                tactical, ddqk = self._reports(
                    candidate,
                    current_schema=True,
                )
                report = json.loads(ddqk.read_text(encoding="utf-8"))
                mutate(report)
                ddqk.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(GateError, message):
                    promote_candidate(
                        candidate,
                        tactical,
                        ddqk,
                        self.root / f"tampered_{index}.pt",
                        expected_candidate_sha256=sha256_file(candidate),
                    )

    def test_promote_replays_every_game_and_rebuilds_openings_from_seed(self) -> None:
        candidate = self._freeze()
        tactical, ddqk = self._reports(candidate)
        original = json.loads(ddqk.read_text(encoding="utf-8"))

        def wrong_result(report: dict[str, object]) -> None:
            game = report["games"][0]
            game["model_result"] = 1.0 - float(game["model_result"])

        def wrong_winner(report: dict[str, object]) -> None:
            game = report["games"][0]
            game["winner"] = 3 - int(game["winner"])

        def wrong_plies(report: dict[str, object]) -> None:
            report["games"][0]["plies"] += 1

        def missing_history(report: dict[str, object]) -> None:
            report["games"][0].pop("moves")

        def forged_manifest(report: dict[str, object]) -> None:
            forged = copy.deepcopy(report["openings"][0])
            forged[0] = [18, 18] if forged[0] != [18, 18] else [0, 0]
            report["openings"][0] = forged
            for game in report["games"][:2]:
                game["opening"] = copy.deepcopy(forged)
            report["signature"]["opening_manifest_sha256"] = hashlib.sha256(
                json.dumps(
                    report["openings"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()

        cases = (
            ("result", wrong_result, "model_result does not match"),
            ("winner", wrong_winner, "winner does not match"),
            ("plies", wrong_plies, "plies do not match"),
            ("missing moves", missing_history, "moves must be a list"),
            ("seed recipe", forged_manifest, "cannot be reproduced"),
        )
        for index, (label, mutate, message) in enumerate(cases):
            with self.subTest(label=label):
                report = copy.deepcopy(original)
                mutate(report)
                ddqk.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(GateError, message):
                    promote_candidate(
                        candidate,
                        tactical,
                        ddqk,
                        self.root / f"replay_tampered_{index}.pt",
                        expected_candidate_sha256=sha256_file(candidate),
                    )


if __name__ == "__main__":
    unittest.main()
