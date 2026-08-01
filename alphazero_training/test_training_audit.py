from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from alphazero_training.training_audit import (
    TrainingAudit,
    _set_control,
    verify_run,
)


class TrainingAuditTests(unittest.TestCase):
    def test_complete_run_has_chained_events_console_and_artifact_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "candidate.pt"
            artifact.write_bytes(b"checkpoint")
            audit = TrainingAudit(
                root=root / "logs",
                run_id="unit-complete",
                trainer="unit",
                mode="full",
                metric_every=2,
                config={"steps": 4, "api_token": "must-not-leak"},
                argv=["trainer", "--steps", "4"],
            )
            print("captured training output")
            audit.event("train_metrics", {"step": 2, "loss": 0.25})
            audit.record_artifact(artifact, role="training_checkpoint")
            audit.finish("completed", {"step": 4})

            run_dir = root / "logs" / "unit-complete"
            verified = verify_run(run_dir, verify_artifacts=True)
            self.assertEqual("completed", verified["status"])
            self.assertEqual(1, verified["artifacts_verified"])
            self.assertIn(
                "captured training output",
                (run_dir / "console.log").read_text(encoding="utf-8"),
            )
            manifest_text = (run_dir / "manifest.json").read_text(encoding="utf-8")
            self.assertNotIn("must-not-leak", manifest_text)
            self.assertIn("<redacted>", manifest_text)

    def test_stop_control_is_observed_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "logs"
            audit = TrainingAudit(
                root=root,
                run_id="unit-control",
                trainer="unit",
                mode="metrics",
            )
            control = _set_control(root, "unit-control", "stop", "test stop")
            self.assertTrue(control["stop_requested"])
            self.assertTrue(audit.check_control(poll_seconds=0.01))
            audit.finish("stopped")
            manifest = json.loads(
                (root / "unit-control" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual("stopped", manifest["status"])
            verify_run(root / "unit-control")

    def test_pause_waits_until_resume_without_requesting_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "logs"
            audit = TrainingAudit(
                root=root,
                run_id="unit-pause",
                trainer="unit",
                mode="metrics",
            )
            _set_control(root, "unit-pause", "pause", "test pause")
            result: list[bool] = []
            worker = threading.Thread(
                target=lambda: result.append(audit.check_control(poll_seconds=0.01))
            )
            worker.start()
            time.sleep(0.05)
            self.assertTrue(worker.is_alive())
            _set_control(root, "unit-pause", "resume", "test resume")
            worker.join(timeout=1)
            self.assertFalse(worker.is_alive())
            self.assertEqual([False], result)
            audit.finish("completed")
            verify_run(root / "unit-pause")

    def test_event_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "logs"
            audit = TrainingAudit(
                root=root,
                run_id="unit-corrupt",
                trainer="unit",
                mode="metrics",
            )
            audit.finish("completed")
            events = root / "unit-corrupt" / "events.jsonl"
            text = events.read_text(encoding="utf-8").replace("run_started", "run_changed", 1)
            events.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "event hash mismatch"):
                verify_run(root / "unit-corrupt")


if __name__ == "__main__":
    unittest.main()
