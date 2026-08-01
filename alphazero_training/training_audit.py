"""Durable, controllable audit logs for every Gomoku training run.

Each run owns an append-only, SHA256-chained event stream, an atomically
updated manifest, a full console transcript, a control file, and hashes for
declared inputs/checkpoints.  The module deliberately uses only the standard
library so it can run on the cloud host before PyTorch is imported.
"""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence, TextIO
import uuid


SCHEMA = "gargantua_training_audit"
SCHEMA_VERSION = 1
AUDIT_MODES = ("full", "metrics", "minimal", "off")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SECRET_PATTERN = re.compile(r"(password|passwd|secret|token|api[_-]?key|credential)", re.I)
CRITICAL_EVENTS = {
    "run_started",
    "run_finished",
    "run_failed",
    "phase",
    "checkpoint",
    "artifact",
    "control",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>" if SECRET_PATTERN.search(str(key)) else _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def default_run_id(trainer: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    safe_trainer = re.sub(r"[^A-Za-z0-9_.-]+", "-", trainer).strip("-.")
    return f"{timestamp}_{safe_trainer}_{uuid.uuid4().hex[:8]}"


def add_audit_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the same audit controls to every training CLI."""

    parser.add_argument(
        "--audit-root",
        type=Path,
        help=(
            "training audit root (default: GOMOKU_TRAINING_LOG_DIR or "
            "alphazero_training/training_logs)"
        ),
    )
    parser.add_argument(
        "--audit-run-id",
        help="explicit unique ID for this process attempt (resume attempts get their own audit)",
    )
    parser.add_argument(
        "--audit-mode",
        choices=AUDIT_MODES,
        default=os.environ.get("GOMOKU_TRAINING_AUDIT", "full"),
        help="full is the default; off must be requested explicitly",
    )
    parser.add_argument(
        "--audit-metric-every",
        type=int,
        default=int(os.environ.get("GOMOKU_TRAINING_AUDIT_EVERY", "25")),
        help="record step metrics every N optimizer steps",
    )


class _Tee(TextIO):
    def __init__(self, original: TextIO, transcript: TextIO):
        self.original = original
        self.transcript = transcript

    def write(self, text: str) -> int:
        written = self.original.write(text)
        self.transcript.write(text)
        self.transcript.flush()
        return written

    def flush(self) -> None:
        self.original.flush()
        self.transcript.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.original, "isatty", lambda: False)())

    @property
    def encoding(self) -> str | None:
        return getattr(self.original, "encoding", "utf-8")


class TrainingAudit:
    """One append-only training audit with file-based runtime control."""

    def __init__(
        self,
        *,
        root: Path,
        run_id: str,
        trainer: str,
        mode: str = "full",
        metric_every: int = 25,
        config: Mapping[str, Any] | None = None,
        argv: Sequence[str] | None = None,
    ):
        if mode not in AUDIT_MODES:
            raise ValueError(f"invalid audit mode {mode!r}")
        if metric_every <= 0:
            raise ValueError("audit metric interval must be positive")
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(f"invalid audit run ID {run_id!r}")
        self.enabled = mode != "off"
        self.mode = mode
        self.metric_every = metric_every
        self.root = root.resolve()
        self.run_id = run_id
        self.run_dir = self.root / run_id
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self.control_path = self.run_dir / "control.json"
        self.transcript_path = self.run_dir / "console.log"
        self._lock = threading.RLock()
        self._sequence = 0
        self._last_hash = "0" * 64
        self._closed = not self.enabled
        self._paused_logged = False
        self._old_stdout: TextIO | None = None
        self._old_stderr: TextIO | None = None
        self._transcript: TextIO | None = None
        self._old_excepthook = None
        self.manifest: dict[str, Any] = {}
        if not self.enabled:
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing.get("status") == "running":
                raise RuntimeError(f"audit run is already active: {self.run_dir}")
            raise FileExistsError(f"audit run ID already exists: {self.run_dir}")
        self.manifest = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "trainer": trainer,
            "status": "running",
            "mode": mode,
            "metric_every": metric_every,
            "started_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "cwd": str(Path.cwd().resolve()),
            "argv": _jsonable(list(argv if argv is not None else sys.argv)),
            "config": _jsonable(dict(config or {})),
            "event_log": self.events_path.name,
            "console_log": self.transcript_path.name,
            "control_file": self.control_path.name,
            "artifacts": [],
            "event_count": 0,
            "last_event_sha256": self._last_hash,
        }
        _atomic_json(
            self.control_path,
            {
                "schema_version": 1,
                "pause_requested": False,
                "stop_requested": False,
                "updated_at_utc": utc_now(),
                "reason": "",
            },
        )
        self._write_manifest()
        if mode == "full":
            self._start_console_capture()
        self.event("run_started", {"trainer": trainer, "config": config or {}}, force=True)
        self._install_exception_hook()
        atexit.register(self._close_at_exit)

    @classmethod
    def from_namespace(
        cls,
        args: argparse.Namespace,
        *,
        trainer: str,
        config: Mapping[str, Any] | None = None,
    ) -> "TrainingAudit":
        root = args.audit_root or Path(
            os.environ.get(
                "GOMOKU_TRAINING_LOG_DIR", "alphazero_training/training_logs"
            )
        )
        run_id = args.audit_run_id or default_run_id(trainer)
        return cls(
            root=root,
            run_id=run_id,
            trainer=trainer,
            mode=args.audit_mode,
            metric_every=args.audit_metric_every,
            config=config if config is not None else vars(args),
        )

    def _start_console_capture(self) -> None:
        self._transcript = self.transcript_path.open("a", encoding="utf-8", buffering=1)
        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(sys.stdout, self._transcript)
        sys.stderr = _Tee(sys.stderr, self._transcript)

    def _stop_console_capture(self) -> None:
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
        if self._old_stderr is not None:
            sys.stderr = self._old_stderr
        if self._transcript is not None:
            self._transcript.flush()
            self._transcript.close()
        self._old_stdout = self._old_stderr = self._transcript = None

    def _install_exception_hook(self) -> None:
        self._old_excepthook = sys.excepthook

        def hook(exc_type: type[BaseException], exc: BaseException, traceback: Any) -> None:
            self.event(
                "run_failed",
                {"exception_type": exc_type.__name__, "message": str(exc)},
                force=True,
            )
            assert self._old_excepthook is not None
            self._old_excepthook(exc_type, exc, traceback)

        sys.excepthook = hook

    def _write_manifest(self) -> None:
        self.manifest["updated_at_utc"] = utc_now()
        _atomic_json(self.manifest_path, self.manifest)

    def event(self, event_type: str, payload: Mapping[str, Any], *, force: bool = False) -> None:
        if not self.enabled or self._closed:
            return
        if not force and self.mode == "minimal" and event_type not in CRITICAL_EVENTS:
            return
        with self._lock:
            base = {
                "schema_version": SCHEMA_VERSION,
                "sequence": self._sequence + 1,
                "timestamp_utc": utc_now(),
                "monotonic_ns": time.monotonic_ns(),
                "event": event_type,
                "payload": _jsonable(dict(payload)),
                "previous_event_sha256": self._last_hash,
            }
            event_hash = hashlib.sha256(_canonical(base)).hexdigest()
            record = {**base, "event_sha256": event_hash}
            with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._sequence += 1
            self._last_hash = event_hash
            self.manifest["event_count"] = self._sequence
            self.manifest["last_event_sha256"] = event_hash
            self._write_manifest()

    def should_log_metric(self, step: int) -> bool:
        return self.enabled and (step == 0 or step == 1 or step % self.metric_every == 0)

    def record_artifact(self, path: Path | str, *, role: str) -> dict[str, Any]:
        resolved = Path(path).resolve()
        item: dict[str, Any] = {"role": role, "path": str(resolved), "exists": resolved.is_file()}
        if resolved.is_file():
            item.update({"bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)})
        if self.enabled:
            artifacts = self.manifest.setdefault("artifacts", [])
            assert isinstance(artifacts, list)
            artifacts.append(item)
            self.event("checkpoint" if "checkpoint" in role else "artifact", item, force=True)
        return item

    def read_control(self) -> dict[str, Any]:
        if not self.enabled:
            return {"pause_requested": False, "stop_requested": False}
        try:
            payload = json.loads(self.control_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            self.event("control", {"state": "invalid", "error": str(error)}, force=True)
            return {"pause_requested": False, "stop_requested": False}
        return payload

    def check_control(self, *, poll_seconds: float = 2.0) -> bool:
        """Block while paused and return True when a graceful stop is requested."""

        if not self.enabled:
            return False
        while True:
            control = self.read_control()
            if bool(control.get("stop_requested")):
                self.event("control", {"state": "stop_observed", "control": control}, force=True)
                return True
            if not bool(control.get("pause_requested")):
                if self._paused_logged:
                    self.event("control", {"state": "resumed"}, force=True)
                    self._paused_logged = False
                return False
            if not self._paused_logged:
                self.event("control", {"state": "paused", "control": control}, force=True)
                self._paused_logged = True
            time.sleep(max(0.1, poll_seconds))

    def finish(self, status: str, payload: Mapping[str, Any] | None = None) -> None:
        if not self.enabled or self._closed:
            return
        with self._lock:
            self.event(
                "run_finished" if status in {"completed", "stopped"} else "run_failed",
                {"status": status, **dict(payload or {})},
                force=True,
            )
            self.manifest["status"] = status
            self.manifest["finished_at_utc"] = utc_now()
            self._write_manifest()
            self._closed = True
            if self._old_excepthook is not None:
                sys.excepthook = self._old_excepthook
            self._stop_console_capture()

    def _close_at_exit(self) -> None:
        if self.enabled and not self._closed:
            self.finish("aborted", {"reason": "process exited without explicit completion"})


def verify_run(run_dir: Path, *, verify_artifacts: bool = False) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported training audit manifest")
    previous = "0" * 64
    count = 0
    with (run_dir / str(manifest["event_log"])).open("r", encoding="utf-8") as handle:
        for count, line in enumerate(handle, start=1):
            record = json.loads(line)
            event_hash = record.pop("event_sha256")
            if record.get("sequence") != count:
                raise ValueError(f"event sequence mismatch at line {count}")
            if record.get("previous_event_sha256") != previous:
                raise ValueError(f"event chain mismatch at line {count}")
            actual = hashlib.sha256(_canonical(record)).hexdigest()
            if actual != event_hash:
                raise ValueError(f"event hash mismatch at line {count}")
            previous = event_hash
    if count != int(manifest.get("event_count", -1)) or previous != manifest.get(
        "last_event_sha256"
    ):
        raise ValueError("manifest does not match the event stream")
    checked = 0
    if verify_artifacts:
        for item in manifest.get("artifacts", []):
            if not item.get("exists") or "sha256" not in item:
                continue
            path = Path(str(item["path"]))
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise ValueError(f"artifact hash mismatch: {path}")
            checked += 1
    return {
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "events": count,
        "last_event_sha256": previous,
        "artifacts_verified": checked,
    }


def _load_run(root: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    run_dir = root.resolve() / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return run_dir, manifest


def _set_control(root: Path, run_id: str, action: str, reason: str) -> dict[str, Any]:
    run_dir, manifest = _load_run(root, run_id)
    path = run_dir / str(manifest["control_file"])
    control = json.loads(path.read_text(encoding="utf-8"))
    if action == "pause":
        control["pause_requested"] = True
    elif action == "resume":
        control["pause_requested"] = False
    elif action == "stop":
        control["stop_requested"] = True
        control["pause_requested"] = False
    else:
        raise ValueError(action)
    control.update({"updated_at_utc": utc_now(), "reason": reason})
    _atomic_json(path, control)
    return control


def _pid_state(pid_file: Path) -> dict[str, Any]:
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError, OSError) as error:
        return {"pid_file": str(pid_file), "alive": False, "error": str(error)}
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        alive = False
    else:
        alive = True
    return {"pid_file": str(pid_file), "pid": pid, "alive": alive}


def _tail(path: Path, lines: int = 5) -> list[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readlines()[-lines:]


def _resource_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
    }
    try:
        disk = shutil.disk_usage(Path.cwd())
        snapshot["disk"] = {"total": disk.total, "used": disk.used, "free": disk.free}
    except OSError as error:
        snapshot["disk_error"] = str(error)
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False
        )
        snapshot["gpu"] = completed.stdout.strip().splitlines()
        if completed.returncode:
            snapshot["gpu_error"] = completed.stderr.strip()
    except (OSError, subprocess.SubprocessError) as error:
        snapshot["gpu_error"] = str(error)
    return snapshot


def attach_existing_run(args: argparse.Namespace) -> int:
    """Index an already-running pipeline without modifying or restarting it."""

    audit = TrainingAudit(
        root=args.root,
        run_id=args.run_id,
        trainer=args.trainer,
        mode="metrics",
        metric_every=1,
        config={
            "attached": True,
            "pid_files": args.pid_file,
            "log_files": args.log_file,
            "artifacts": args.artifact,
            "completion_markers": args.completion_marker,
            "interval_seconds": args.interval,
        },
    )
    pid_files = [path.resolve() for path in args.pid_file]
    log_files = [path.resolve() for path in args.log_file]
    completion_markers = [path.resolve() for path in args.completion_marker]
    audit.event("phase", {"name": "attached_monitor", "state": "started"}, force=True)
    while True:
        pid_states = [_pid_state(path) for path in pid_files]
        logs = [
            {
                "path": str(path),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else 0,
                "tail": [line.rstrip("\r\n") for line in _tail(path)],
            }
            for path in log_files
        ]
        audit.event(
            "resource_snapshot",
            {"processes": pid_states, "logs": logs, **_resource_snapshot()},
        )
        if not any(bool(item.get("alive")) for item in pid_states):
            break
        if audit.check_control(poll_seconds=min(2.0, args.interval)):
            audit.finish("stopped", {"reason": "monitor control request"})
            return 0
        time.sleep(args.interval)

    for path in log_files:
        audit.record_artifact(path, role="pipeline_log")
    for path in args.artifact:
        audit.record_artifact(path, role="pipeline_artifact")
    complete = bool(completion_markers) and all(path.is_file() for path in completion_markers)
    audit.finish(
        "completed" if complete else "stopped",
        {
            "completion_markers": [str(path) for path in completion_markers],
            "all_completion_markers_present": complete,
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("GOMOKU_TRAINING_LOG_DIR", "alphazero_training/training_logs")),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--limit", type=int, default=20)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("run_id")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("run_id")
    verify_parser.add_argument("--artifacts", action="store_true")
    control_parser = subparsers.add_parser("control")
    control_parser.add_argument("run_id")
    control_parser.add_argument("action", choices=("pause", "resume", "stop"))
    control_parser.add_argument("--reason", default="operator request")
    attach_parser = subparsers.add_parser("attach")
    attach_parser.add_argument("--run-id", required=True)
    attach_parser.add_argument("--trainer", default="external_pipeline")
    attach_parser.add_argument("--pid-file", type=Path, action="append", required=True)
    attach_parser.add_argument("--log-file", type=Path, action="append", default=[])
    attach_parser.add_argument("--artifact", type=Path, action="append", default=[])
    attach_parser.add_argument("--completion-marker", type=Path, action="append", default=[])
    attach_parser.add_argument("--interval", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.command == "attach":
        if args.interval <= 0:
            raise SystemExit("--interval must be positive")
        return attach_existing_run(args)
    if args.command == "list":
        rows = []
        if root.is_dir():
            for manifest_path in root.glob("*/manifest.json"):
                try:
                    rows.append(json.loads(manifest_path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
        rows.sort(key=lambda item: str(item.get("started_at_utc", "")), reverse=True)
        print(json.dumps(rows[: args.limit], ensure_ascii=False, indent=2))
    elif args.command == "status":
        _, manifest = _load_run(root, args.run_id)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    elif args.command == "verify":
        print(
            json.dumps(
                verify_run(root / args.run_id, verify_artifacts=args.artifacts),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "control":
        print(
            json.dumps(
                _set_control(root, args.run_id, args.action, args.reason),
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
