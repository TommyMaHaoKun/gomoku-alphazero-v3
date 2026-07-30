"""Generate bilingual Gargantua training and evaluation charts.

The script extracts auditable metrics from the archived V3G experiment, saves
compact CSV snapshots beside the figures, and exports presentation-ready PNGs.
It never treats training loss or integration checks as direct playing strength.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ARCHIVE = Path("v3g_archive_20260730_i100")

BG = "#FAFAF7"
INK = "#191A1D"
MUTED = "#6B6F76"
GRID = "#D9D9D2"
BLUE = "#3569C8"
ORANGE = "#D97732"
GOLD = "#B88A2F"
PINK = "#B95676"
LIGHT_BLUE = "#C9D8F2"
LIGHT_ORANGE = "#F2D3BE"


def configure_fonts() -> str:
    """Use a Windows font that contains both Latin and Chinese glyphs."""
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = family
            break
    else:
        family = "DejaVu Sans"
        plt.rcParams["font.family"] = family
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.unicode_minus": False,
        }
    )
    return family


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_warmup(archive: Path) -> list[dict[str, float]]:
    text = (archive / "warmup" / "train_v3f.log").read_text(encoding="utf-8")
    pattern = re.compile(
        r"step=(\d+)/800 loss=([0-9.]+) policy=([0-9.]+) value=([0-9.]+)"
        r".*?safe_margin=([0-9.]+) lr=([0-9.eE+-]+)"
    )
    rows = []
    for match in pattern.finditer(text):
        step, loss, policy, value, safe_margin, lr = match.groups()
        rows.append(
            {
                "step": int(step),
                "total_loss": float(loss),
                "policy_loss": float(policy),
                "value_loss": float(value),
                "safe_margin_loss": float(safe_margin),
                "learning_rate": float(lr),
            }
        )
    if len(rows) < 20:
        raise ValueError(f"warm-up log is too sparse: {len(rows)} points")
    return rows


def extract_selfplay(archive: Path) -> list[dict[str, float]]:
    text = (archive / "logs" / "train_v3.log").read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        match = re.search(r"iteration=(\d+) complete metrics=(\{.*\}) checkpoint=", line)
        if not match:
            continue
        iteration = int(match.group(1))
        metrics = json.loads(match.group(2))
        training = metrics["training"]
        selfplay = metrics["selfplay"]
        results = selfplay["results"]
        rows.append(
            {
                "iteration": iteration,
                "total_loss": float(training["loss"]),
                "policy_loss": float(training["policy_loss"]),
                "value_loss": float(training["value_loss"]),
                "positions": int(selfplay["positions"]),
                "positions_per_second": float(selfplay["positions_per_second"]),
                "replay_size": int(metrics["replay_size"]),
                "black_wins": int(results.get("black", 0)),
                "white_wins": int(results.get("white", 0)),
                "draws": int(results.get("draw", results.get("draws", 0))),
            }
        )
    rows.sort(key=lambda row: int(row["iteration"]))
    if [row["iteration"] for row in rows] != list(range(1, 101)):
        raise ValueError("expected one complete record for each self-play iteration 1-100")
    return rows


def extract_evaluations(archive: Path) -> list[dict[str, float]]:
    rows = []
    for iteration in range(5, 101, 5):
        tactics = read_json(archive / "selfplay" / f"raw_tactics_i{iteration}.json")
        white = read_json(
            archive / "selfplay" / f"white_defense_dev_i{iteration}.json"
        )
        rows.append(
            {
                "iteration": iteration,
                "tactical_top1": float(tactics["raw_network"]["top1"]),
                "tactical_oracle_mass": float(
                    tactics["raw_network"]["mean_oracle_mass"]
                ),
                "white_safe_top1": float(white["metrics"]["top1_in_safe_set"]),
                "white_safe_mass": float(white["metrics"]["safe_probability_mass"]),
            }
        )
    return rows


def extract_approved_results(archive: Path) -> list[dict[str, object]]:
    tactics = read_json(archive / "selfplay" / "raw_tactics_iapproved.json")
    white = read_json(archive / "selfplay" / "white_defense_dev_iapproved.json")
    tactical_total = int(tactics["samples"])
    tactical_passed = round(float(tactics["raw_network"]["top1"]) * tactical_total)
    white_total = int(white["metrics"]["records"])
    white_passed = int(white["metrics"]["top1_in_safe_set_count"])
    return [
        {
            "metric": "Raw tactical network",
            "passed": tactical_passed,
            "total": tactical_total,
            "scope": "held-out tactical positions",
        },
        {
            "metric": "White-defense choices",
            "passed": white_passed,
            "total": white_total,
            "scope": "held-out white-to-move positions",
        },
        {
            "metric": "Game integration",
            "passed": 159,
            "total": 159,
            "scope": "software regression checks",
        },
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def moving_average(values: Iterable[float], window: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    result = np.full_like(array, np.nan)
    if len(array) >= window:
        result[window - 1 :] = np.convolve(array, np.ones(window) / window, mode="valid")
    return result


def new_figure() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=200)
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.80)
    return fig, ax


def title(fig: plt.Figure, heading: str, subtitle: str) -> None:
    fig.text(0.06, 0.93, heading, fontsize=22, fontweight="bold", ha="left")
    fig.text(0.06, 0.875, subtitle, fontsize=10.5, color=MUTED, ha="left")


def source_note(fig: plt.Figure, note: str) -> None:
    fig.text(0.06, 0.035, note, fontsize=8.5, color=MUTED, ha="left")


def style_axis(ax: plt.Axes, *, percent: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    if percent:
        from matplotlib.ticker import PercentFormatter

        ax.yaxis.set_major_formatter(PercentFormatter(1.0))


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, facecolor=BG)
    plt.close(fig)


def chart_warmup(rows: list[dict], output: Path) -> None:
    fig, ax = new_figure()
    x = [row["step"] for row in rows]
    for key, label, color in [
        ("total_loss", "Total loss", INK),
        ("policy_loss", "Policy loss", BLUE),
        ("value_loss", "Value loss", ORANGE),
    ]:
        ax.plot(x, [row[key] for row in rows], marker="o", markersize=3.2, linewidth=2, color=color, label=label)
    title(
        fig,
        "Supervised Warm-up Loss",
        "800 optimization steps; lower loss means closer agreement with the training targets.",
    )
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, 810)
    ax.set_ylim(bottom=0)
    style_axis(ax)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    source_note(fig, "Source: archived supervised warm-up log, 2026-07-30.")
    save(fig, output)


def chart_selfplay_loss(rows: list[dict], output: Path) -> None:
    fig, ax = new_figure()
    x = np.asarray([row["iteration"] for row in rows])
    for key, label, color in [
        ("total_loss", "Total loss", INK),
        ("policy_loss", "Policy loss", BLUE),
        ("value_loss", "Value loss", ORANGE),
    ]:
        raw = np.asarray([row[key] for row in rows])
        ax.plot(x, raw, color=color, alpha=0.18, linewidth=1)
        ax.plot(x, moving_average(raw, 7), color=color, linewidth=2.5, label=label)
    title(
        fig,
        "Self-play Training Loss",
        "Seven-iteration moving average across 100 reinforcement-learning iterations; loss is not proof of playing strength.",
    )
    ax.set_xlabel("Self-play iteration")
    ax.set_ylabel("Loss")
    ax.set_xlim(1, 100)
    ax.set_ylim(bottom=0)
    style_axis(ax)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    source_note(fig, "Source: archived V3G self-play training log; faint lines show raw values.")
    save(fig, output)


def chart_replay_growth(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), dpi=200)
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.16, top=0.78, wspace=0.25)
    x = np.asarray([row["iteration"] for row in rows])
    replay = np.asarray([row["replay_size"] for row in rows])
    positions = np.asarray([row["positions"] for row in rows])
    axes[0].plot(x, replay, color=BLUE, linewidth=2.6)
    axes[0].fill_between(x, replay, color=LIGHT_BLUE, alpha=0.55)
    axes[0].scatter([x[-1]], [replay[-1]], color=BLUE, s=32, zorder=3)
    axes[0].annotate(f"{replay[-1]:,}", (x[-1], replay[-1]), xytext=(-8, 10), textcoords="offset points", ha="right", fontweight="bold")
    axes[0].set_title("Replay buffer growth", fontsize=14, loc="left", fontweight="bold")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Stored positions")
    axes[0].set_xlim(1, 100)
    axes[0].set_ylim(0, replay.max() * 1.12)
    style_axis(axes[0])

    axes[1].bar(x, positions, color=LIGHT_ORANGE, edgecolor=ORANGE, linewidth=0.35, width=0.8)
    axes[1].plot(x, moving_average(positions, 7), color=ORANGE, linewidth=2.5, label="7-iteration average")
    axes[1].set_title("New self-play positions", fontsize=14, loc="left", fontweight="bold")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Positions")
    axes[1].set_xlim(1, 100)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper right")
    style_axis(axes[1])
    title(
        fig,
        "Self-play Data Generation",
        "100 iterations produced 105,485 archived replay positions.",
    )
    source_note(fig, "Source: archived V3G iteration metrics and replay manifest.")
    save(fig, output)


def chart_tactical_evaluation(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), dpi=200)
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.16, top=0.78, wspace=0.24)
    x = np.asarray([row["iteration"] for row in rows])
    accuracy = np.asarray([row["tactical_top1"] for row in rows])
    mass = np.asarray([row["tactical_oracle_mass"] for row in rows])
    approved = 47 / 48

    axes[0].plot(x, accuracy, color=BLUE, marker="o", markersize=4, linewidth=2.2, label="Experimental snapshots")
    axes[0].axhline(approved, color=INK, linestyle="--", linewidth=1.6, label="Approved model: 47/48")
    axes[0].set_ylim(0.94, 0.99)
    axes[0].set_xlim(5, 100)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Top-1 accuracy")
    axes[0].set_title("Correct move ranked first", fontsize=14, loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8.2, loc="lower right")
    style_axis(axes[0], percent=True)

    axes[1].plot(x, mass, color=ORANGE, marker="o", markersize=4, linewidth=2.2)
    axes[1].set_xlim(5, 100)
    axes[1].set_ylim(0.68, 0.77)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Mean probability")
    axes[1].set_title("Probability on correct tactical moves", fontsize=14, loc="left", fontweight="bold")
    style_axis(axes[1], percent=True)
    title(
        fig,
        "Held-out Tactical Evaluation",
        "The 100-iteration experiment remained at 46/48 and did not replace the approved 47/48 model.",
    )
    source_note(fig, "Source: 20 held-out raw-network evaluations, every 5 iterations; 48 positions each.")
    save(fig, output)


def chart_white_evaluation(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), dpi=200)
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.16, top=0.78, wspace=0.24)
    x = np.asarray([row["iteration"] for row in rows])
    top1 = np.asarray([row["white_safe_top1"] for row in rows])
    safe_mass = np.asarray([row["white_safe_mass"] for row in rows])
    approved = 16 / 18

    axes[0].plot(x, top1, color=BLUE, marker="o", markersize=4, linewidth=2.2, label="Experimental snapshots")
    axes[0].axhline(approved, color=INK, linestyle="--", linewidth=1.6, label="Approved model: 16/18")
    axes[0].set_ylim(0.80, 0.91)
    axes[0].set_xlim(5, 100)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Safe top-1 rate")
    axes[0].set_title("Safe move ranked first", fontsize=14, loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8.2, loc="lower right")
    style_axis(axes[0], percent=True)

    axes[1].plot(x, safe_mass, color=ORANGE, marker="o", markersize=4, linewidth=2.2)
    axes[1].fill_between(x, safe_mass, 0.79, color=LIGHT_ORANGE, alpha=0.45)
    axes[1].set_ylim(0.79, 0.84)
    axes[1].set_xlim(5, 100)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Safe probability mass")
    axes[1].set_title("Confidence assigned to safe moves", fontsize=14, loc="left", fontweight="bold")
    style_axis(axes[1], percent=True)
    title(
        fig,
        "Held-out White-defense Evaluation",
        "Confidence increased, but top-1 accuracy stayed at 15/18—below the approved model's 16/18.",
    )
    source_note(fig, "Source: 20 held-out raw-network evaluations, every 5 iterations; 18 positions each.")
    save(fig, output)


def chart_final_results(rows: list[dict], output: Path) -> None:
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=200)
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.15, top=0.80)
    labels = [str(row["metric"]) for row in rows]
    values = [int(row["passed"]) / int(row["total"]) for row in rows]
    y = np.arange(len(rows))
    colors = [BLUE, ORANGE, GOLD]
    ax.barh(y, values, color=colors, height=0.52, edgecolor=INK, linewidth=0.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Pass rate")
    for index, (value, row) in enumerate(zip(values, rows)):
        ax.text(
            value - 0.015,
            index,
            f"{int(row['passed'])}/{int(row['total'])}  ({value:.1%})",
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
            color="white",
        )
    style_axis(ax)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    title(
        fig,
        "Approved Gargantua V2 Validation",
        "The three tests have different scopes; integration measures software reliability, not playing strength.",
    )
    source_note(fig, "Source: approved checkpoint evaluations and the current Pygame integration suite.")
    save(fig, output)


def make_contact_sheet(images: list[Path], output: Path, font_family: str) -> None:
    thumb_w, thumb_h = 900, 506
    margin, label_h = 40, 42
    canvas = Image.new("RGB", (thumb_w * 2 + margin * 3, (thumb_h + label_h) * 3 + margin * 4), BG)
    draw = ImageDraw.Draw(canvas)
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    for index, path in enumerate(images):
        row, col = divmod(index, 2)
        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + label_h + margin)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            canvas.paste(image, (x, y))
        draw.text((x, y + thumb_h + 8), path.stem, fill=INK, font=font)
    canvas.save(output, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    archive = args.archive.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not archive.exists():
        raise FileNotFoundError(f"training archive not found: {archive}")

    font_family = configure_fonts()
    warmup = extract_warmup(archive)
    selfplay = extract_selfplay(archive)
    evaluations = extract_evaluations(archive)
    approved = extract_approved_results(archive)

    write_csv(data_dir / "warmup_loss.csv", warmup)
    write_csv(data_dir / "selfplay_metrics.csv", selfplay)
    write_csv(data_dir / "evaluation_snapshots.csv", evaluations)
    write_csv(data_dir / "approved_results.csv", approved)
    (data_dir / "provenance.json").write_text(
        json.dumps(
            {
                "source_archive": archive.name,
                "warmup_points": len(warmup),
                "selfplay_iterations": len(selfplay),
                "evaluation_snapshots": len(evaluations),
                "notes": [
                    "Training loss is an optimization metric, not direct playing strength.",
                    "Integration checks measure software behavior, not match win rate.",
                    "The experimental 100-iteration candidate was not promoted over Gargantua V2.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    images = [
        output / "01_supervised_warmup_loss.png",
        output / "02_selfplay_training_loss.png",
        output / "03_selfplay_data_generation.png",
        output / "04_tactical_evaluation_over_training.png",
        output / "05_white_defense_evaluation_over_training.png",
        output / "06_approved_model_results.png",
    ]
    chart_warmup(warmup, images[0])
    chart_selfplay_loss(selfplay, images[1])
    chart_replay_growth(selfplay, images[2])
    chart_tactical_evaluation(evaluations, images[3])
    chart_white_evaluation(evaluations, images[4])
    chart_final_results(approved, images[5])
    make_contact_sheet(images, output / "00_all_charts_preview.png", font_family)
    print(f"generated {len(images)} charts in {output}")


if __name__ == "__main__":
    main()
