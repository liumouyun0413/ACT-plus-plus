#!/usr/bin/env python3
"""Audit and materialize complete-task episodes for ACT fine-tuning.

Run without --selected-list to create a review CSV/template. After reviewing the
videos, put one source_episode per line in the template and rerun with
--selected-list. Selected HDF5 files are symlinked or copied without modifying
segment boundaries or alignment metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


def text_attr(root, name, default=""):
    value = root.attrs.get(name, default)
    return value.decode() if isinstance(value, bytes) else str(value)


def inspect(path: Path, visualization_dir: Path | None) -> dict:
    with h5py.File(path, "r") as root:
        source = text_attr(root, "source_episode", path.stem)
        frames = len(root["action"])
        lengths = (
            root["segment_lengths"][:].astype(int)
            if "segment_lengths" in root else np.asarray([frames])
        )
        correction = int(root.attrs.get("action_alignment_correction_frames", 0))
        confidence = float(root.attrs.get("action_alignment_confidence", np.nan))
    videos = []
    if visualization_dir:
        for candidate in visualization_dir.rglob("*.mp4"):
            if path.stem in candidate.name or source in candidate.name:
                videos.append(str(candidate))
    return {
        "source_episode": source,
        "hdf5": str(path),
        "frames": frames,
        "duration_seconds": round(frames / 30.0, 3),
        "segments": len(lengths),
        "min_segment_frames": int(lengths.min()),
        "max_segment_frames": int(lengths.max()),
        "lag_correction_frames": correction,
        "lag_confidence": confidence,
        "visualizations": ";".join(videos),
        "review_complete_task": "",
        "review_notes": "",
    }


def read_selected(path: Path) -> list[str]:
    selected = [
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("selected list must contain unique source episodes")
    return selected


def main(args):
    input_dir = Path(args.input_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = Path(args.visualization_dir) if args.visualization_dir else None
    rows = [
        inspect(path, visualization_dir)
        for path in sorted(input_dir.glob("episode_*.hdf5"))
    ]
    if not rows:
        raise ValueError(f"no HDF5 episodes found in {input_dir}")

    csv_path = report_dir / "complete_task_review.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    template = report_dir / "complete_source_episodes.txt"
    if not template.exists():
        template.write_text(
            "# After video review, put one COMPLETE source_episode per line.\n"
            "# Example: episode_000067\n"
        )
    print(f"Review CSV: {csv_path}")
    print(f"Selection template: {template}")

    if not args.selected_list:
        print("Review videos and rerun with --selected-list; no dataset was created.")
        return

    selected = read_selected(Path(args.selected_list))
    by_source = {row["source_episode"]: row for row in rows}
    missing = sorted(set(selected) - set(by_source))
    if missing:
        raise ValueError(f"selected sources are absent from input dataset: {missing}")
    output_dir = Path(args.output_dir)
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("output-dir must differ from input-dir")
    if args.overwrite:
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")

    manifest = []
    for output_index, source in enumerate(selected):
        source_path = Path(by_source[source]["hdf5"])
        destination = output_dir / f"episode_{output_index:06d}.hdf5"
        if args.copy:
            shutil.copy2(source_path, destination)
        else:
            destination.symlink_to(source_path.resolve())
        manifest.append({
            "act_episode": output_index,
            "source_episode": source,
            "source_hdf5": str(source_path),
            "output": str(destination),
            "selection": "manual_complete_task_review",
        })
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / "source_episode_order.txt").write_text("\n".join(selected) + "\n")
    (output_dir / "summary.json").write_text(json.dumps({
        "complete_task_episodes": len(manifest),
        "storage": "copy" if args.copy else "symlink",
        "input_dir": str(input_dir),
    }, indent=2))
    print(f"Fine-tune dataset created: {output_dir} ({len(manifest)} episodes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--visualization-dir")
    parser.add_argument("--selected-list")
    parser.add_argument("--output-dir")
    parser.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args()
    if parsed.selected_list and not parsed.output_dir:
        parser.error("--output-dir is required with --selected-list")
    main(parsed)
