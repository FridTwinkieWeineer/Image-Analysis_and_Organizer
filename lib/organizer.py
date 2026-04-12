"""File organization and manifest generation."""

import csv
import os
import platform
import shutil
import subprocess
from pathlib import Path


def build_file_map(
    clusters: dict[str, int],
    labels: dict[str, str],  # cluster_id (as str) -> label
    reassignments: dict[str, int],
    descriptions: dict[str, str],
) -> list[dict]:
    """
    Build the list of copy operations.
    Returns list of {original_path, cluster_label, description}.
    """
    records = []
    for path, cluster_id in clusters.items():
        # Apply manual reassignment if present
        effective_id = reassignments.get(path, cluster_id)
        label = labels.get(str(effective_id), f"cluster_{effective_id}")

        if label.lower() in ("skip", "unsorted"):
            continue

        records.append({
            "original_path": path,
            "cluster_label": label,
            "description": descriptions.get(path, ""),
        })
    return records


def copy_to_folders(
    records: list[dict],
    output_dir: str,
    dry_run: bool = False,
) -> list[dict]:
    """
    Copy files into label-named subfolders under output_dir.
    Returns records with 'output_path' added.
    """
    results = []
    for rec in records:
        label_dir = Path(output_dir) / _sanitize_folder_name(rec["cluster_label"])
        dest = label_dir / Path(rec["original_path"]).name

        # Handle name collisions
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            counter = 1
            while dest.exists():
                dest = label_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        rec_out = {**rec, "output_path": str(dest)}
        results.append(rec_out)

        if not dry_run:
            label_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rec["original_path"], dest)

    return results


def generate_manifest(records: list[dict], output_dir: str) -> str:
    """Write manifest.csv and return its path."""
    manifest_path = Path(output_dir) / "manifest.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "original_path", "output_path", "cluster_label", "description"
        ])
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    return str(manifest_path)


def reveal_in_finder(path: str) -> None:
    """Open the given path in Finder (macOS)."""
    if platform.system() == "Darwin":
        subprocess.run(["open", path])


def _sanitize_folder_name(name: str) -> str:
    """Make a string safe for use as a folder name."""
    safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name)
    return safe.strip().replace(" ", "_") or "unnamed"
