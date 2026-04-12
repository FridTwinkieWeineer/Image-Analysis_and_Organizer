"""Persistent project state management via project.json."""

import json
import os
from pathlib import Path

DEFAULT_STATE = {
    "source_folder": "",
    "output_folder": "",
    "descriptions": {},      # {image_path: description_string}
    "embeddings": {},        # {image_path: [float, ...]}
    "clusters": {},          # {image_path: cluster_id}
    "labels": {},            # {cluster_id: label_string}
    "reassignments": {},     # {image_path: new_cluster_id} manual overrides
}

STATE_FILE = "project.json"


def get_state_path(base_dir: str = ".") -> Path:
    return Path(base_dir) / STATE_FILE


def load_state(base_dir: str = ".") -> dict:
    path = get_state_path(base_dir)
    if path.exists():
        with open(path, "r") as f:
            saved = json.load(f)
        # Merge with defaults so new keys are always present
        state = {**DEFAULT_STATE, **saved}
        return state
    return {**DEFAULT_STATE}


def save_state(state: dict, base_dir: str = ".") -> None:
    path = get_state_path(base_dir)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def update_description(state: dict, image_path: str, description: str) -> dict:
    state["descriptions"][image_path] = description
    return state


def clear_clusters(state: dict) -> dict:
    state["clusters"] = {}
    state["labels"] = {}
    state["reassignments"] = {}
    state["embeddings"] = {}
    return state
