"""Image Analysis & Organizer — Streamlit UI."""

import os
from collections import Counter

import numpy as np
import streamlit as st

from lib.state import load_state, save_state, update_description
from lib.image_utils import find_images, make_thumbnail
from lib.ollama_client import check_ollama_available, describe_image
from lib.clustering import embed_descriptions, cluster_embeddings, find_similar, get_embedder
from lib.organizer import (
    build_file_map, copy_to_folders, generate_manifest, reveal_in_finder, _sanitize_folder_name,
)

# ---------------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Image Organizer",
    page_icon="🖼️",
    layout="wide",
)

STATE_DIR = "."

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def init_session():
    if "state" not in st.session_state:
        st.session_state.state = load_state(STATE_DIR)
    if "phase" not in st.session_state:
        st.session_state.phase = "Setup"
    if "describe_running" not in st.session_state:
        st.session_state.describe_running = False
    if "describe_paused" not in st.session_state:
        st.session_state.describe_paused = False


def persist():
    save_state(st.session_state.state, STATE_DIR)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.title("Image Organizer")
        st.markdown("---")

        phases = ["Setup", "Describe", "Cluster", "Label & Merge", "Organize", "Find Similar"]
        for p in phases:
            if st.button(p, key=f"nav_{p}", use_container_width=True,
                         type="primary" if st.session_state.phase == p else "secondary"):
                st.session_state.phase = p
                st.rerun()

        st.markdown("---")
        state = st.session_state.state
        n_desc = len(state["descriptions"])
        n_clust = len(set(v for v in state["clusters"].values() if v != -1)) if state["clusters"] else 0
        n_labels = len([v for v in state["labels"].values() if v.lower() not in ("skip", "unsorted", "")])

        st.caption(f"Described: {n_desc} images")
        st.caption(f"Clusters: {n_clust}")
        st.caption(f"Labels: {n_labels}")

        st.markdown("---")
        ok, msg = check_ollama_available()
        if ok:
            st.success("Ollama: connected", icon="✅")
        else:
            st.error(f"Ollama: {msg}", icon="❌")


# ---------------------------------------------------------------------------
# Phase: Setup
# ---------------------------------------------------------------------------

def render_setup():
    st.header("Setup")
    state = st.session_state.state

    col1, col2 = st.columns(2)
    with col1:
        source = st.text_input("Source image folder", value=state["source_folder"],
                                placeholder="/path/to/your/images")
    with col2:
        output = st.text_input("Output folder", value=state["output_folder"],
                                placeholder="/path/to/output")

    state["source_folder"] = source
    state["output_folder"] = output
    persist()

    if source and os.path.isdir(source):
        images = find_images(source)
        already = sum(1 for img in images if img in state["descriptions"])
        st.info(f"Found **{len(images)}** images — **{already}** already described, **{len(images) - already}** remaining")

        if st.button("Start Describing →", type="primary"):
            st.session_state.phase = "Describe"
            st.rerun()
    elif source:
        st.warning("Source folder does not exist.")
    else:
        st.info("Enter a source folder path to get started.")


# ---------------------------------------------------------------------------
# Phase: Describe
# ---------------------------------------------------------------------------

def render_describe():
    st.header("Describe Images")
    state = st.session_state.state

    if not state["source_folder"] or not os.path.isdir(state["source_folder"]):
        st.warning("Go to Setup and set a valid source folder first.")
        return

    ok, msg = check_ollama_available()
    if not ok:
        st.error(msg)
        return

    images = find_images(state["source_folder"])
    remaining = [img for img in images if img not in state["descriptions"]]
    total = len(images)
    done = total - len(remaining)

    st.progress(done / total if total > 0 else 0, text=f"{done}/{total} described")

    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.button("▶ Start" if not st.session_state.describe_running else "Running...",
                           disabled=st.session_state.describe_running or len(remaining) == 0)
    with col2:
        pause = st.button("⏸ Pause", disabled=not st.session_state.describe_running)
    with col3:
        reanalyze = st.button("🔄 Re-analyze All")

    if reanalyze:
        state["descriptions"] = {}
        persist()
        st.rerun()

    if pause:
        st.session_state.describe_paused = True

    if start or st.session_state.describe_running:
        st.session_state.describe_running = True
        st.session_state.describe_paused = False

        progress_bar = st.progress(done / total if total > 0 else 0)
        status_area = st.empty()
        preview_area = st.container()

        for i, img_path in enumerate(remaining):
            if st.session_state.describe_paused:
                st.session_state.describe_running = False
                st.info("Paused. Click Start to resume.")
                persist()
                break

            status_area.text(f"Processing: {os.path.basename(img_path)}")

            desc, err = describe_image(img_path)

            if err:
                with preview_area:
                    st.warning(f"⚠️ {os.path.basename(img_path)}: {err}")
            else:
                state = update_description(state, img_path, desc)
                st.session_state.state = state

                # Save every 5 images to avoid data loss
                if (done + i + 1) % 5 == 0:
                    persist()

                with preview_area:
                    cols = st.columns([1, 3])
                    with cols[0]:
                        try:
                            thumb = make_thumbnail(img_path, (150, 150))
                            st.image(thumb, width=150)
                        except Exception:
                            st.text("(thumbnail error)")
                    with cols[1]:
                        st.text(desc[:300] + ("..." if len(desc) > 300 else ""))

            current_done = done + i + 1
            progress_bar.progress(current_done / total)

        else:
            # Loop completed without break
            st.session_state.describe_running = False
            persist()
            st.success("All images described!")

    # Show recent descriptions
    if state["descriptions"]:
        st.markdown("---")
        st.subheader("Recent Descriptions")
        items = list(state["descriptions"].items())[-6:]
        cols = st.columns(3)
        for idx, (path, desc) in enumerate(reversed(items)):
            with cols[idx % 3]:
                try:
                    thumb = make_thumbnail(path, (200, 200))
                    st.image(thumb, use_container_width=True)
                except Exception:
                    st.text("(image error)")
                st.caption(desc[:150] + "..." if len(desc) > 150 else desc)


# ---------------------------------------------------------------------------
# Phase: Cluster
# ---------------------------------------------------------------------------

def render_cluster():
    st.header("Cluster Images")
    state = st.session_state.state

    if not state["descriptions"]:
        st.warning("No descriptions yet. Run the Describe phase first.")
        return

    n_desc = len(state["descriptions"])
    st.info(f"**{n_desc}** images with descriptions ready for clustering.")

    col1, col2 = st.columns(2)
    with col1:
        eps = st.slider("DBSCAN epsilon (lower = tighter clusters)", 0.1, 2.0, 0.5, 0.05)
    with col2:
        min_samples = st.slider("Min samples per cluster", 2, 10, 2)

    if st.button("Run Clustering", type="primary"):
        with st.spinner("Embedding descriptions..."):
            emb_dict, emb_matrix = embed_descriptions(state["descriptions"])
            state["embeddings"] = emb_dict

        with st.spinner("Clustering..."):
            paths = list(state["descriptions"].keys())
            clusters = cluster_embeddings(emb_matrix, paths, eps=eps, min_samples=min_samples)
            state["clusters"] = clusters

            # Auto-generate default labels
            unique_ids = sorted(set(clusters.values()))
            labels = {}
            for cid in unique_ids:
                if cid == -1:
                    labels[str(cid)] = "Unclustered"
                else:
                    labels[str(cid)] = f"Cluster {cid + 1}"
            state["labels"] = labels
            state["reassignments"] = {}

        persist()
        st.rerun()

    # Display clusters
    if state["clusters"]:
        _render_cluster_grid(state)


def _render_cluster_grid(state):
    clusters = state["clusters"]
    labels = state["labels"]
    descriptions = state["descriptions"]

    # Group by cluster
    groups = {}
    for path, cid in clusters.items():
        effective_id = state.get("reassignments", {}).get(path, cid)
        groups.setdefault(effective_id, []).append(path)

    # Sort: named clusters first, unclustered last
    sorted_ids = sorted(groups.keys(), key=lambda x: (x == -1, x))

    for cid in sorted_ids:
        paths = groups[cid]
        label = labels.get(str(cid), f"Cluster {cid}")

        st.subheader(f"{label} ({len(paths)} images)")

        # Show as thumbnail grid
        n_cols = 5
        for row_start in range(0, len(paths), n_cols):
            row_paths = paths[row_start:row_start + n_cols]
            cols = st.columns(n_cols)
            for j, path in enumerate(row_paths):
                with cols[j]:
                    try:
                        thumb = make_thumbnail(path, (200, 200))
                        st.image(thumb, use_container_width=True)
                    except Exception:
                        st.text("(error)")
                    desc = descriptions.get(path, "")
                    st.caption(desc[:100] + "..." if len(desc) > 100 else desc)

        st.markdown("---")


# ---------------------------------------------------------------------------
# Phase: Label & Merge
# ---------------------------------------------------------------------------

def render_label():
    st.header("Label & Merge Clusters")
    state = st.session_state.state

    if not state["clusters"]:
        st.warning("No clusters yet. Run the Cluster phase first.")
        return

    clusters = state["clusters"]
    labels = state["labels"]
    descriptions = state["descriptions"]
    reassignments = state.get("reassignments", {})

    # Group by effective cluster
    groups = {}
    for path, cid in clusters.items():
        effective_id = reassignments.get(path, cid)
        groups.setdefault(effective_id, []).append(path)

    sorted_ids = sorted(groups.keys(), key=lambda x: (x == -1, x))

    changed = False

    for cid in sorted_ids:
        paths = groups[cid]
        current_label = labels.get(str(cid), f"Cluster {cid}")

        col_header, col_input = st.columns([1, 2])
        with col_header:
            st.subheader(f"{current_label} ({len(paths)} images)")
        with col_input:
            new_label = st.text_input(
                "Label",
                value=current_label,
                key=f"label_{cid}",
                label_visibility="collapsed",
            )
            if new_label != current_label:
                labels[str(cid)] = new_label
                changed = True

            mark_unsorted = st.checkbox("Mark as Unsorted", key=f"unsorted_{cid}",
                                        value=current_label.lower() == "unsorted")
            if mark_unsorted and new_label.lower() != "unsorted":
                labels[str(cid)] = "Unsorted"
                changed = True
            elif not mark_unsorted and current_label.lower() == "unsorted" and new_label.lower() == "unsorted":
                labels[str(cid)] = f"Cluster {cid}" if cid != -1 else "Unclustered"
                changed = True

        # Thumbnail grid with reassignment checkboxes
        n_cols = 6
        for row_start in range(0, len(paths), n_cols):
            row_paths = paths[row_start:row_start + n_cols]
            cols = st.columns(n_cols)
            for j, path in enumerate(row_paths):
                with cols[j]:
                    try:
                        thumb = make_thumbnail(path, (150, 150))
                        st.image(thumb, use_container_width=True)
                    except Exception:
                        st.text("(error)")

                    # Reassignment dropdown
                    other_options = [f"{labels.get(str(oid), f'Cluster {oid}')} (#{oid})"
                                     for oid in sorted_ids if oid != cid]
                    if other_options:
                        move_to = st.selectbox(
                            "Move to",
                            ["Keep here"] + other_options,
                            key=f"move_{path}",
                            label_visibility="collapsed",
                        )
                        if move_to != "Keep here":
                            # Parse target cluster id
                            target_id = int(move_to.split("(#")[1].rstrip(")"))
                            reassignments[path] = target_id
                            changed = True

        st.markdown("---")

    if changed:
        state["labels"] = labels
        state["reassignments"] = reassignments
        persist()

    # Merge info
    label_values = [v for v in labels.values() if v.lower() not in ("skip", "unsorted")]
    unique_labels = set(label_values)
    if len(unique_labels) < len(label_values):
        st.info("Clusters with the same label name will be merged during organization.")

    if st.button("Save Labels & Continue →", type="primary"):
        persist()
        st.session_state.phase = "Organize"
        st.rerun()


# ---------------------------------------------------------------------------
# Phase: Organize
# ---------------------------------------------------------------------------

def render_organize():
    st.header("Organize Files")
    state = st.session_state.state

    if not state["clusters"] or not state["labels"]:
        st.warning("Complete the Cluster and Label phases first.")
        return

    output_dir = state.get("output_folder", "")
    if not output_dir:
        st.warning("Set an output folder in Setup first.")
        return

    records = build_file_map(
        state["clusters"],
        state["labels"],
        state.get("reassignments", {}),
        state["descriptions"],
    )

    if not records:
        st.warning("No images to organize (all marked as skip/unsorted).")
        return

    # Summary table
    st.subheader("Summary")
    label_counts = Counter(r["cluster_label"] for r in records)

    summary_data = []
    for label, count in sorted(label_counts.items()):
        dest = os.path.join(output_dir, _sanitize_folder_name(label))
        summary_data.append({"Label": label, "Images": count, "Destination": dest})

    st.table(summary_data)
    st.text(f"Total: {len(records)} images to organize")

    dry_run = st.toggle("Dry run (preview only)", value=True)

    if st.button("Organize Files", type="primary"):
        with st.spinner("Organizing..."):
            results = copy_to_folders(records, output_dir, dry_run=dry_run)
            manifest_path = generate_manifest(results, output_dir)

        if dry_run:
            st.info("Dry run complete — no files were copied.")
            st.subheader("Would copy:")
            for r in results[:20]:
                st.text(f"  {os.path.basename(r['original_path'])} → {r['output_path']}")
            if len(results) > 20:
                st.text(f"  ... and {len(results) - 20} more")
        else:
            st.success(f"Organized {len(results)} files!")
            st.text(f"Manifest: {manifest_path}")

            if st.button("Reveal in Finder"):
                reveal_in_finder(output_dir)


# ---------------------------------------------------------------------------
# Phase: Find Similar
# ---------------------------------------------------------------------------

def render_find_similar():
    st.header("Find Similar Images")
    state = st.session_state.state

    if not state["descriptions"]:
        st.warning("No descriptions yet. Run the Describe phase first.")
        return

    uploaded = st.file_uploader("Upload a reference image", type=["jpg", "jpeg", "png", "webp", "gif"])

    if uploaded:
        st.image(uploaded, width=300, caption="Reference image")

        if st.button("Find Similar", type="primary"):
            # Save temp file and describe it
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            with st.spinner("Describing reference image..."):
                desc, err = describe_image(tmp_path)

            os.unlink(tmp_path)

            if err:
                st.error(f"Could not describe reference image: {err}")
                return

            st.info(f"**Reference description:** {desc[:300]}")

            with st.spinner("Finding similar images..."):
                model = get_embedder()
                query_emb = model.encode([desc])

                # Get all embeddings
                if state["embeddings"]:
                    paths = list(state["embeddings"].keys())
                    all_embs = np.array([state["embeddings"][p] for p in paths])
                else:
                    emb_dict, all_embs = embed_descriptions(state["descriptions"])
                    paths = list(emb_dict.keys())
                    state["embeddings"] = emb_dict
                    persist()

                results = find_similar(query_emb, all_embs, paths, top_k=12)

            st.subheader("Most Similar Images")
            n_cols = 4
            for row_start in range(0, len(results), n_cols):
                row = results[row_start:row_start + n_cols]
                cols = st.columns(n_cols)
                for j, (path, score) in enumerate(row):
                    with cols[j]:
                        try:
                            thumb = make_thumbnail(path, (200, 200))
                            st.image(thumb, use_container_width=True)
                        except Exception:
                            st.text("(error)")
                        st.caption(f"Similarity: {score:.3f}")
                        desc = state["descriptions"].get(path, "")
                        st.caption(desc[:100] + "..." if len(desc) > 100 else desc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_session()
    render_sidebar()

    phase = st.session_state.phase
    if phase == "Setup":
        render_setup()
    elif phase == "Describe":
        render_describe()
    elif phase == "Cluster":
        render_cluster()
    elif phase == "Label & Merge":
        render_label()
    elif phase == "Organize":
        render_organize()
    elif phase == "Find Similar":
        render_find_similar()


if __name__ == "__main__":
    main()
