# Image Analysis & Organizer

A local image organization tool with a Streamlit web UI that uses Ollama's vision model to analyze AI-generated images and automatically organize them by the fictional characters depicted.

Point it at a folder of images, let the AI describe each one, cluster similar characters together, label them with your own names, and copy everything into neatly organized subfolders — all through a visual interface.

## Features

- **AI-Powered Descriptions** — Uses Ollama's `llama3.2-vision` model running locally to generate detailed character descriptions (hair, clothing, accessories, art style)
- **Smart Clustering** — Embeds descriptions with sentence-transformers and groups similar characters using DBSCAN with an adjustable sensitivity slider
- **Visual Interface** — Thumbnail grids for every phase; no terminal workflow required
- **Manual Refinement** — Rename clusters, reassign individual images between groups, mark clusters as unsorted
- **Non-Destructive** — Always copies files; originals are never moved or modified
- **Resumable** — Progress is saved to `project.json` automatically; close and reopen without losing work
- **Find Similar** — Upload a reference image to find the most visually similar images in your collection

## Screenshots

The app walks you through 6 phases via the sidebar:

| Phase | What It Does |
|-------|-------------|
| **Setup** | Pick source/output folders, see image count |
| **Describe** | Ollama analyzes each image with live progress and thumbnails |
| **Cluster** | DBSCAN groups similar characters into visual grids |
| **Label & Merge** | Name your clusters, drag images between groups |
| **Organize** | Preview and execute the file copy with manifest CSV |
| **Find Similar** | Upload a reference image, see ranked matches |

## Requirements

- **Python 3.10+**
- **Ollama** running locally with the `llama3.2-vision` model
- **16GB+ RAM** recommended (32GB+ ideal for smooth performance)
- **GPU recommended** — NVIDIA GPU with VRAM will significantly speed up vision analysis

## Quick Start

```bash
# 1. Install Ollama and pull the vision model
# https://ollama.com/download
ollama pull llama3.2-vision

# 2. Clone and install
git clone https://github.com/FridTwinkieWeineer/Image-Analysis_and_Organizer.git
cd Image-Analysis_and_Organizer
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## How It Works

1. **Describe** — Each image is resized to 1024px, base64-encoded, and sent to Ollama's chat API with a character-description prompt. Results are cached in `project.json`.
2. **Cluster** — Descriptions are embedded using `all-MiniLM-L6-v2` (sentence-transformers) and clustered with DBSCAN. Epsilon controls cluster tightness.
3. **Label** — You name each cluster (e.g., "red-haired warrior", "blonde mage"). Same-name clusters merge automatically.
4. **Organize** — Files are copied into labeled subfolders. A `manifest.csv` maps every file to its original path, destination, cluster label, and description.

## Project Structure

```
├── app.py                  # Streamlit UI (all 6 phases)
├── lib/
│   ├── state.py            # Project state persistence
│   ├── image_utils.py      # Image discovery, resizing, base64
│   ├── ollama_client.py    # Ollama vision API client
│   ├── clustering.py       # Embeddings + DBSCAN
│   └── organizer.py        # File copy + manifest generation
├── requirements.txt
└── project.json            # Auto-generated state (gitignored)
```

## Tips

- **Timeout errors?** Your machine may not have enough RAM for the vision model. Try on a machine with 32GB+ RAM or a dedicated GPU.
- **Want tighter clusters?** Lower the epsilon slider in the Cluster phase.
- **Re-describe images?** Use the "Re-analyze All" button in the Describe phase.
- **Dry run first** — The Organize phase defaults to dry-run mode so you can preview before copying.

## License

MIT
