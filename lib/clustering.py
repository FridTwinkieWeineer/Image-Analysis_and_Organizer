"""Description embedding and clustering utilities."""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


def get_embedder():
    """Lazy-load sentence transformer model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_descriptions(descriptions: dict[str, str]) -> tuple[dict[str, list[float]], np.ndarray]:
    """
    Embed all descriptions using sentence-transformers.
    Returns (path->embedding dict, embedding matrix with rows in same order as paths).
    """
    if not descriptions:
        return {}, np.array([])

    model = get_embedder()
    paths = list(descriptions.keys())
    texts = [descriptions[p] for p in paths]
    embeddings = model.encode(texts, show_progress_bar=False)

    emb_dict = {}
    for i, path in enumerate(paths):
        emb_dict[path] = embeddings[i].tolist()

    return emb_dict, embeddings


def cluster_embeddings(
    embeddings: np.ndarray,
    paths: list[str],
    eps: float = 0.5,
    min_samples: int = 2,
) -> dict[str, int]:
    """
    Run DBSCAN on embeddings. Returns {path: cluster_id}.
    Outliers get cluster_id = -1.
    """
    if len(embeddings) == 0:
        return {}

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = db.fit_predict(embeddings)

    clusters = {}
    for i, path in enumerate(paths):
        clusters[path] = int(labels[i])
    return clusters


def find_similar(
    query_embedding: np.ndarray,
    all_embeddings: np.ndarray,
    paths: list[str],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Find top_k most similar images by cosine similarity.
    Returns list of (path, similarity_score) sorted descending.
    """
    if len(all_embeddings) == 0:
        return []

    query = query_embedding.reshape(1, -1)
    sims = cosine_similarity(query, all_embeddings)[0]

    indices = np.argsort(sims)[::-1][:top_k]
    return [(paths[i], float(sims[i])) for i in indices]
