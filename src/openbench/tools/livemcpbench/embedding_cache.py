"""
Persistent embedding cache for LiveMCPBench tools.

This module provides disk-based caching of tool embeddings to avoid
re-computing them on every evaluation run.
"""

import json
import os
import pickle
import hashlib
from typing import Dict, Optional, Any
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PersistentEmbeddingCache:
    """
    Disk-based cache for tool embeddings.

    Stores embeddings on disk so they persist across evaluation runs.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize persistent cache.

        Args:
            cache_dir: Directory to store cache files.
                      Defaults to ~/.cache/openbench/embeddings
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/openbench/embeddings")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.embeddings_file = self.cache_dir / "embeddings.pkl"

        self._load_cache()

    def _load_cache(self):
        """Load cache from disk."""
        self.metadata = {}
        self.embeddings = {}

        # Load metadata
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self.metadata = {}

        # Load embeddings
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, "rb") as f:
                    self.embeddings = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
                self.embeddings = {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            # Save metadata
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)

            # Save embeddings
            with open(self.embeddings_file, "wb") as f:
                pickle.dump(self.embeddings, f)

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _get_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model combination."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.

        Args:
            text: Text that was embedded
            model: Embedding model used

        Returns:
            Cached embedding or None if not found
        """
        key = self._get_key(text, model)
        return self.embeddings.get(key)

    def set(self, text: str, model: str, embedding: np.ndarray):
        """
        Store embedding in cache.

        Args:
            text: Text that was embedded
            model: Embedding model used
            embedding: The embedding vector
        """
        key = self._get_key(text, model)
        self.embeddings[key] = embedding

        # Update metadata
        self.metadata[key] = {
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "model": model,
            "dim": embedding.shape[0],
        }

        # Save to disk
        self._save_cache()

    def clear(self):
        """Clear all cached embeddings."""
        self.metadata = {}
        self.embeddings = {}
        self._save_cache()
        logger.info("Cleared embedding cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_embeddings": len(self.embeddings),
            "cache_size_mb": sum(emb.nbytes for emb in self.embeddings.values())
            / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
        }


# Global cache instance
_global_cache = None


def get_embedding_cache() -> PersistentEmbeddingCache:
    """Get or create global embedding cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = PersistentEmbeddingCache()
    return _global_cache
