import os
import faiss
import numpy as np
import json
import logging
import threading
import pickle
from typing import Tuple, List

logger = logging.getLogger("database")

class VectorDB:
    def __init__(self, db_path: str = "data/vectordb", embedding_size: int = 512):
        self.db_path = db_path
        self.embedding_size = embedding_size
        self.index_file = os.path.join(db_path, "faiss_index.bin")
        self.meta_file = os.path.join(db_path, "metadata.json")
        
        # Thread safety lock
        self.lock = threading.RLock()
        
        # Internal storage
        self.index = None
        self.metadata = [] # List of names corresponding to index ID
        
        # Load or Create
        if not self.load():
            self._create_new_index()

    def _create_new_index(self):
        os.makedirs(self.db_path, exist_ok=True)
        # IndexFlatIP = Inner Product (Cosine similarity if vectors are normalized)
        self.index = faiss.IndexFlatIP(self.embedding_size)
        self.metadata = []
        logger.info("Created new Vector Database")

    def add_face(self, embedding: np.ndarray, name: str):
        """Add a single face to the index."""
        # Ensure embedding is 2D array [1, 512] and float32
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)
        
        # L2 Normalize (Critical for ArcFace/Cosine Similarity)
        faiss.normalize_L2(embedding)
        
        with self.lock:
            self.index.add(embedding.astype(np.float32))
            self.metadata.append(name)
            logger.info(f"Added face: {name} (Total: {len(self.metadata)})")

    def search(self, embedding: np.ndarray, threshold: float = 0.45) -> Tuple[str, float]:
        """Search for the closest face."""
        if self.index.ntotal == 0:
            return "Unknown", 0.0

        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)
        
        faiss.normalize_L2(embedding)
        
        with self.lock:
            # Search for top 1
            similarities, indices = self.index.search(embedding.astype(np.float32), 1)
            
        sim_score = float(similarities[0][0])
        idx = indices[0][0]
        
        if sim_score > threshold and idx != -1 and idx < len(self.metadata):
            return self.metadata[idx], sim_score
        
        return "Unknown", sim_score

    def save(self):
        """Persist index to disk."""
        with self.lock:
            try:
                faiss.write_index(self.index, self.index_file)
                with open(self.meta_file, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, ensure_ascii=False)
                logger.info("Database saved to disk")
            except Exception as e:
                logger.error(f"Failed to save database: {e}")

    def load(self) -> bool:
        """Load index from disk."""
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.meta_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded database with {self.index.ntotal} identities")
                return True
            except Exception as e:
                logger.error(f"Corrupt database found, recreating: {e}")
                return False
        return False