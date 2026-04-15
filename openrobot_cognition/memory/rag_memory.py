"""Layer 4: RAG-based long-term memory using ChromaDB."""

import uuid
from typing import List, Dict, Any, Optional


class RAGMemory:
    """Simple vector-store memory for robot skills and episodic experiences."""

    def __init__(self, collection_name: str = "openrobot_memory", embedding_function=None):
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb is required for RAGMemory")

        self.client = chromadb.Client()
        self.embedding_fn = embedding_function
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )

    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        doc_id = str(uuid.uuid4())
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id],
        )
        return doc_id

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            {
                "id": ids[i],
                "text": docs[i],
                "metadata": metas[i],
                "distance": distances[i] if distances else None,
            }
            for i in range(len(ids))
        ]

    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_fn,
        )
