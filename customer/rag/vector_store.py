"""
Vector store management for LangChain.
"""
from typing import Optional, List, Dict, Any
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from customer.utils.logger import logger


class VectorStoreManager:
    """Manager for LangChain vector stores."""

    def __init__(self):
        """Initialize vector store manager."""
        self.vector_stores = {}  # Cache for vector stores

    def add_vector_store(
        self,
        vector_store: VectorStore,
        store_name: str = "default"
    ) -> None:
        """
        Add a vector store to the manager.

        Args:
            vector_store: Vector store instance to add
            store_name: Name for the vector store
        """
        self.vector_stores[store_name] = vector_store
        logger.info(f"Added vector store '{store_name}'")

    def get_vector_store(self, store_name: str = "default") -> Optional[VectorStore]:
        """
        Get a cached vector store by name.

        Args:
            store_name: Name of the vector store

        Returns:
            Cached vector store or None
        """
        return self.vector_stores.get(store_name)

    def search_vector_store(
        self,
        store_name: str,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search a vector store with a query.

        Args:
            store_name: Name of the vector store to search
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional search arguments

        Returns:
            List of search results
        """
        try:
            vector_store = self.get_vector_store(store_name)
            if not vector_store:
                logger.warning(f"Vector store '{store_name}' not found")
                return []

            # Perform similarity search
            docs_and_scores = vector_store.similarity_search_with_score(query, k=top_k, **kwargs)

            # Format results
            formatted_results = []
            for doc, score in docs_and_scores:
                formatted_results.append({
                    "content": doc.page_content,
                    "score": score,
                    "metadata": doc.metadata,
                    "doc_id": getattr(doc, 'id', None)
                })

            logger.info(f"Search '{store_name}' for '{query}' returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed for vector store '{store_name}': {str(e)}")
            return []

    def add_documents_to_vector_store(
        self,
        store_name: str,
        documents: List[Document]
    ) -> bool:
        """
        Add documents to an existing vector store.

        Args:
            store_name: Name of the vector store
            documents: Documents to add

        Returns:
            Success status
        """
        try:
            vector_store = self.get_vector_store(store_name)
            if not vector_store:
                logger.warning(f"Vector store '{store_name}' not found")
                return False

            # Add documents
            vector_store.add_documents(documents)

            logger.info(f"Added {len(documents)} documents to vector store '{store_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents to vector store '{store_name}': {str(e)}")
            return False
