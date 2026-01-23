"""
Vector store management for LlamaIndex.
"""
from typing import Optional, List, Dict, Any
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document
from customer.utils.logger import logger


class VectorStoreManager:
    """Manager for LlamaIndex vector stores and indices."""

    def __init__(self):
        """Initialize vector store manager."""
        self.indices = {}  # Cache for loaded indices
        self.vector_stores = {}  # Cache for vector stores

    def create_index_from_documents(
        self,
        documents: List[Document],
        index_name: str = "default",
        embed_model = None,
        **kwargs
    ) -> VectorStoreIndex:
        """
        Create a vector store index from documents.

        Args:
            documents: List of documents to index
            index_name: Name for the index
            embed_model: Embedding model to use
            **kwargs: Additional arguments for index creation

        Returns:
            Created vector store index
        """
        try:
            logger.info(f"Creating index '{index_name}' with {len(documents)} documents")

            # Create index
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=embed_model,
                **kwargs
            )

            # Cache the index
            self.indices[index_name] = index

            logger.info(f"Successfully created index '{index_name}'")
            return index

        except Exception as e:
            logger.error(f"Failed to create index '{index_name}': {str(e)}")
            raise

    def load_index_from_storage(
        self,
        persist_dir: str,
        index_name: str = "default"
    ) -> VectorStoreIndex:
        """
        Load index from persistent storage.

        Args:
            persist_dir: Directory where index is stored
            index_name: Name for the loaded index

        Returns:
            Loaded vector store index
        """
        try:
            logger.info(f"Loading index '{index_name}' from {persist_dir}")

            # Load storage context
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

            # Load index
            index = VectorStoreIndex.from_storage(storage_context)

            # Cache the index
            self.indices[index_name] = index

            logger.info(f"Successfully loaded index '{index_name}'")
            return index

        except Exception as e:
            logger.error(f"Failed to load index '{index_name}': {str(e)}")
            raise

    def save_index(
        self,
        index: VectorStoreIndex,
        persist_dir: str,
        index_name: str = "default"
    ) -> bool:
        """
        Save index to persistent storage.

        Args:
            index: Index to save
            persist_dir: Directory to save to
            index_name: Name of the index

        Returns:
            Success status
        """
        try:
            logger.info(f"Saving index '{index_name}' to {persist_dir}")

            # Save index
            index.storage_context.persist(persist_dir=persist_dir)

            logger.info(f"Successfully saved index '{index_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to save index '{index_name}': {str(e)}")
            return False

    def get_index(self, index_name: str = "default") -> Optional[VectorStoreIndex]:
        """
        Get a cached index by name.

        Args:
            index_name: Name of the index

        Returns:
            Cached index or None
        """
        return self.indices.get(index_name)

    def search_index(
        self,
        index_name: str,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search an index with a query.

        Args:
            index_name: Name of the index to search
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional search arguments

        Returns:
            List of search results
        """
        try:
            index = self.get_index(index_name)
            if not index:
                logger.warning(f"Index '{index_name}' not found")
                return []

            # Perform search
            retriever = index.as_retriever(similarity_top_k=top_k, **kwargs)
            results = retriever.retrieve(query)

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.node.text,
                    "score": result.score,
                    "metadata": result.node.metadata,
                    "node_id": result.node.node_id
                })

            logger.info(f"Search '{index_name}' for '{query}' returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed for index '{index_name}': {str(e)}")
            return []

    def add_documents_to_index(
        self,
        index_name: str,
        documents: List[Document]
    ) -> bool:
        """
        Add documents to an existing index.

        Args:
            index_name: Name of the index
            documents: Documents to add

        Returns:
            Success status
        """
        try:
            index = self.get_index(index_name)
            if not index:
                logger.warning(f"Index '{index_name}' not found")
                return False

            # Add documents
            for doc in documents:
                index.insert(doc)

            logger.info(f"Added {len(documents)} documents to index '{index_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents to index '{index_name}': {str(e)}")
            return False
