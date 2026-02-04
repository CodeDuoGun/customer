"""
Milvus processor for data insertion and retrieval operations.
"""
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Generator
import numpy as np
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema,
    DataType, utility, AnnSearchRequest, RRFRanker
)
from customer.config.config import config
from customer.utils.logger import logger
from customer.service.embeddings import DoubaoEmbeddings


class MilvusProcessor:
    """Milvus processor for handling vector database operations."""

    def __init__(self, host: str = None, port: str = None):
        """
        Initialize Milvus processor.

        Args:
            host: Milvus host
            port: Milvus port
        """
        self.host = host or config.MILVUS_HOST
        self.port = port or config.MILVUS_PORT
        self.connection_alias = "default"
        self.embed_model = DoubaoEmbeddings()

        # Connect to Milvus
        self._connect()

    def _connect(self):
        """Connect to Milvus server."""
        try:
            connections.connect(
                alias=self.connection_alias,
                host=self.host,
                port=self.port
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise

    def disconnect(self):
        """Disconnect from Milvus."""
        try:
            connections.disconnect(alias=self.connection_alias)
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to disconnect from Milvus: {str(e)}")

    def create_collection(
        self,
        collection_name: str,
        fields_config: Dict[str, Any],
        description: str = ""
    ) -> bool:
        """
        Create a collection with specified schema.

        Args:
            collection_name: Name of the collection
            fields_config: Field configurations
            description: Collection description

        Returns:
            Success status
        """
        try:
            # Drop existing collection if exists
            if utility.has_collection(collection_name, using=self.connection_alias):
                utility.drop_collection(collection_name, using=self.connection_alias)
                logger.info(f"Dropped existing collection: {collection_name}")

            # Create field schemas
            fields = []
            for field_name, field_config in fields_config.items():
                field_type = field_config.get("type")
                if field_type == "VARCHAR":
                    fields.append(FieldSchema(
                        name=field_name,
                        dtype=DataType.VARCHAR,
                        max_length=field_config.get("max_length", 65535),
                        is_primary=field_config.get("is_primary", False)
                    ))
                elif field_type == "INT64":
                    fields.append(FieldSchema(
                        name=field_name,
                        dtype=DataType.INT64,
                        is_primary=field_config.get("is_primary", False)
                    ))
                elif field_type == "FLOAT_VECTOR":
                    fields.append(FieldSchema(
                        name=field_name,
                        dtype=DataType.FLOAT_VECTOR,
                        dim=field_config.get("dim", 4096)
                    ))
                elif field_type == "JSON":
                    fields.append(FieldSchema(
                        name=field_name,
                        dtype=DataType.JSON
                    ))

            # Create collection schema
            schema = CollectionSchema(
                fields=fields,
                description=description
            )

            # Create collection
            collection = Collection(
                name=collection_name,
                schema=schema,
                using=self.connection_alias
            )

            logger.info(f"Created collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {str(e)}")
            return False

    def insert_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        text_field: str = "content",
        vector_field: str = "vector"
    ) -> bool:
        """
        Insert documents with vector embeddings.

        Args:
            collection_name: Target collection name
            documents: List of documents to insert
            text_field: Field name containing text content
            vector_field: Field name for vector embeddings

        Returns:
            Success status
        """
        try:
            collection = Collection(collection_name, using=self.connection_alias)

            # Prepare data for insertion
            insert_data = []
            field_names = [field.name for field in collection.schema.fields]

            for doc in documents:
                row_data = {}
                text_content = doc.get(text_field, "")

                # Generate embedding if text field exists and vector field is needed
                if text_content and vector_field in field_names:
                    embedding = self.embed_model.embed_query(text_content)
                    doc[vector_field] = embedding

                # Prepare data for each field
                for field_name in field_names:
                    if field_name in doc:
                        row_data[field_name] = doc[field_name]
                    elif field_name == "id":
                        # Auto-generate ID if not provided
                        row_data[field_name] = str(doc.get("id", hash(str(doc))))

                insert_data.append(row_data)

            # Insert data in batches
            batch_size = 1000
            for i in range(0, len(insert_data), batch_size):
                batch = insert_data[i:i + batch_size]
                collection.insert(batch)
                logger.info(f"Inserted batch {i//batch_size + 1} with {len(batch)} documents")

            # Flush to ensure data persistence
            collection.flush()

            # Build index for vector field if exists
            if vector_field in field_names:
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 256}
                }
                collection.create_index(vector_field, index_params)
                logger.info(f"Created index for vector field: {vector_field}")

            # Load collection for search
            collection.load()

            logger.info(f"Successfully inserted {len(documents)} documents into {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to insert documents into {collection_name}: {str(e)}")
            return False

    def insert_qa_pairs(
        self,
        collection_name: str,
        qa_pairs: List[Dict[str, Any]],
        embed_model=None
    ) -> bool:
        """
        Insert QA pairs with embeddings.

        Args:
            collection_name: Target collection name
            qa_pairs: List of QA pairs
            embed_model: Embedding model to use

        Returns:
            Success status
        """
        try:
            collection = Collection(collection_name, using=self.connection_alias)

            # Prepare data for insertion
            insert_data = []
            field_names = [field.name for field in collection.schema.fields]

            embed_model = embed_model or self.embed_model

            for qa in qa_pairs:
                row_data = {}

                # Generate embeddings for question and answer
                question = qa.get("question", "")
                answer = qa.get("answer", "")

                if question:
                    q_embedding = embed_model.embed_query(question)
                    qa["q_vector"] = q_embedding

                if answer:
                    a_embedding = embed_model.embed_query(answer)
                    qa["answer_vector"] = a_embedding

                # Prepare data for each field
                for field_name in field_names:
                    if field_name in qa:
                        row_data[field_name] = qa[field_name]
                    elif field_name == "id":
                        row_data[field_name] = str(qa.get("id", hash(question)))

                insert_data.append(row_data)

            # Insert data
            collection.insert(insert_data)
            collection.flush()

            # Create indexes
            if "q_vector" in field_names:
                collection.create_index("q_vector", {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 256}
                })

            if "answer_vector" in field_names:
                collection.create_index("answer_vector", {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 256}
                })

            collection.load()

            logger.info(f"Successfully inserted {len(qa_pairs)} QA pairs into {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to insert QA pairs into {collection_name}: {str(e)}")
            return False

    def hybrid_search(
        self,
        collection_name: str,
        query: str,
        text_field: str = "question",
        vector_field: str = "q_vector",
        limit: int = 10,
        text_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and text search.

        Args:
            collection_name: Collection to search
            query: Search query
            text_field: Text field for filtering
            vector_field: Vector field for similarity search
            limit: Number of results to return
            text_filter: Optional text filter

        Returns:
            Search results
        """
        try:
            collection = Collection(collection_name, using=self.connection_alias)

            # Generate query embedding
            query_embedding = self.embed_model.embed_query(query)

            # Prepare search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 128}
            }

            # Vector search request
            vector_search = AnnSearchRequest(
                data=[query_embedding],
                anns_field=vector_field,
                search_params=search_params,
                limit=limit * 2  # Get more candidates for reranking
            )

            # Combine searches with RRF reranking
            search_requests = [vector_search]
            rerank = RRFRanker(k=limit)

            # Execute search
            results = collection.hybrid_search(
                search_requests,
                rerank=rerank,
                limit=limit,
                output_fields=["*"]
            )

            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "score": hit.score,
                        "entity": hit.entity
                    }
                    formatted_results.append(result)

            logger.info(f"Hybrid search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            return []

    def vector_search(
        self,
        collection_name: str,
        query: str,
        vector_field: str = "vector",
        limit: int = 10,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Args:
            collection_name: Collection to search
            query: Search query
            vector_field: Vector field name
            limit: Number of results
            filter_expr: Optional filter expression

        Returns:
            Search results
        """
        try:
            collection = Collection(collection_name, using=self.connection_alias)

            # Generate query embedding
            query_embedding = self.embed_model.embed_query(query)

            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 128}
            }

            # Execute search
            results = collection.search(
                data=[query_embedding],
                anns_field=vector_field,
                search_params=search_params,
                limit=limit,
                expr=filter_expr,
                output_fields=["*"]
            )

            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "score": hit.score,
                        "entity": hit.entity
                    }
                    formatted_results.append(result)

            logger.info(f"Vector search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    def text_search(
        self,
        collection_name: str,
        query: str,
        text_field: str = "content",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform text-based search using filtering.

        Args:
            collection_name: Collection to search
            query: Search query
            text_field: Text field name
            limit: Number of results

        Returns:
            Search results
        """
        try:
            collection = Collection(collection_name, using=self.connection_alias)

            # Create text filter expression
            filter_expr = f'{text_field} like "%{query}%"'

            # Query with filter
            results = collection.query(
                expr=filter_expr,
                output_fields=["*"],
                limit=limit
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id"),
                    "score": 1.0,  # Text search doesn't have scores
                    "entity": result
                })

            logger.info(f"Text search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Text search failed: {str(e)}")
            return []


# Global instance
milvus_processor = MilvusProcessor()
