"""
Index builder for creating LangChain documents from various data sources.
"""
import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from customer.config.config import config
from customer.utils.logger import logger


class IndexBuilder:
    """Builder for creating LangChain documents from various data sources."""

    def __init__(self, embed_model=None, text_splitter=None):
        """
        Initialize index builder.

        Args:
            embed_model: Embedding model to use
            text_splitter: Text splitter for document processing
        """
        self.embed_model = embed_model
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
        )

    def build_from_directory(
        self,
        directory_path: str,
        index_name: str = "directory_index",
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
        **kwargs
    ) -> List[Document]:
        """
        Build documents from a directory.

        Args:
            directory_path: Path to directory containing documents
            index_name: Name for the index
            file_extensions: File extensions to include
            recursive: Whether to search recursively
            **kwargs: Additional arguments

        Returns:
            List of documents
        """
        try:
            if not os.path.exists(directory_path):
                raise ValueError(f"Directory does not exist: {directory_path}")

            logger.info(f"Building documents from directory: {directory_path}")

            # Create directory loader
            loader_kwargs = {
                "path": directory_path,
                "recursive": recursive
            }

            if file_extensions:
                # Convert extensions to glob patterns
                glob_patterns = [f"*.{ext}" for ext in file_extensions]
                loader_kwargs["glob"] = glob_patterns

            loader = DirectoryLoader(**loader_kwargs)
            documents = loader.load()

            logger.info(f"Loaded {len(documents)} documents from directory")
            return documents

        except Exception as e:
            logger.error(f"Failed to build from directory: {str(e)}")
            raise

    def build_from_files(
        self,
        file_paths: List[str],
        index_name: str = "files_index",
        **kwargs
    ) -> List[Document]:
        """
        Build documents from file paths.

        Args:
            file_paths: List of file paths
            index_name: Name for the index
            **kwargs: Additional arguments

        Returns:
            List of documents
        """
        try:
            logger.info(f"Building documents from {len(file_paths)} files")

            documents = []
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"File does not exist: {file_path}")
                    continue
                #  TODO 不同文件不同loader
                loader = TextLoader(file_path)
                file_documents = loader.load()
                documents.extend(file_documents)

            logger.info(f"Loaded {len(documents)} documents from files")
            return documents

        except Exception as e:
            logger.error(f"Failed to build from files: {str(e)}")
            raise

    def build_from_data(
        self,
        data: List[Dict[str, Any]],
        index_name: str = "data_index",
        text_field: str = "content",
        metadata_fields: Optional[List[str]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Build documents from structured data.

        Args:
            data: List of data dictionaries
            index_name: Name for the index
            text_field: Field containing the text content
            metadata_fields: Fields to include as metadata
            **kwargs: Additional arguments

        Returns:
            List of documents
        """
        try:
            logger.info(f"Building documents from {len(data)} data items")

            documents = []
            for item in data:
                # Extract text content
                text = item.get(text_field, "")
                if not text:
                    continue

                # Extract metadata
                metadata = {}
                if metadata_fields:
                    for field in metadata_fields:
                        if field in item:
                            metadata[field] = item[field]

                # Create document
                doc = Document(
                    page_content=text,
                    metadata=metadata,
                    **kwargs
                )
                documents.append(doc)

            logger.info(f"Created {len(documents)} documents from data")
            return documents

        except Exception as e:
            logger.error(f"Failed to build from data: {str(e)}")
            raise

    def build_doctor_index(
        self,
        doctor_data: List[Dict[str, Any]],
        index_name: str = "doctor_index"
    ) -> List[Document]:
        """
        Build doctor index from doctor data.

        Args:
            doctor_data: List of doctor information
            index_name: Name for the index

        Returns:
            List of documents
        """
        try:
            logger.info(f"Building doctor index with {len(doctor_data)} doctors")

            documents = []
            for doctor in doctor_data:
                # Build text content from doctor info
                text_parts = []

                if doctor.get("姓名"):
                    text_parts.append(f"医生姓名：{doctor['姓名']}")

                if doctor.get("简介"):
                    text_parts.append(f"医生简介：{doctor['简介']}")

                if doctor.get("擅长"):
                    text_parts.append(f"医生擅长：{doctor['擅长']}")

                if doctor.get("出诊地点"):
                    text_parts.append(f"出诊地点：{doctor['出诊地点']}")

                if doctor.get("职称"):
                    text_parts.append(f"医生职称：{doctor['职称']}")

                text = "\n".join(text_parts)

                if not text:
                    continue

                # Create metadata
                metadata = {
                    "type": "doctor",
                    "doctor_id": doctor.get("ID", doctor.get("id", "")),
                    "name": doctor.get("姓名", ""),
                    "specialty": doctor.get("擅长", ""),
                    "hospital": doctor.get("出诊地点", ""),
                    "title": doctor.get("职称", "")
                }

                # Create document
                doc = Document(
                    page_content=text,
                    metadata=metadata,
                    id=f"doctor_{doctor.get('ID', doctor.get('id', ''))}"
                )
                documents.append(doc)

            logger.info(f"Created {len(documents)} doctor documents")
            return documents

        except Exception as e:
            logger.error(f"Failed to build doctor index: {str(e)}")
            raise

    def build_qa_index(
        self,
        qa_data: List[Dict[str, Any]],
        index_name: str = "qa_index"
    ) -> List[Document]:
        """
        Build QA index from Q&A data.

        Args:
            qa_data: List of Q&A pairs
            index_name: Name for the index

        Returns:
            List of documents
        """
        try:
            logger.info(f"Building QA index with {len(qa_data)} Q&A pairs")

            documents = []
            for qa in qa_data:
                question = qa.get("question", "")
                answer = qa.get("answer", "")

                if not question or not answer:
                    continue

                # Create text content
                text = f"问题：{question}\n答案：{answer}"

                # Create metadata
                metadata = {
                    "type": "qa",
                    "question": question,
                    "answer": answer,
                    "category": qa.get("category", ""),
                    "tags": qa.get("tags", [])
                }

                # Create document
                doc = Document(
                    page_content=text,
                    metadata=metadata,
                    id=f"qa_{hash(question)}"
                )
                documents.append(doc)

            logger.info(f"Created {len(documents)} QA documents")
            return documents

        except Exception as e:
            logger.error(f"Failed to build QA index: {str(e)}")
            raise

    def process_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[Document]:
        """
        Process documents into chunks.

        Args:
            documents: List of documents to process
            **kwargs: Additional arguments for text splitter

        Returns:
            List of document chunks
        """
        try:
            logger.info(f"Processing {len(documents)} documents into chunks")

            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents, **kwargs)

            logger.info(f"Created {len(chunks)} chunks from documents")
            return chunks

        except Exception as e:
            logger.error(f"Failed to process documents: {str(e)}")
            raise
