"""
Doctor search and recommendation service using LlamaIndex with hybrid retrieval.

混合检索说明：
- 使用 Elasticsearch 的 AsyncDenseVectorStrategy(hybrid=True)
- 结合文本搜索（基于治疗特色等字段）和向量搜索（基于goodvector字段）
- 支持 RRF (Reciprocal Rank Fusion) 结果融合
- 如果向量字段为空，请先运行 update_doctor_vectors() 方法填充向量数据

使用示例：

1. 混合检索医生：
   service = DoctorSearchService()
   results = service.hybrid_search("不孕症治疗", similarity_top_k=10)
   # 返回包含完整医生信息的列表

2. 更新向量数据：
   service.update_doctor_vectors(batch_size=100)
   # 从治疗特色字段生成向量并存储到goodvector字段

3. 使用自定义向量检索：
   embed_model = DoubaoEmbeddings()
   query_vector = embed_model.get_text_embedding("心脏病专家")
   results = service.retrieve_with_custom_vector(query_vector)

4. 标准医生搜索（会自动使用混合检索）：
   doctors = service.search_doctors(llm_instance, "不孕症", history, trace_id)
"""
from math import log
import requests
from typing import List, Dict, Any, Optional, Generator
import json
import time
from customer.config.config import config
from customer.service.embeddings import DoubaoEmbeddings
from customer.utils.logger import logger
from customer.utils.constants import Speeches, MessageEventStatus
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore, AsyncDenseVectorStrategy
from customer.utils.tools import chunk_text, generate_msg_id

def print_results(results):
    for rank, result in enumerate(results, 1):
        print(
            f"{rank}. title={result.metadata['title']} score={result.get_score()} text={result.get_text()}"
        )

class DoctorSearchService:
    """Service for searching and recommending doctors."""

    def __init__(self, entity_extractor=None, embed_model=None, index=None):
        """
        Initialize doctor search service.

        Args:
            entity_extractor: Entity extraction service
            embed_model: Embedding model for custom vector retrieval
            index: LlamaIndex index for doctor search
        """
        self.entity_extractor = entity_extractor
        self.embed_model = embed_model
        self.vector_store = ElasticsearchStore(
            index_name="alpha_doctor",
            es_url=f"http://{config.ES_HOST}:{config.ES_PORT}",
            es_user=config.ES_USER,
            es_password=config.ES_AUTH,
            retrieval_strategy=AsyncDenseVectorStrategy(hybrid=True),
            # 明确指定字段映射
            vector_field="goodvector",
            text_field="擅长"
            )

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=embed_model,
            storage_context=storage_context,
        )

        # 不创建 query_engine，避免依赖 OpenAI LLM
        # self.query_engine = self.index.as_query_engine()
        self.query_engine = None

        # 初始化混合检索器 - 配置ES混合检索参数
        self.hybrid_retriever = self.index.as_retriever(
            similarity_top_k=10,
            vector_store_kwargs={
                # Elasticsearch 混合检索配置
                "search_type": "hybrid",  # 明确指定混合搜索
                "hybrid": True,
                "text_field": "擅长",  # 文本字段
                "vector_field": "goodvector",  # 向量字段
                "num_candidates": 100,  # 候选数量
            }
        )


    def get_full_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """
        根据文档ID从ES获取完整的文档数据。

        Args:
            doc_id: 文档ID

        Returns:
            完整的文档数据
        """
        try:
            es_url = f"http://{config.ES_HOST}:{config.ES_PORT}"
            index_name = "alpha_doctor_info"
            url = f"{es_url}/{index_name}/_doc/{doc_id}"

            auth = (config.ES_USER, config.ES_AUTH)
            response = requests.get(url, auth=auth)

            if response.status_code == 200:
                doc_data = response.json()
                return {
                    "_id": doc_id,
                    "_index": index_name,
                    **doc_data.get("_source", {}),
                    **doc_data.get("fields", {})  # 包含fields中的数据
                }
            else:
                logger.warning(f"Failed to get document {doc_id}: {response.status_code}")
                return {}

        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {str(e)}")
            return {}

    def get_full_documents_batch(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量获取多个文档的完整数据。

        Args:
            doc_ids: 文档ID列表

        Returns:
            ID到文档数据的映射
        """
        try:
            es_url = f"http://{config.ES_HOST}:{config.ES_PORT}"
            index_name = "alpha_doctor_info"
            url = f"{es_url}/{index_name}/_mget"

            auth = (config.ES_USER, config.ES_AUTH)
            body = {"ids": doc_ids}

            response = requests.post(url, auth=auth, json=body)

            if response.status_code == 200:
                result = response.json()
                docs_map = {}

                for doc in result.get("docs", []):
                    if doc.get("found"):
                        doc_id = doc["_id"]
                        # 合并 _source 和 fields 数据，优先使用fields中的数据（因为ES查询可能返回fields）
                        source_data = doc.get("_source", {})
                        fields_data = doc.get("fields", {})

                        # 如果fields中有数组类型的数据，需要展开
                        processed_fields = {}
                        for key, value in fields_data.items():
                            if isinstance(value, list) and len(value) == 1:
                                processed_fields[key] = value[0]  # 展开单元素数组
                            else:
                                processed_fields[key] = value

                        full_doc = {
                            "_id": doc_id,
                            "_index": index_name,
                            **source_data,
                            **processed_fields  # 展开后的fields数据
                        }
                        docs_map[doc_id] = full_doc

                return docs_map
            else:
                logger.warning(f"Failed to batch get documents: {response.status_code}")
                return {}

        except Exception as e:
            logger.error(f"Error batch getting documents: {str(e)}")
            return {}

    def hybrid_search(self, query: str, similarity_top_k: int = 10) -> List[Dict[str, Any]]:
        """
        执行混合检索，直接返回ES文档的完整metadata信息。

        Args:
            query: 搜索查询
            similarity_top_k: 返回的相似结果数量

        Returns:
            ES文档的完整字段数据列表，包含所有metadata
        """
        try:
            logger.info(f"Performing direct ES hybrid search for query: {query}")

            # 直接使用ES进行混合搜索，返回完整metadata
            es_results = self._direct_es_hybrid_search(query, similarity_top_k)

            if es_results:
                logger.info(f"Direct ES hybrid search returned {len(es_results)} complete documents")
                return es_results

            # 如果直接ES搜索失败，回退到LlamaIndex方式
            logger.warning("Direct ES search failed, falling back to LlamaIndex retriever")
            return self._fallback_llamaindex_search(query, similarity_top_k)

        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._fallback_llamaindex_search(query, similarity_top_k)

    def _fallback_llamaindex_search(self, query: str, similarity_top_k: int = 10) -> List[Dict[str, Any]]:
        """
        LlamaIndex后备搜索方法，当直接ES搜索失败时使用。

        Args:
            query: 搜索查询
            similarity_top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        try:
            logger.info("Using LlamaIndex fallback search")

            # 尝试使用混合检索器
            results = self.hybrid_retriever.retrieve(query)

            if not results:
                # 尝试简单检索器
                retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
                results = retriever.retrieve(query)

            if not results:
                return []

            # 转换结果为完整文档格式
            formatted_results = []
            for result in results:
                doctor_info = result.node.metadata.copy()
                doctor_info["score"] = result.score
                doctor_info["content"] = result.node.text
                doctor_info["_id"] = result.node.node_id
                doctor_info["retrieval_method"] = "llamaindex_fallback"
                doctor_info["query"] = query
                formatted_results.append(doctor_info)

            logger.info(f"LlamaIndex fallback search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"LlamaIndex fallback search failed: {str(e)}")
            return []

    def _direct_es_hybrid_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        直接使用ES API进行混合搜索，返回完整的文档metadata。

        Args:
            query: 搜索查询
            limit: 返回结果数量

        Returns:
            包含所有ES文档字段的完整数据列表
        """
        try:
            es_url = f"http://{config.ES_HOST}:{config.ES_PORT}"
            index_name = "alpha_doctor"
            search_url = f"{es_url}/{index_name}/_search"

            # 获取查询向量
            query_vector = self.embed_model.get_text_embedding(query)
            logger.info(f"Query vector dimension: {len(query_vector)}")

            # 构建混合查询：结合文本搜索和向量搜索
            search_body = {
                "size": limit,
                "query": {
                    "bool": {
                        "should": [
                            # 文本搜索 - 在多个字段中搜索
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["擅长^2", "姓名"],
                                    "type": "best_fields",
                                    "boost": 1.5
                                }
                            },
                            # 向量搜索 - 余弦相似度
                            {
                                "script_score": {
                                    "query": {
                                        "exists": {"field": "goodvector"}
                                    },
                                    "script": {
                                        "source": """
                                            double similarity = cosineSimilarity(params.query_vector, 'goodvector');
                                            return similarity > 0 ? similarity : 0;
                                        """,
                                        "params": {
                                            "query_vector": query_vector
                                        }
                                    },
                                    "boost": 2.0
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "sort": [
                    {"_score": {"order": "desc"}}
                ],
                "_source": True  # 返回所有字段
            }

            logger.info(f"Direct ES hybrid search with query: '{query}'")
            response = requests.post(search_url, auth=(config.ES_USER, config.ES_AUTH), json=search_body)

            if response.status_code == 200:
                data = response.json()
                hits = data.get('hits', {}).get('hits', [])

                logger.info(f"Direct ES hybrid search returned {len(hits)} results")

                # 直接返回ES文档的完整数据
                results = []
                for hit in hits:
                    doc_data = hit['_source']
                    # 添加ES特有的字段
                    doc_data['_id'] = hit['_id']
                    doc_data['_index'] = hit['_index']
                    doc_data['_score'] = hit['_score']
                    # 添加一些额外的检索信息
                    doc_data['retrieval_method'] = 'direct_es_hybrid'
                    doc_data['query'] = query

                    results.append(doc_data)

                return results
            else:
                logger.error(f"Direct ES hybrid search failed: {response.status_code}, {response.text}")
                return []

        except Exception as e:
            logger.error(f"Direct ES hybrid search error: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _direct_es_search(self, query: str, limit: int = 10) -> List[Any]:
        """
        直接使用ES API进行混合搜索，作为最后的后备方案

        Args:
            query: 搜索查询
            limit: 返回结果数量

        Returns:
            模拟的检索结果列表
        """
        try:
            es_url = f"http://{config.ES_HOST}:{config.ES_PORT}"
            index_name = "alpha_doctor_info"
            search_url = f"{es_url}/{index_name}/_search"

            # 获取查询向量
            query_vector = self.embed_model.get_text_embedding(query)
            logger.info(f"Query vector dimension: {len(query_vector)}")

            # 构建混合查询：结合文本搜索和向量搜索
            search_body = {
                "size": limit,
                "query": {
                    "bool": {
                        "should": [
                            # 文本搜索 - 在多个字段中搜索
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["治疗特色^2", "擅长病种", "姓名", "主要成就"],
                                    "type": "best_fields",
                                    "boost": 1.5
                                }
                            },
                            # 向量搜索 - 余弦相似度
                            {
                                "script_score": {
                                    "query": {
                                        "exists": {"field": "goodvector"}
                                    },
                                    "script": {
                                        "source": """
                                            double similarity = cosineSimilarity(params.query_vector, 'goodvector');
                                            return similarity > 0 ? similarity : 0;
                                        """,
                                        "params": {
                                            "query_vector": query_vector
                                        }
                                    },
                                    "boost": 2.0
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "sort": [
                    {"_score": {"order": "desc"}}
                ],
                "_source": ["治疗特色", "goodvector", "姓名", "职称/职务", "擅长病种"]
            }

            logger.info(f"Direct ES hybrid search with query: '{query}'")
            response = requests.post(search_url, auth=(config.ES_USER, config.ES_AUTH), json=search_body)

            if response.status_code == 200:
                data = response.json()
                hits = data.get('hits', {}).get('hits', [])

                logger.info(f"Direct ES search returned {len(hits)} results")

                # 转换为与LlamaIndex结果类似的格式
                results = []
                for hit in hits:
                    # 创建一个模拟的检索结果对象
                    class MockResult:
                        def __init__(self, node_id, score, metadata):
                            self.node = type('MockNode', (), {
                                'node_id': node_id,
                                'text': metadata.get('治疗特色', ''),
                                'metadata': metadata
                            })()
                            self.score = score

                    # 使用ES文档ID作为node_id
                    node_id = f"doctor_{hit['_id']}"
                    score = hit['_score'] or 0.0
                    metadata = hit['_source']

                    results.append(MockResult(node_id, score, metadata))

                return results
            else:
                logger.error(f"Direct ES search failed: {response.status_code}, {response.text}")
                logger.error(f"Response body: {response.text[:500]}")
                return []

        except Exception as e:
            logger.error(f"Direct ES search error: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def update_doctor_vectors(self, batch_size: int = 100) -> bool:
        """
        批量更新ES中医生数据的向量字段。
        从治疗特色等文本字段生成向量并存储到goodvector字段。

        Args:
            batch_size: 每批处理的文档数量

        Returns:
            是否成功
        """
        try:
            # ES连接信息
            es_url = f"http://{config.ES_HOST}:{config.ES_PORT}"
            auth = (config.ES_USER, config.ES_AUTH)
            index_name = "alpha_doctor_info"

            # 首先获取所有文档
            search_url = f"{es_url}/{index_name}/_search"
            scroll_url = f"{es_url}/_search/scroll"

            # 初始搜索请求
            search_body = {
                "size": batch_size,
                "query": {"match_all": {}},
                "scroll": "2m"
            }

            response = requests.post(search_url, auth=auth, json=search_body)
            if response.status_code != 200:
                logger.error(f"Failed to search documents: {response.text}")
                return False

            data = response.json()
            scroll_id = data["_scroll_id"]
            hits = data["hits"]["hits"]

            processed_count = 0

            while hits:
                logger.info(f"Processing batch of {len(hits)} documents...")

                # 批量更新向量
                bulk_updates = []
                for hit in hits:
                    doc_id = hit["_id"]
                    source = hit["_source"]

                    # 从治疗特色等字段生成向量
                    text_content = source.get("治疗特色", "")
                    if text_content:
                        # 生成向量
                        embedding = self.embed_model.get_text_embedding(text_content)

                        if embedding:
                            # 构建更新请求
                            bulk_updates.extend([
                                {"update": {"_id": doc_id, "_index": index_name}},
                                {"doc": {"goodvector": embedding}}
                            ])

                # 执行批量更新
                if bulk_updates:
                    bulk_url = f"{es_url}/_bulk"
                    bulk_body = "\n".join([json.dumps(item) for item in bulk_updates]) + "\n"

                    bulk_response = requests.post(
                        bulk_url,
                        auth=auth,
                        data=bulk_body,
                        headers={"Content-Type": "application/x-ndjson"}
                    )

                    if bulk_response.status_code == 200:
                        bulk_data = bulk_response.json()
                        if bulk_data.get("errors"):
                            logger.warning(f"Some updates failed in batch")
                        else:
                            logger.info(f"Successfully updated {len(bulk_updates)//2} documents")
                    else:
                        logger.error(f"Bulk update failed: {bulk_response.text}")

                processed_count += len(hits)

                # 获取下一批数据
                scroll_body = {
                    "scroll": "2m",
                    "scroll_id": scroll_id
                }

                response = requests.post(scroll_url, auth=auth, json=scroll_body)
                if response.status_code != 200:
                    logger.error(f"Failed to scroll: {response.text}")
                    break

                data = response.json()
                scroll_id = data["_scroll_id"]
                hits = data["hits"]["hits"]

            logger.info(f"Vector update completed. Processed {processed_count} documents.")
            return True

        except Exception as e:
            logger.error(f"Vector update failed: {str(e)}")
            return False


    def retrieve_with_custom_vector(self, query_vector: List[float], similarity_top_k: int = 5) -> List[Dict[str, Any]]:
        """
        使用自定义向量进行检索（而不是文本查询）。

        注意：LlamaIndex 的标准检索器通常不直接支持预计算向量。
        这个方法主要用于演示概念，实际使用时可能需要：
        1. 使用向量存储的底层 API
        2. 创建自定义检索器
        3. 或在查询时使用嵌入模型的特定配置

        Args:
            query_vector: 预计算的查询向量
            similarity_top_k: 返回的相似结果数量

        Returns:
            检索结果列表
        """
        try:
            if not self.vector_store:
                logger.warning("Vector store not available for custom vector retrieval")
                return []

            # 方法1：使用向量存储的底层查询接口（如果支持）
            try:
                # 尝试使用 Elasticsearch 向量存储的查询方法
                results = self.vector_store.query(
                    query_embedding=query_vector,
                    similarity_top_k=similarity_top_k
                )

                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "content": getattr(result, 'text', str(result)),
                        "score": getattr(result, 'score', 0.0),
                        "metadata": getattr(result, 'metadata', {}),
                        "node_id": getattr(result, 'id', "")
                    })

                logger.info(f"Custom vector retrieval returned {len(formatted_results)} results")
                return formatted_results

            except AttributeError:
                # 方法2：如果向量存储不支持，直接使用嵌入模型进行文本检索
                logger.warning("Vector store doesn't support direct embedding queries, falling back to text retrieval")
                return []

        except Exception as e:
            logger.error(f"Custom vector retrieval failed: {str(e)}")
            return []

    def retrieve_with_text_and_custom_embedding(self, query: str, custom_embed_model=None, similarity_top_k: int = 5) -> List[Dict[str, Any]]:
        """
        使用自定义嵌入模型对文本进行编码，然后进行检索。

        Args:
            query: 文本查询
            custom_embed_model: 自定义嵌入模型，如果为None则使用默认模型
            similarity_top_k: 返回的相似结果数量

        Returns:
            检索结果列表
        """
        try:
            if not self.index:
                logger.warning("Index not available for custom embedding retrieval")
                return []

            # 使用指定的嵌入模型
            embed_model = custom_embed_model or self.embed_model
            if embed_model:
                # 生成查询向量
                query_embedding = embed_model.get_query_embedding(query)

                # 使用向量进行检索
                return self.retrieve_with_custom_vector(query_embedding, similarity_top_k)
            else:
                # 回退到标准文本检索
                retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
                results = retriever.retrieve(query)

                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "content": result.node.text,
                        "score": result.score,
                        "metadata": result.node.metadata,
                        "node_id": result.node.node_id
                    })

                return formatted_results

        except Exception as e:
            logger.error(f"Custom embedding retrieval failed: {str(e)}")
            return []

    def search_doctors(
        self,
        llm,
        user_query: str,
        history: List[Dict[str, Any]],
        trace_id: str,
        rewritten_query: str = "",
        must_name: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for doctors based on user query.

        Args:
            llm: LLM instance
            user_query: User query
            history: Conversation history
            trace_id: Trace ID
            rewritten_query: Rewritten query
            must_name: Whether doctor name is required

        Returns:
            List of doctor information
        """
        try:
            # Extract entities from query
            if self.entity_extractor:
                entities = self.entity_extractor.extract_entities(
                    trace_id, user_query, rewritten_query, must_name, history
                )
            else:
                # Fallback entity extraction
                entities = {
                    "doctor_name": "",
                    "doctor_location": "",
                    "condition": user_query,
                    "doctor_hospital": ""
                }

            name = entities.get("doctor_name", "")
            condition = entities.get("condition", user_query)
            doctor_location = entities.get("doctor_location", "")
            doctor_hospital = entities.get("doctor_hospital", "")

            # Determine search strategy
            by_name = bool(name) or must_name
            if must_name and not name:
                # If doctor name is required but not found, return empty
                logger.info(f"{trace_id}: Doctor name required but not found")
                return []

            # Search doctors
            if by_name:
                # Search by doctor name - use name field by default
                doctors = self._search_doctors_by_name(trace_id, name, search_fields=["姓名"])
            else:
                # Search by condition and location
                doctors = self._search_doctors_by_condition(
                    trace_id, condition, doctor_hospital, doctor_location
                )

            # Remove duplicates and limit results
            unique_doctors = self._deduplicate_doctors(doctors)
            limited_doctors = unique_doctors[:8]  # Limit to 8 results

            logger.info(f"{trace_id}: Found {len(limited_doctors)} doctors")
            return limited_doctors

        except Exception as e:
            logger.error(f"{trace_id}: Doctor search failed: {str(e)}")
            return []

    def _search_doctors_by_name(self, trace_id: str, name: str, search_fields: List[str] = None) -> List[Dict[str, Any]]:
        """Search doctors by keyword in specified fields.

        Args:
            trace_id: Trace ID for logging
            name: Keyword to search for
            search_fields: List of fields to search in. If None, defaults to ["姓名"]

        Returns:
            List of doctor information
        """
        try:
            # Default search fields if not specified
            if search_fields is None:
                search_fields = ["姓名"]

            # Split keywords by comma if multiple
            keywords = [k.strip() for k in name.split(",") if k.strip()]

            all_doctors = []
            for keyword in keywords:
                # Use Elasticsearch for keyword search
                doctors = self._keyword_search(trace_id, keyword, search_fields, limit=5)
                all_doctors.extend(doctors)

            logger.info(f"{trace_id}: Found {len(all_doctors)} doctors by keyword '{name}' in fields {search_fields}")
            return all_doctors

        except Exception as e:
            logger.error(f"{trace_id}: Doctor keyword search failed: {str(e)}")
            return []

    def _keyword_search(self, trace_id: str, keyword: str, search_fields: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Perform keyword search in specified Elasticsearch fields.

        Args:
            trace_id: Trace ID for logging
            keyword: Keyword to search for
            search_fields: List of fields to search in
            limit: Maximum number of results to return

        Returns:
            List of doctor information
        """
        try:
            es_url = f"http://{config.ES_HOST}:{config.ES_PORT}"
            index_name = "alpha_doctor_info"
            search_url = f"{es_url}/{index_name}/_search"

            # Build multi_match query for keyword search
            search_body = {
                "size": limit,
                "query": {
                    "multi_match": {
                        "query": keyword,
                        "fields": search_fields,
                        "type": "best_fields",
                        "fuzziness": "AUTO",  # Allow fuzzy matching
                        "operator": "or"
                    }
                },
                "sort": [
                    {"_score": {"order": "desc"}}
                ],
                "_source": True  # Return all fields
            }

            logger.info(f"{trace_id}: Keyword search for '{keyword}' in fields {search_fields}")

            response = requests.post(search_url, auth=(config.ES_USER, config.ES_AUTH), json=search_body)

            if response.status_code == 200:
                data = response.json()
                hits = data.get('hits', {}).get('hits', [])

                logger.info(f"{trace_id}: Keyword search returned {len(hits)} results")

                results = []
                for hit in hits:
                    doc_data = hit['_source']
                    # Add ES-specific fields
                    doc_data['_id'] = hit['_id']
                    doc_data['_index'] = hit['_index']
                    doc_data['_score'] = hit['_score']
                    # Add retrieval metadata
                    doc_data['retrieval_method'] = 'keyword_search'
                    doc_data['search_keyword'] = keyword
                    doc_data['search_fields'] = search_fields

                    results.append(doc_data)

                return results
            else:
                logger.error(f"{trace_id}: Keyword search failed: {response.status_code}, {response.text}")
                return []

        except Exception as e:
            logger.error(f"{trace_id}: Keyword search error: {str(e)}")
            return []

    def _search_doctors_by_condition(
        self,
        trace_id: str,
        condition: str,
        hospital: str = "",
        location: str = ""
    ) -> List[Dict[str, Any]]:
        """Search doctors by medical condition."""
        try:
            # Build search query
            query_parts = [condition]
            if hospital:
                query_parts.append(f"医院：{hospital}")
            if location:
                query_parts.append(f"地点：{location}")

            search_query = " ".join(query_parts)
            logger.info(f"search_query: {search_query}")
            doctors = []
            if self.index:
                # 使用混合检索（结合文本和向量搜索）
                results = self.hybrid_search(search_query, similarity_top_k=10)
                logger.info(f"Hybrid search results: {len(results)} found")

                for result in results:
                    doctor_info = result
                    doctor_info["score"] = result.get("_score", 0.0)
                    doctors.append(doctor_info)
            else:
                # Fallback: return mock data for demonstration
                logger.warning(f"{trace_id}: No index available, using mock data")
                doctors = [
                    {
                        "ID": "mock_doctor_1",
                        "姓名": "李医生",
                        "简介": "从事中医临床工作20年",
                        "擅长": "内科常见疾病的中医治疗",
                        "出诊地点": "四惠中医医院",
                        "score": 0.85
                    }
                ]

            logger.info(f"{trace_id}: Found {len(doctors)} doctors by condition")
            return doctors

        except Exception as e:
            logger.error(f"{trace_id}: Doctor condition search failed: {str(e)}")
            return []

    def _deduplicate_doctors(self, doctors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate doctors based on ID."""
        seen_ids = set()
        unique_doctors = []

        for doctor in doctors:
            doctor_id = doctor.get("ID", doctor.get("id", ""))
            if doctor_id and doctor_id not in seen_ids:
                seen_ids.add(doctor_id)
                unique_doctors.append(doctor)

        return unique_doctors

    def get_doctor_details(self, doctor_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get detailed doctor information by IDs.

        Args:
            doctor_ids: List of doctor IDs

        Returns:
            List of detailed doctor information
        """
        try:
            if not config.doctor_search_url:
                logger.warning("Doctor search URL not configured")
                return []

            response = requests.post(
                config.doctor_search_url,
                json={"doctor_ids": doctor_ids},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                doctors = result.get("data", [])
                logger.info(f"Retrieved {len(doctors)} doctor details")
                return doctors
            else:
                logger.error(f"Doctor search API failed: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Doctor details retrieval failed: {str(e)}")
            return []

    def introduce_doctor(
        self,
        llm,
        doctor_info: Dict[str, Any],
        sse_data: Dict[str, Any],
        history: List[Dict[str, Any]],
        conversation_id: str,
        trace_id: str
    ) -> Generator[str, None, None]:
        """
        Generate doctor introduction using LLM.

        Args:
            llm: LLM instance
            doctor_info: Doctor information
            sse_data: SSE data structure
            history: Conversation history
            conversation_id: Conversation ID
            trace_id: Trace ID

        Yields:
            SSE formatted response chunks
        """
        try:
            # Build doctor info text
            doctor_text = f"{doctor_info.get('姓名', '医生')}"
            if doctor_info.get('简介'):
                doctor_text += f"\n简介：{doctor_info.get('简介', '').strip('、')}"
            if doctor_info.get('出诊地点'):
                doctor_text += f"\n出诊地点：{doctor_info.get('出诊地点', '').strip('、')}"

            specialty = doctor_info.get('擅长', '').strip('、')

            # Doctor introduction prompt
            prompt = f"""你是一个专业的医疗助手，擅长介绍医生信息。根据你现有的知识，请根据以下医生资料:
{doctor_text}
擅长：{specialty}

生成一段口语化的介绍.

限制：
    1.禁止给出医生资料之外的信息，如联系方式、出诊费用、出诊时间。
    2.只输出医生的介绍，包括：姓名、简介、出诊地点、擅长。严格禁止输出其他内容。
    3.最后请用自然、亲切的语气结束，例如："如果您有需要，我可以帮您推荐合适的医生进行挂号～还有其他想了解的吗？"
    4.对于资料中没有的信息，请明确指出"暂时没有相关信息"或"资料中未提及"，并提示：您也可以直接描述您的症状，小惠会尽力为您推荐合适的医生哟。
"""

            messages = [
                {"role": "system", "content": prompt}
            ]

            # Add history context
            if history:
                messages.extend(history[-4:])  # Recent history

            # Stream response
            responses = llm.chat_coze_stream(
                sse_data, messages, history, conversation_id, trace_id,
                "介绍医生", chat_model="coze_deepseek",
                need_stream_content=True, need_update_history=True
            )

            for chunk in responses:
                yield chunk

            logger.info(f"{trace_id}: Doctor introduction completed")

        except Exception as e:
            logger.error(f"{trace_id}: Doctor introduction failed: {str(e)}")
            # Return error message
            error_chunks = chunk_text("抱歉，小惠暂时无法生成医生介绍。请稍后再试。")
            for chunk in error_chunks:
                sse_data["content"] = chunk
                yield f"event: conversation\n\ndata: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
                time.sleep(0.04)

            sse_data["event"] = MessageEventStatus.COMPLETED
            yield f"event: conversation\n\ndata: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
            yield "event: [DONE]\ndata: [DONE]\n\n"
