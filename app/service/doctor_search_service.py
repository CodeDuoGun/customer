"""
Doctor search and recommendation service using LlamaIndex.
"""
import requests
from typing import List, Dict, Any, Optional, Generator
import json
import time
from customer.config.config import config
from customer.utils.logger import logger
from customer.utils.constants import Speeches, MessageEventStatus
from customer.utils.tools import chunk_text, generate_msg_id


class DoctorSearchService:
    """Service for searching and recommending doctors."""

    def __init__(self, entity_extractor=None, vector_store=None, index=None):
        """
        Initialize doctor search service.

        Args:
            entity_extractor: Entity extraction service
            vector_store: Vector store for doctor data
            index: LlamaIndex index for doctor search
        """
        self.entity_extractor = entity_extractor
        self.vector_store = vector_store
        self.index = index

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

            logger.info(f"{trace_id}: Extracted entities: {entities}")

            # Determine search strategy
            by_name = bool(name) or must_name
            if must_name and not name:
                # If doctor name is required but not found, return empty
                logger.info(f"{trace_id}: Doctor name required but not found")
                return []

            # Search doctors
            if by_name:
                # Search by doctor name
                doctors = self._search_doctors_by_name(trace_id, name)
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

    def _search_doctors_by_name(self, trace_id: str, name: str) -> List[Dict[str, Any]]:
        """Search doctors by name."""
        try:
            # Split names by comma
            doctor_names = [n.strip() for n in name.split(",") if n.strip()]

            all_doctors = []
            for doctor_name in doctor_names:
                # Use vector store or index for search
                if self.index:
                    # Use LlamaIndex retrieval
                    retriever = self.index.as_retriever(similarity_top_k=5)
                    results = retriever.retrieve(f"医生姓名：{doctor_name}")

                    for result in results:
                        doctor_info = result.node.metadata.copy()
                        doctor_info["score"] = result.score
                        all_doctors.append(doctor_info)
                else:
                    # Fallback: simulate search
                    logger.warning(f"{trace_id}: No index available for doctor search")
                    continue

            logger.info(f"{trace_id}: Found {len(all_doctors)} doctors by name")
            return all_doctors

        except Exception as e:
            logger.error(f"{trace_id}: Doctor name search failed: {str(e)}")
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

            doctors = []
            if self.index:
                # Use LlamaIndex for hybrid search
                retriever = self.index.as_retriever(similarity_top_k=10)
                results = retriever.retrieve(search_query)

                for result in results:
                    doctor_info = result.node.metadata.copy()
                    doctor_info["score"] = result.score
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
