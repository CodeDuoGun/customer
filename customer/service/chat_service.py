"""
Main chat service that orchestrates all components.
"""
import json
import time
from tkinter import NO
import uuid
from typing import Generator, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from customer.config.config import config
from customer.schema.chat import ChatRequest, ConversationMessage
from customer.service.intent_detector import IntentDetector
from customer.service.entity_extractor import EntityExtractor
from customer.service.doctor_search_service import DoctorSearchService
from customer.service.qa_search_service import QASearchService
from customer.service.conversation_service import conversation_service
from customer.service.llm_factory import llm_factory
from customer.utils.logger import logger
from customer.utils.constants import Speeches, MessageEventStatus, IntentTypes
from customer.utils.tools import generate_msg_id, chunk_text, limit_history_length


class ChatService:
    """Main chat service that orchestrates all components."""

    def __init__(
        self,
        intent_detector: Optional[IntentDetector] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        doctor_service: Optional[DoctorSearchService] = None,
        qa_service: Optional[QASearchService] = None
    ):
        """
        Initialize chat service.

        Args:
            intent_detector: Intent detection service
            entity_extractor: Entity extraction service
            doctor_service: Doctor search service
            qa_service: QA search service
        """
        self.intent_detector = intent_detector or IntentDetector()
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.doctor_service = doctor_service or DoctorSearchService()
        self.qa_service = qa_service or QASearchService()

        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=4)

    def process_chat_request(self, request: ChatRequest, backend_id: str) -> Generator[str, None, None]:
        """
        Process a chat request and return streaming response.

        Args:
            request: Chat request
            backend_id: Backend identifier for tracing

        Yields:
            SSE formatted response chunks
        """
        trace_id = f"{backend_id}_{request.conversation_id}"

        try:
            logger.info(f"{trace_id}: Processing chat request: {request.query}")

            # Initialize LLM
            llm = llm_factory.create_llm(request.model_name)

            # Initialize client based on model
            if request.model_name == "deepseek":
                llm.init_client(config.deepseek_api_key, config.deepseek_api_url)
            else:
                llm.init_client(config.ark_api_key, config.ark_base_url)

            # Get conversation history
            history = conversation_service.get_conversation(request.conversation_id)
            if history:
                history = limit_history_length(history)
            else:
                history = []
                conversation_service.create_conversation(request.conversation_id)

            # Clean query
            query = request.query.strip()

            # Prepare SSE data structure
            sse_data = {
                "conversation_id": request.conversation_id,
                "msg_id": generate_msg_id(),
                "event": MessageEventStatus.DELTA,
                "role": "assistant",
                "content": "",
                "content_type": "text",
                "reasoning_content": "",
                "finish_reasoning": False,
                "tools": []
            }

            # Add user message to history
            history.append({"role": "user", "content": query})

            # Handle special models (R1 with reasoning)
            if request.model_name == "coze_deepseek-r1":
                yield from self._handle_reasoning_model(llm, query, history, sse_data, request.conversation_id, trace_id)
                return

            # Concurrent intent detection and query rewriting
            future_intent = self.executor.submit(self._detect_intent_with_llm, llm, query, history, trace_id)
            future_rewrite = self.executor.submit(self._rewrite_query, llm, query, history, trace_id)

            # Wait for concurrent tasks
            intent = future_intent.result()
            rewritten_query = future_rewrite.result()

            logger.info(f"{trace_id}: Intent: {intent}, Rewritten: {rewritten_query}")

            # Route to appropriate handler based on intent
            if intent == IntentTypes.DISEASE_SEARCH:
                yield from self._handle_disease_search(llm, query, rewritten_query, history, sse_data, request, trace_id)
            elif intent == IntentTypes.DOCTOR_SEARCH:
                yield from self._handle_doctor_search(llm, query, rewritten_query, history, sse_data, request, trace_id)
            elif intent == IntentTypes.QA_SEARCH:
                yield from self._handle_qa_search(llm, query, rewritten_query, history, sse_data, request, trace_id)
            elif intent == IntentTypes.DOCTOR_INTRODUCE:
                yield from self._handle_doctor_introduction(llm, query, rewritten_query, history, sse_data, request, trace_id)
            else:  # other intent
                yield from self._handle_general_chat(llm, query, history, sse_data, request, trace_id)

        except Exception as e:
            logger.error(f"{trace_id}: Chat processing failed: {str(e)}")
            yield from self._handle_error(sse_data, trace_id)

    def _detect_intent_with_llm(self, llm, query: str, history: List[Dict], trace_id: str) -> str:
        """Detect intent using LLM."""
        try:
            self.intent_detector.llm = llm
            intent = self.intent_detector.detect_intent(query, f"{trace_id}_intent", history)
            return intent
        except Exception as e:
            logger.error(f"{trace_id}: Intent detection failed: {str(e)}")
            return "other"

    def _rewrite_query(self, llm, query: str, history: List[Dict], trace_id: str) -> str:
        """Rewrite query for better search."""
        try:
            rewrite_prompt = """你是一个专业的医疗查询改写专家。请根据对话历史，将用户的问题改写成更清晰、专业的医疗查询。

要求：
1. 保持原意，不要改变用户核心诉求
2. 使用规范的医疗术语
3. 补充必要的上下文信息
4. 如果问题已经很清晰，可以保持原样

直接输出改写后的问题，不要添加其他内容。"""

            messages = [
                {"role": "system", "content": rewrite_prompt}
            ]

            if history:
                messages.extend(history[-3:])  # Recent history

            messages.append({"role": "user", "content": query})

            response = llm.call_intent_stream(
                messages,
                chat_model=config.doubao_text_model,
                temperature=0.3,
                stream=False
            )

            rewritten = response.strip() if response else query
            logger.info(f"{trace_id}: Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.error(f"{trace_id}: Query rewrite failed: {str(e)}")
            return query

    def _handle_reasoning_model(self, llm, query: str, history: List[Dict], sse_data: Dict, conversation_id: str, trace_id: str):
        """Handle reasoning model (R1) requests."""
        try:
            think_prompt = """你是四惠医疗的智能健康顾问小惠。你善于运用中医理论和现代医学知识，深入思考并详细解答用户的问题。

请遵循以下原则：
1. 运用中医辨证论治思维，分析问题本质
2. 结合现代医学证据，提供科学建议
3. 深入思考问题的多个层面，给出全面解答
4. 保持专业、严谨、亲切的语气"""

            messages = [
                {"role": "system", "content": think_prompt}
            ]

            if history:
                messages.extend(history[-4:])

            # Stream reasoning response
            responses = llm.chat_coze_stream(
                sse_data, messages, history, conversation_id, trace_id, query,
                chat_model="coze_deepseek-r1", max_tokens=1000, temperature=0.8
            )

            for chunk in responses:
                yield chunk

        except Exception as e:
            logger.error(f"{trace_id}: Reasoning model failed: {str(e)}")
            yield from self._handle_error(sse_data, trace_id)

    def _handle_disease_search(self, llm, query: str, rewritten_query: str, history: List[Dict], sse_data: Dict, request: ChatRequest, trace_id: str):
        """Handle disease search intent."""
        try:
            # Extract entities and search doctors
            self.entity_extractor.llm = llm
            entities = self.entity_extractor.extract_entities(trace_id, query, rewritten_query)

            # Search doctors by disease
            doctors = self.doctor_service.search_doctors(llm, query, history, trace_id, rewritten_query)

            if not doctors:
                # No doctors found, return general response
                yield from self._send_chunked_response(sse_data, Speeches.NoDoctorSpeech, history, request.conversation_id, trace_id)
                return

            # Get doctor details
            doctor_ids = [doc.get("ID") for doc in doctors if doc.get("ID")]
            doctor_details = self.doctor_service.get_doctor_details(doctor_ids)

            if not doctor_details:
                yield from self._send_chunked_response(sse_data, Speeches.NoDoctorSpeech, history, request.conversation_id, trace_id)
                return

            # Send recommendation header
            yield from self._send_chunked_response(sse_data, Speeches.RecommendDoctorSpeech, history, request.conversation_id, trace_id)

            # Send doctor list
            sse_data["content_type"] = "object"
            sse_data["content"] = {"doctor_data": doctor_details}
            sse_data["event"] = MessageEventStatus.COMPLETED
            yield f"event: conversation\n\ndata: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

            # Update history
            history.append({"role": "assistant", "content": Speeches.RecommendDoctorSpeech})
            history.append({
                "role": "tool",
                "name": "doctor_search",
                "tool_call_id": "0",
                "content": json.dumps(doctor_details, ensure_ascii=False)
            })
            conversation_service.update_conversation(request.conversation_id, history, trace_id)

            yield "event: [DONE]\ndata: [DONE]\n\n"

        except Exception as e:
            logger.error(f"{trace_id}: Disease search failed: {str(e)}")
            yield from self._handle_error(sse_data, trace_id)

    def _handle_doctor_search(self, llm, query: str, rewritten_query: str, history: List[Dict], sse_data: Dict, request: ChatRequest, trace_id: str):
        """Handle doctor search intent."""
        # Similar to disease search but with different logic
        yield from self._handle_disease_search(llm, query, rewritten_query, history, sse_data, request, trace_id)

    def _handle_qa_search(self, llm, query: str, rewritten_query: str, history: List[Dict], sse_data: Dict, request: ChatRequest, trace_id: str):
        """Handle QA search intent."""
        try:
            # Perform parallel QA searches
            qa_results, qa_answer_results = self.qa_service.perform_parallel_searches(rewritten_query, query)

            if not qa_results and not qa_answer_results:
                # No results found
                error_msg = "您好！这个问题小惠还在学习中，您先联系人工看看，或者拨打客服热线400−689−6699吧。我会加速学习，来更好的为您服务！"
                yield from self._send_chunked_response(sse_data, error_msg, history, request.conversation_id, trace_id)
                return

            # Rerank results if reranker is available
            if hasattr(self.qa_service, 'reranker') and self.qa_service.reranker:
                reranked_results = self.qa_service.rerank_results(trace_id, rewritten_query, qa_results, qa_answer_results)
                most_similar_answer = ""
            else:
                # Find most similar question
                questions = [hit["_source"]["question"] for hit in qa_results]
                most_similar_answer, _ = self.qa_service.find_most_similar_question(
                    llm, trace_id, rewritten_query, qa_results, questions, history
                )
                reranked_results = qa_results

            # Prepare reference answers
            if reranked_results:
                final_refer_answers = "\n\n".join([hit['answer'] for hit in reranked_results[:6]])
                final_refer_answers = "\n参考资料:\n" + final_refer_answers
                if most_similar_answer:
                    final_refer_answers += "\n" + most_similar_answer
            else:
                final_refer_answers = most_similar_answer or "暂无相关信息"

            # Generate final answer
            for chunk in self.qa_service.generate_answer(
                llm, trace_id, query, final_refer_answers, history, sse_data, request.conversation_id
            ):
                yield chunk

        except Exception as e:
            logger.error(f"{trace_id}: QA search failed: {str(e)}")
            yield from self._handle_error(sse_data, trace_id)

    def _handle_doctor_introduction(self, llm, query: str, rewritten_query: str, history: List[Dict], sse_data: Dict, request: ChatRequest, trace_id: str):
        """Handle doctor introduction intent."""
        try:
            # Search doctors for introduction
            doctors = self.doctor_service.search_doctors(llm, query, history, trace_id, rewritten_query, must_name=True)

            if not doctors:
                yield from self._send_chunked_response(sse_data, Speeches.NoDoctorSpeech, history, request.conversation_id, trace_id)
                return

            # Introduce the first doctor
            first_doctor = doctors[0]
            for chunk in self.doctor_service.introduce_doctor(
                llm, first_doctor, sse_data, history, request.conversation_id, trace_id
            ):
                yield chunk

        except Exception as e:
            logger.error(f"{trace_id}: Doctor introduction failed: {str(e)}")
            yield from self._handle_error(sse_data, trace_id)

    def _handle_general_chat(self, llm, query: str, history: List[Dict], sse_data: Dict, request: ChatRequest, trace_id: str):
        """Handle general chat (fallback)."""
        try:
            # General knowledge prompt
            prompt = """你是四惠医疗的智能健康顾问小惠，擅长中医诊疗咨询。

技能:
1. 医学常识普及、疾病咨询、药品咨询等。从病因与症状、中医辨证、调理建议、预防措施等方面进行回复。
2. 针对四惠医疗线下医院，线上互联网医院的看诊问诊、预约挂号、中药使用、肿瘤治疗等问题，精准解答。
3. 针对四惠医疗的问题，如果没有相关信息，回复："小惠还在学习中，您可以先联系人工客服，或者拨打400-689−6699吧"
4. 在非医疗内容中保持中立、客观，不引导用户做出决定，需注明"仅供参考"。
5. 交通类问题解答

限制：
1. 医疗建议需注明"仅供参考，具体请咨询主治医师"
2. 对用药指导需特别谨慎，需注明"仅供参考，具体请咨询主治医师"
3. 涉及急症时提示立即就医
4. 保护患者隐私，不存储个人信息
5. 必须始终以"小惠"的口吻回答"""

            messages = [
                {"role": "system", "content": prompt}
            ]

            if history:
                messages.extend(history[-4:])

            # Stream response
            responses = llm.chat_coze_stream(
                sse_data, messages, history, request.conversation_id, trace_id, query,
                need_update_history=True, chat_model="coze_deepseek",
                need_stream_content=True
            )

            for chunk in responses:
                yield chunk

            # Add AI warning
            sse_data['content'] = f"\n{Speeches.AIWarningSpeech}"
            sse_data['content_type'] = "warning_signal"
            yield f"event: conversation\n\ndata: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
            yield "event: [DONE]\ndata: [DONE]\n\n"

        except Exception as e:
            logger.error(f"{trace_id}: General chat failed: {str(e)}")
            yield from self._handle_error(sse_data, trace_id)

    def _send_chunked_response(self, sse_data: Dict, content: str, history: List[Dict], conversation_id: str, trace_id: str):
        """Send chunked text response."""
        try:
            text_chunks = chunk_text(content, config.chunk_size)

            for chunk in text_chunks:
                sse_data["content"] = chunk
                yield f"event: conversation\n\ndata: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
                time.sleep(0.04)  # Control streaming speed

            # Send completion
            sse_data["event"] = MessageEventStatus.COMPLETED
            sse_data["content"] = content
            history.append({"role": "assistant", "content": content})
            conversation_service.update_conversation(conversation_id, history, trace_id)

            yield f"event: conversation\n\ndata: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
            yield "event: [DONE]\ndata: [DONE]\n\n"

        except Exception as e:
            logger.error(f"{trace_id}: Chunked response failed: {str(e)}")

    def _handle_error(self, sse_data: Dict, trace_id: str):
        """Handle errors in chat processing."""
        try:
            error_msg = "抱歉，小惠暂时遇到了一些问题，请稍后再试。"
            sse_data["content"] = error_msg
            sse_data["event"] = MessageEventStatus.COMPLETED

            yield f"event: conversation\n\ndata: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
            yield "event: [DONE]\ndata: [DONE]\n\n"

        except Exception as e:
            logger.error(f"{trace_id}: Error handling failed: {str(e)}")


# Global chat service instance
# chat_service = ChatService()
chat_service = None
