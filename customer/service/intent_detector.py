"""
Intent detection service using LlamaIndex.
"""
from typing import List, Dict, Any, Optional
from customer.config.config import config
from customer.prompts.customer_prompt import customer_system_prompt
from customer.utils.logger import logger
from customer.utils.tools import get_cur_timezone_time


class IntentDetector:
    """Service for detecting user intent using LLM."""

    def __init__(self, llm=None):
        """
        Initialize intent detector.

        Args:
            llm: LLM instance to use for detection
        """
        self.llm = llm

    def detect_intent(self, query: str, conversation_id: str, history: List[Dict[str, Any]] = None) -> str:
        """
        Detect user intent from query.

        Args:
            query: User query
            conversation_id: Conversation ID for tracing
            history: Conversation history

        Returns:
            Detected intent type
        """
        if not self.llm:
            logger.warning(f"{conversation_id}: No LLM provided for intent detection, defaulting to 'other'")
            return "other"

        try:
            # Intent detection prompt
            now, weekday = get_cur_timezone_time()
            system_prompt = f'现在是北京时间 {now}，{weekday}。\n{customer_system_prompt}'

            messages = [
                {"role": "system", "content": system_prompt}
            ]

            # Add history context if available
            if history:
                messages.extend(history[-4:])  # Last 4 messages for context

            messages.append({"role": "user", "content": query})

            # Define expected schema
            json_schema = None

            # Call LLM for intent detection
            response = self.llm.call_intent_stream(
                messages,
                json_schema=json_schema,
                stream=False,
            )

            detected_intent = response
            logger.info(f"{conversation_id}: Detected intent for '{query}' -> {detected_intent}")

            return detected_intent

        except Exception as e:
            logger.error(f"{conversation_id}: Intent detection failed: {str(e)}")
            return "other"

    def judge_transfer_intent(self, query: str, conversation_id: str) -> int:
        """
        Judge if user wants to transfer to human customer service.

        Args:
            query: User query
            conversation_id: Conversation ID

        Returns:
            0 if needs transfer, 1 otherwise
        """
        if not self.llm:
            return 1

        try:
            prompt = """
# 角色: 你是一个精准的意图识别专家，判断用户是否需要寻找人工客服帮助
# 背景知识：客服上班时间为每天的 8:00-22:00, 非24h

# 技能
- 精准识别文本内容是否包含"转人工", "人工客服" "人工"关键词
- 输出0或1: 0表示需要寻找人工，1表示其他意图

# 约束
1. 严格限制返回数字 0 或 1。
2. 咨询人工客服工作时间时，输出1。
3. 当前非人工上班时间时，输出0。
4. 严格依据"转人工", "人工客服" "人工"这三个关键词。
            """

            now, weekday = get_cur_timezone_time()
            messages = [
                {"role": "system", "content": f'现在是北京时间 {now}，{weekday}。\n{prompt}'},
                {"role": "user", "content": query}
            ]

            json_schema = {
                "type": "object",
                "properties": {
                    "intent": {"type": "integer", "enum": [0, 1]}
                },
                "required": ["intent"]
            }

            response = self.llm.call_intent_stream(
                messages,
                json_schema=json_schema,
                stream=False
            )

            result = response.get("intent", 1)
            logger.info(f"{conversation_id}: Transfer intent result: {result}")

            return result if isinstance(result, int) else int(result)

        except Exception as e:
            logger.error(f"{conversation_id}: Transfer intent detection failed: {str(e)}")
            return 1
