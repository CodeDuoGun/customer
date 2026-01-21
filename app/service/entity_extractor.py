"""
Entity extraction service using LlamaIndex.
"""
from typing import Dict, Any, List, Optional
from customer.config.config import config
from customer.utils.logger import logger


class EntityExtractor:
    """Service for extracting medical entities from user queries."""

    def __init__(self, llm=None):
        """
        Initialize entity extractor.

        Args:
            llm: LLM instance to use for extraction
        """
        self.llm = llm

    def extract_entities(
        self,
        trace_id: str,
        user_query: str,
        rewritten_query: str = "",
        must_name: bool = False,
        history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract medical entities from user query.

        Args:
            trace_id: Trace ID for logging
            user_query: Original user query
            rewritten_query: Rewritten query if available
            must_name: Whether doctor name is required
            history: Conversation history

        Returns:
            Dictionary of extracted entities
        """
        if not self.llm:
            logger.warning(f"{trace_id}: No LLM provided for entity extraction")
            return {
                "doctor_name": "",
                "doctor_location": "",
                "condition": user_query,
                "doctor_hospital": ""
            }

        try:
            # Entity extraction prompt
            extraction_prompt = """
# 角色: 你是专业的医疗实体提取专家，从用户查询中精准提取医疗相关实体信息

# 任务: 从用户输入中提取以下实体：
- doctor_name: 医生姓名，支持多个医生名，用逗号分隔
- doctor_location: 医生所在地区或医院位置
- condition: 疾病症状描述
- doctor_hospital: 医生所在医院名称

# 提取规则:
1. doctor_name: 直接提到的医生姓名，如"李医生"、"张教授"等
2. doctor_location: 地区信息，如"北京"、"上海"等，或医院位置描述
3. condition: 疾病名称、症状描述，如"感冒"、"头痛"、"肿瘤"等
4. doctor_hospital: 医院名称，如"四惠中医医院"、"北京医院"等

# 输出格式: 必须是有效的JSON格式
{
    "doctor_name": "提取的医生姓名",
    "doctor_location": "提取的地区信息",
    "condition": "提取的疾病症状",
    "doctor_hospital": "提取的医院名称"
}

# 注意事项:
- 如果某实体未提及，对应的值设为空字符串""
- 保持提取的准确性，不要过度推测
- 对于医生姓名，支持模糊匹配和别名识别
            """

            query_to_process = rewritten_query or user_query

            messages = [
                {"role": "system", "content": extraction_prompt}
            ]

            # Add recent history for context
            if history:
                recent_history = history[-4:]  # Last 4 messages
                messages.extend(recent_history)

            messages.append({"role": "user", "content": f"用户查询：{query_to_process}"})

            # Define JSON schema for structured output
            json_schema = {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string"},
                    "doctor_location": {"type": "string"},
                    "condition": {"type": "string"},
                    "doctor_hospital": {"type": "string"}
                },
                "required": ["doctor_name", "doctor_location", "condition", "doctor_hospital"]
            }

            # Call LLM for entity extraction
            response = self.llm.call_intent_stream(
                messages,
                json_schema=json_schema,
                stream=False,
                chat_model="coze_deepseek"
            )

            logger.info(f"{trace_id}: Extracted entities: {response}")

            # Ensure all required fields are present
            entities = {
                "doctor_name": response.get("doctor_name", ""),
                "doctor_location": response.get("doctor_location", ""),
                "condition": response.get("condition", user_query),
                "doctor_hospital": response.get("doctor_hospital", "")
            }

            return entities

        except Exception as e:
            logger.error(f"{trace_id}: Entity extraction failed: {str(e)}")
            return {
                "doctor_name": "",
                "doctor_location": "",
                "condition": user_query,
                "doctor_hospital": ""
            }
