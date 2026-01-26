#!/usr/bin/env python3
"""
Demo workflow for the customer service system.
This demonstrates how to use the refactored chat service.
"""

import asyncio
import json
import sys
import os
from typing import Generator

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from customer.config.config import config
from customer.prompts.customer_prompt import customer_system_prompt
from customer.schema.chat import ChatRequest
from customer.service.chat_service import chat_service
from customer.service.embeddings import DoubaoEmbeddings
from customer.utils.logger import logger


def demo_chat_request():
    """
    Demo function showing how to use the chat service.

    This simulates a real chat request and processes it through
    the refactored LlamaIndex-based architecture.
    """
    print("=== Chat Request Demo ===")
    print("Note: Full chat processing requires LlamaIndex and API keys.")
    print("This demo shows the structure without actual API calls.")

    # Create a sample chat request
    request = ChatRequest(
        conversation_id="demo_conversation_001",
        query="我想找一位治疗感冒的医生",
        model_name="doubao"
    )

    print(f"Sample request created: {request.query}")
    print("In production, you would call: chat_service.process_chat_request(request, backend_id)")
    return []


def demo_intent_detection():
    """Demo intent detection functionality."""
    from customer.service.intent_detector import IntentDetector
    from customer.service.llm_factory import llm_factory

    print("=== Intent Detection Demo ===")

    # Create intent detector
    llm = llm_factory.create_llm("doubao")
    intent_detector = IntentDetector(llm)

    # Test queries
    test_queries = [
        "我想找一位治疗感冒的医生",
        # "如何预约挂号？",
        # "介绍一下张医生",
        # "今天天气怎么样？"
    ]

    for query in test_queries:
        try:
            intent = intent_detector.detect_intent(query, "demo_intent", [])
            print(f"Query: '{query}' -> Intent: {intent}")
        except Exception as e:
            print(f"Query: '{query}' -> Error: {str(e)}")


def demo_entity_extraction():
    """Demo entity extraction functionality."""
    from customer.service.entity_extractor import EntityExtractor
    from customer.service.llm_factory import llm_factory

    print("\n=== Entity Extraction Demo ===")

    # Create entity extractor
    llm = llm_factory.create_llm("doubao")
    entity_extractor = EntityExtractor(llm)

    # Test queries
    test_queries = [
        "我想找北京四惠中医医院的李医生",
        "感冒了，想找医生看看",
        "杭州四惠医院有什么专家"
    ]

    for query in test_queries:
        try:
            entities = entity_extractor.extract_entities("demo_entity", query)
            print(f"Query: '{query}' -> Entities: {entities}")
        except Exception as e:
            print(f"Query: '{query}' -> Error: {str(e)}")


def demo_doctor_search():
    """Demo doctor search functionality."""
    from customer.service.doctor_search_service import DoctorSearchService
    from customer.service.llm_factory import llm_factory
    from customer.service.entity_extractor import EntityExtractor

    print("\n=== Doctor Search Demo ===")

    # Create services
    llm = llm_factory.create_llm("doubao")
    entity_extractor = EntityExtractor(llm)
    doubao_embeddings = DoubaoEmbeddings()
    doctor_service = DoctorSearchService(entity_extractor, embed_model=doubao_embeddings)

    # Test doctor search
    query = "我找许润三医生"
    history = []

    try:
        doctors = doctor_service.search_doctors(llm, query, history, "demo_doctor")
        logger.debug(doctors)
        logger.info(f"Search for '{query}' found {len(doctors)} doctors")

    except Exception as e:
        print(f"Doctor search failed: {str(e)}")


def demo_conversation_management():
    """Demo conversation management."""
    from customer.service.conversation_service import conversation_service

    print("\n=== Conversation Management Demo ===")

    conversation_id = "demo_conv"
    user_id = "001"

    # Create conversation
    success = conversation_service.create_conversation(conversation_id, user_id)
    print(f"Created conversation: {success}")

    # Add messages
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "您好！我是小惠，很高兴为您服务。"},
        {"role": "user", "content": "我想咨询感冒治疗"}
    ]
    logger.debug(f"messages: {messages}")

    success = conversation_service.update_conversation(conversation_id, messages, conversation_id)

    # Retrieve conversation
    history = conversation_service.get_conversation(conversation_id)
    print(f"Retrieved {len(history)} messages from conversation")

    for msg in history:
        print(f"  {msg['role']}: {msg['content']}")


def main():
    """Main demo function."""
    print("🗂️ LlamaIndex Customer Service Demo")
    print("=" * 50)

    # Run demos
    try:
        # Demo intent detection
        # demo_intent_detection()

        # Demo entity extraction
        # demo_entity_extraction()

        # Demo doctor search
        demo_doctor_search()

        # Demo conversation management
        # demo_conversation_management()

        print("\n=== Chat Request Demo ===")
        print("Processing full chat request...")

        # Demo full chat request (commented out to avoid actual API calls)
        # for chunk in demo_chat_request():
        #     pass  # Process chunks

        print("✅ Demo completed successfully!")
        print("\n📚 To run the full chat service:")
        print("1. Set up your API keys in environment variables")
        print("2. Configure LlamaIndex indices for doctor and QA data")
        print("3. Call chat_service.process_chat_request() with your requests")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")


if __name__ == "__main__":
    main()
