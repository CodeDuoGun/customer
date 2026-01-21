"""
QA search and retrieval service using LlamaIndex RAG.
"""
from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures
import json
from customer.config.config import config
from customer.utils.logger import logger
from customer.utils.constants import Speeches, MessageEventStatus


class QASearchService:
    """Service for QA search and retrieval using LlamaIndex."""

    def __init__(self, qa_index=None, reranker=None, tokenizer=None):
        """
        Initialize QA search service.

        Args:
            qa_index: LlamaIndex index for QA data
            reranker: Reranker model for result ranking
            tokenizer: Tokenizer for reranking
        """
        self.qa_index = qa_index
        self.reranker = reranker
        self.tokenizer = tokenizer

    def search_qa(self, index_name: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search QA by question.

        Args:
            index_name: Index name
            query: Search query
            top_k: Number of results to return

        Returns:
            List of QA results
        """
        try:
            if not self.qa_index:
                logger.warning(f"No QA index available for search")
                return []

            retriever = self.qa_index.as_retriever(similarity_top_k=top_k)
            results = retriever.retrieve(query)

            qa_results = []
            for result in results:
                qa_item = {
                    "question": result.node.metadata.get("question", ""),
                    "answer": result.node.metadata.get("answer", ""),
                    "score": result.score,
                    "_source": {
                        "question": result.node.metadata.get("question", ""),
                        "answer": result.node.metadata.get("answer", "")
                    }
                }
                qa_results.append(qa_item)

            logger.info(f"Found {len(qa_results)} QA results for query: {query}")
            return qa_results

        except Exception as e:
            logger.error(f"QA search failed: {str(e)}")
            return []

    def search_qa_by_answer(self, index_name: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search QA by answer content.

        Args:
            index_name: Index name
            query: Search query
            top_k: Number of results to return

        Returns:
            List of QA results
        """
        try:
            if not self.qa_index:
                logger.warning(f"No QA index available for answer search")
                return []

            # Search by answer content (can use same index with different retrieval strategy)
            retriever = self.qa_index.as_retriever(similarity_top_k=top_k)
            results = retriever.retrieve(f"答案内容：{query}")

            qa_results = []
            for result in results:
                qa_item = {
                    "question": result.node.metadata.get("question", ""),
                    "answer": result.node.metadata.get("answer", ""),
                    "score": result.score,
                    "_source": {
                        "question": result.node.metadata.get("question", ""),
                        "answer": result.node.metadata.get("answer", "")
                    }
                }
                qa_results.append(qa_item)

            logger.info(f"Found {len(qa_results)} QA by answer results for query: {query}")
            return qa_results

        except Exception as e:
            logger.error(f"QA answer search failed: {str(e)}")
            return []

    def perform_parallel_searches(self, rewritten_query: str, original_query: str) -> Tuple[List, List]:
        """
        Perform parallel QA searches.

        Args:
            rewritten_query: Rewritten query
            original_query: Original query

        Returns:
            Tuple of (qa_results, qa_answer_results)
        """
        try:
            index_name = f"{config.env_version}_qa"
            qa_size = config.qa_size
            qa_answer_size = config.qa_answer_size

            # Define search tasks
            tasks = [
                (self.search_qa, index_name, rewritten_query, qa_size),
                (self.search_qa_by_answer, index_name, original_query, qa_answer_size)
            ]

            # Execute tasks in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(func, *args) for func, *args in tasks]
                results = []

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Search task failed: {str(e)}")
                        results.append([])

            # Ensure we have two results
            if len(results) == 2:
                qa_results, qa_answer_results = results
            else:
                qa_results = results[0] if results else []
                qa_answer_results = []

            logger.info(f"Parallel search completed: {len(qa_results)} questions, {len(qa_answer_results)} answers")
            return qa_results, qa_answer_results

        except Exception as e:
            logger.error(f"Parallel search failed: {str(e)}")
            return [], []

    def rerank_results(
        self,
        trace_id: str,
        query: str,
        qa_results: List[Dict],
        qa_answer_results: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Rerank QA results using reranker model.

        Args:
            trace_id: Trace ID
            query: Search query
            qa_results: Question-based results
            qa_answer_results: Answer-based results

        Returns:
            Reranked results
        """
        try:
            if not self.reranker or not self.tokenizer:
                logger.warning(f"{trace_id}: No reranker available, returning combined results")
                combined = qa_results + qa_answer_results
                return combined[:10]  # Return top 10

            # Prepare data for reranking
            rerank_data = []
            for hit in qa_results:
                rerank_data.append([query, hit["_source"]["question"], hit["_source"]["answer"]])

            for hit in qa_answer_results:
                rerank_data.append([query, hit["_source"]["question"], hit["_source"]["answer"]])

            # Perform reranking in batches
            batch_size = 4
            all_scores = []

            for i in range(0, len(rerank_data), batch_size):
                batch = rerank_data[i:i + batch_size]
                batch_queries = [item[0] for item in batch]
                batch_questions = [item[1] for item in batch]

                # Tokenize batch
                inputs = self.tokenizer(
                    [batch_queries, batch_questions],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=256
                )

                # Move to device
                inputs = {k: v.to(self.reranker.device) for k, v in inputs.items()}

                # Get reranker scores
                with self.reranker.no_grad():
                    batch_scores = self.reranker(**inputs, return_dict=True).logits.view(-1).float()

                all_scores.extend(batch_scores.cpu().tolist())

            # Combine results with scores
            reranked_results = []
            for (query_text, question, answer), score in zip(rerank_data, all_scores):
                reranked_results.append({
                    "question": question,
                    "answer": answer,
                    "score": score
                })

            # Sort by score
            reranked_results.sort(key=lambda x: x["score"], reverse=True)

            # Remove duplicates by answer
            seen_answers = set()
            unique_results = []
            for result in reranked_results:
                if result["answer"] not in seen_answers:
                    unique_results.append(result)
                    seen_answers.add(result["answer"])

            logger.info(f"{trace_id}: Reranked {len(unique_results)} unique results")
            return unique_results

        except Exception as e:
            logger.error(f"{trace_id}: Reranking failed: {str(e)}")
            # Fallback to simple combination
            combined = qa_results + qa_answer_results
            return combined[:10]

    def find_most_similar_question(
        self,
        llm,
        trace_id: str,
        rewritten_query: str,
        qa_results: List[Dict],
        questions: List[str],
        history: List[Dict[str, Any]]
    ) -> Tuple[str, int]:
        """
        Find most similar question using LLM.

        Args:
            llm: LLM instance
            trace_id: Trace ID
            rewritten_query: Rewritten query
            qa_results: QA results
            questions: List of questions
            history: Conversation history

        Returns:
            Tuple of (best_answer, question_index)
        """
        try:
            if not qa_results:
                return "", -1

            answers = [hit["_source"]["answer"] for hit in qa_results]

            # Build prompt for question matching
            qa_prompt = """你将获得一个问题，不要回答问题，根据对话历史找出最符合用户想要咨询的问题。
并输出对应的序号。

## 要求：
"""

            for q_idx, q in enumerate(questions):
                qa_prompt += f"如果问题:{q} 最符合用户咨询的问题,就输出{q_idx}\n"

            qa_prompt += f"""
\n用户咨询的问题：{rewritten_query}

## 判断标准强化说明：
1.识别医院/平台：
用户问题中如果提及"互联网医院"、"线上医院"或未指明但语境明显为线上服务，则归类为"互联网医院"。
准确区分不同院区（见下方医院别名表），避免混淆四惠、郁证中心等其他机构的问题。
2.理解用户意图：
关注关键词：退挂号费、退款、退药、物流、加号、预约、复诊、客服、看诊、视频问题等。

## 限制：
1. 只输出数字，不要输出其他内容
"""

            messages = [
                {"role": "system", "content": qa_prompt}
            ]

            if history:
                messages.extend(history[-4:])  # Recent history

            # Call LLM to find best match
            final_qa_idx = llm.call_intent_stream(
                messages,
                json_schema=None,
                stream=False,
                chat_model=config.doubao_text_model,
                temperature=0.1
            )

            try:
                final_qa_idx = int(final_qa_idx)
                if 0 <= final_qa_idx < len(answers):
                    best_answer = answers[final_qa_idx]
                    logger.info(f"{trace_id}: Most similar question index: {final_qa_idx}")
                    return best_answer, final_qa_idx
                else:
                    logger.warning(f"{trace_id}: Invalid question index: {final_qa_idx}")
                    return answers[0], 0
            except (ValueError, TypeError):
                logger.warning(f"{trace_id}: Failed to parse question index: {final_qa_idx}")
                return answers[0], 0

        except Exception as e:
            logger.error(f"{trace_id}: Most similar question search failed: {str(e)}")
            return "", -1

    def generate_answer(
        self,
        llm,
        trace_id: str,
        query: str,
        final_refer_answers: str,
        history: List[Dict[str, Any]],
        sse_data: Dict[str, Any],
        conversation_id: str
    ):
        """
        Generate final answer using retrieved context.

        Args:
            llm: LLM instance
            trace_id: Trace ID
            query: User query
            final_refer_answers: Retrieved reference answers
            history: Conversation history
            sse_data: SSE data structure
            conversation_id: Conversation ID

        Yields:
            Response chunks
        """
        try:
            # Answer generation prompt
            answer_prompt = f"""根据对话历史，针对用户问题 {query} 从答案{final_refer_answers}中精确提取最相关的内容，
简洁清晰有礼貌的给出答案。禁止输出其他无关内容

技能：
    1. 精准提取答案中的关键信息，紧扣用户问题进行回应，保证回答内容与问题匹配，不能答非所问。
    2. 若答案中没有能准确回应用户问题的信息（即使"部分相关"但不是用户"真正关注点"），参考回复如下：
        您好！这个问题小惠还在学习中，您先联系人工看看，或者拨打客服热线400−689−6699吧。我会加速学习，来更好的为您服务！
    3. 精准区分用户问题是针对以下哪个医疗机构，并给出相应的回答。
        机构名称：
        北京四惠中医医院
        北京四惠西区医院
        北京四惠南区门诊部
        上海圣保堂中医门诊
        南宁桂派中医门诊
        杭州四惠医院
        郑州四惠中医门诊部：郑州医馆、郑州门诊/医院
        北京四惠中西医结合肿瘤会诊中心
        郁证临床学科示范基地：郁证中心、郁证示范基地、郁证基地
        互联网医院：互联网医院、线上医院

    4. 如果询问"退挂号费" "退号/取消预约/取消挂号"时，从以下内容中提取相关信息：
        您好～为了不耽误您的退号办理，小惠将各家医院的退号退款方式整理如下啦👇
            1. 线下医院（如北京四惠中医医院、西区、南区、杭州、上海、南宁、郑州）：
            - 请在就诊当天，携带身份证或医保卡前往挂号窗口或自助机办理退号
            - 拨打各医院客服电话协助您退款退号: 1.北京四惠中医医院：010-67289999
2.北京四惠西区医院：010-88849999
3.北京四惠南区医院：010-67289966
4.中西医结合肿瘤会诊中心：010-67289999
5.杭州四惠医院：0571-8868 5312
6.南宁桂派中医门诊：0771-5556788
7.上海圣保堂中医门诊： 021-66261616
8.郁证中心：010-67289999
9.郑州四惠中医门诊：0371-65012822

            2. 互联网医院：
            - 可联系您的医助协助退款，或拨打客服热线 400-689-6699
            - 费用一般 3-7 个工作日到账，如有问题可随时联系小惠哈～

        您好!线下医院、四惠中医医院、四惠西区医院、四惠南区中医门诊部、杭州四惠医院、圣保堂中医门诊部、南宁桂派中医门诊部、中西医结合肿瘤会诊中心、郑州门诊退号方法及流程如下：
1.携带身份证或医保卡
2.前往挂号窗口或自助机办理退号
3.退号需在就诊当天完成
4.联系人工客服或拨打各医院的客服电话协助您退号退款，给您带来的不便，敬请谅解！
        四惠医疗互联网医院、郁症示范基地的退号方法如下：
1、联系服务您的医助，医助会及时为您办理
2、联系人工客服或者拨打400-689-6699客服电话

    限制：
        1.必须始终以"小惠"的角色，秉持"专业、客气、亲切"态度回答。
        2.严禁给出答案中没有的机构或者组织，如果不知道，就说"请联系人工客服，或者拨打客服热线400−689−6699"
        3.严格区分答案中"线下医院" 和 "互联网医院"两个主体。
        4.严禁答案中出现多个换行符,输出结果时只能使用"\n"表示换行，禁止连续出现"\n\n"。

"""

            messages = [
                {"role": "system", "content": answer_prompt}
            ]

            if history:
                messages.extend(history[-4:])

            # Stream the answer
            max_tokens = 800
            responses = llm.chat_coze_stream(
                sse_data, messages, history, conversation_id, trace_id, query,
                need_update_history=True, chat_model="coze_deepseek",
                need_stream_content=True, max_tokens=max_tokens
            )

            for chunk in responses:
                yield chunk

            logger.info(f"{trace_id}: Answer generation completed")

        except Exception as e:
            logger.error(f"{trace_id}: Answer generation failed: {str(e)}")
            # Return error message
            sse_data["content"] = "抱歉，小惠暂时无法回答这个问题。请稍后再试。"
            sse_data["event"] = MessageEventStatus.COMPLETED
            yield f"event: conversation\n\ndata: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
            yield "event: [DONE]\ndata: [DONE]\n\n"
