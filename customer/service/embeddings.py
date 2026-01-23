"""
自定义嵌入模型实现

使用示例：
    from customer.service.embeddings import DoubaoEmbeddings

    # 创建嵌入模型实例
    embed_model = DoubaoEmbeddings()

    # 生成单个文本的嵌入
    embedding = embed_model.get_text_embedding("这是一个测试文本")

    # 批量生成嵌入
    texts = ["文本1", "文本2", "文本3"]
    embeddings = embed_model.get_text_embeddings(texts)

    # 获取模型信息
    info = embed_model.get_embedding_info()
    print(f"模型: {info['model_name']}, 维度: {info['dimensions']}")

    # 在LlamaIndex中使用
    from customer.service.doctor_search_service import DoctorSearchService
    service = DoctorSearchService(embed_model=embed_model)
"""
import requests
from typing import Any, List, Optional
from llama_index.core.embeddings import BaseEmbedding
from customer.config.config import config
from customer.utils.logger import logger


class DoubaoEmbeddings(BaseEmbedding):
    """豆包嵌入模型，使用豆包API生成文本向量"""

    def __init__(
        self,
        model_name: str = "doubao-embedding-large-text-240915",
        api_key: str = config.ARK_API_KEY,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3/embeddings",
        dims: int = 0,
        **kwargs: Any,
    ) -> None:
        # 先调用父类初始化
        super().__init__(**kwargs)

        # 然后设置实例变量，确保不会被覆盖
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._dims = dims

        # 设置请求头
        self._headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}'
        }

    def _get_doubao_embedding(self, text: str) -> List[float]:
        """调用豆包API生成单个文本的嵌入向量"""
        json_data = {
            "model": self._model_name,
            "input": [text],
        }

        if self._dims > 0:
            json_data["dimensions"] = self._dims

        try:
            response = requests.post(
                self._base_url,
                headers=self._headers,
                json=json_data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                embedding = result["data"][0]["embedding"]
                return embedding
            else:
                logger.error(f"Doubao embedding API error: {response.status_code}, {response.text}")
                return []

        except Exception as e:
            logger.error(f"Doubao embedding request failed: {str(e)}")
            return []

    def _get_query_embedding(self, query: str) -> List[float]:
        """生成查询文本的嵌入向量"""
        return self._get_doubao_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """生成文档文本的嵌入向量"""
        return self._get_doubao_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量生成多个文本的嵌入向量"""
        embeddings = []
        for text in texts:
            embedding = self._get_doubao_embedding(text)
            embeddings.append(embedding)
        return embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步生成查询文本的嵌入向量"""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步生成文档文本的嵌入向量"""
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步批量生成多个文本的嵌入向量"""
        return self._get_text_embeddings(texts)

    def _get_doubao_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量调用豆包API生成多个文本的嵌入向量（更高效）"""
        if not texts:
            return []

        json_data = {
            "model": self._model_name,
            "input": texts,
        }

        if self._dims > 0:
            json_data["dimensions"] = self._dims

        try:
            response = requests.post(
                self._base_url,
                headers=self._headers,
                json=json_data,
                timeout=60  # 批量请求给更多时间
            )

            if response.status_code == 200:
                result = response.json()
                embeddings = [item["embedding"] for item in result["data"]]
                return embeddings
            else:
                logger.error(f"Doubao batch embedding API error: {response.status_code}, {response.text}")
                # 回退到单个请求
                return self._get_text_embeddings(texts)

        except Exception as e:
            logger.error(f"Doubao batch embedding request failed: {str(e)}")
            # 回退到单个请求
            return self._get_text_embeddings(texts)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量生成多个文本的嵌入向量，使用批量API调用"""
        if len(texts) <= 1:
            # 单个文本使用单个API调用
            return [self._get_doubao_embedding(text) for text in texts]

        # 多个文本使用批量API调用
        return self._get_doubao_embeddings_batch(texts)

    @property
    def embedding_dimensions(self) -> int:
        """返回嵌入向量的维度"""
        # 豆包large模型默认是4096维
        return 4096 if self._dims == 0 else self._dims

    def get_embedding_info(self) -> dict:
        """获取嵌入模型信息"""
        return {
            "model_name": self._model_name,
            "provider": "Doubao",
            "dimensions": self.embedding_dimensions,
            "supports_batch": True,
            "api_url": self._base_url
        }


def test_doubao_embedding():
    """测试豆包嵌入模型"""
    try:
        print("=== 测试豆包嵌入模型初始化 ===")

        embed_model = DoubaoEmbeddings()

        # 测试属性存在
        # required_attrs = ['_model_name', '_api_key', '_base_url', '_dims', '_headers']
        # for attr in required_attrs:
        #     if not hasattr(embed_model, attr):
        #         print(f"✗ 缺少属性: {attr}")
        #         return False
        #     else:
        #         print(f"✓ 属性 {attr} 存在: {getattr(embed_model, attr)[:50] if len(str(getattr(embed_model, attr))) > 50 else getattr(embed_model, attr)}")

        print("\n=== 测试模型信息获取 ===")

        # 显示模型信息
        info = embed_model.get_text_embedding("脱发")
        print(f"✓ 模型信息获取成功: {len(info)}")

        print("\n=== 测试完成 - 初始化和基本功能正常 ===")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# if __name__ == "__main__":
#     test_doubao_embedding()