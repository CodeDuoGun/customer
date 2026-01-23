#!/usr/bin/env python3
"""
演示如何使用自定义的豆包嵌入模型
"""

from customer.service import DoubaoEmbeddings, DoctorSearchService


def demo_doubao_embedding():
    """演示豆包嵌入模型的使用"""
    print("=== 豆包嵌入模型演示 ===\n")

    # 1. 创建嵌入模型实例
    print("1. 创建豆包嵌入模型...")
    embed_model = DoubaoEmbeddings()
    print("✓ 嵌入模型创建成功")

    # 2. 获取模型信息
    print("\n2. 模型信息:")
    info = embed_model.get_embedding_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    # 3. 测试单个文本嵌入
    print("\n3. 测试单个文本嵌入:")
    test_text = "心血管疾病专家"
    embedding = embed_model.get_text_embedding(test_text)
    print(f"   文本: {test_text}")
    print(f"   嵌入维度: {len(embedding) if embedding else 0}")
    if embedding:
        print(f"   向量前5个值: {embedding[:5]}")
        print("✓ 单个文本嵌入成功")

    # 4. 测试批量文本嵌入
    print("\n4. 测试批量文本嵌入:")
    test_texts = [
        "心脏病专家",
        "糖尿病治疗",
        "中医内科医生",
        "儿科专家"
    ]
    embeddings = embed_model.get_text_embeddings(test_texts)
    print(f"   输入文本数量: {len(test_texts)}")
    print(f"   输出嵌入数量: {len(embeddings)}")
    print("✓ 批量文本嵌入成功")

    # 5. 计算向量相似度
    print("\n5. 计算向量相似度:")
    if len(embeddings) >= 2:
        from numpy import dot
        from numpy.linalg import norm

        vec1 = embeddings[0]  # "心脏病专家"
        vec2 = embeddings[1]  # "糖尿病治疗"

        cosine_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
        print(".4f"
    # 6. 在医生搜索服务中使用
    print("\n6. 在医生搜索服务中使用:")
    try:
        service = DoctorSearchService(embed_model=embed_model)
        print("✓ 医生搜索服务创建成功，使用自定义嵌入模型")

        # 测试自定义嵌入检索
        results = service.retrieve_with_text_and_custom_embedding(
            query="心脏病专家",
            similarity_top_k=3
        )
        print(f"   检索到 {len(results)} 个结果")

    except Exception as e:
        print(f"✗ 服务创建失败: {str(e)}")

    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    demo_doubao_embedding()
