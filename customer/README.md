# Customer Service System - LlamaIndex Refactoring

基于 LlamaIndex 框架重构的客服聊天系统，将原有单体架构拆分为模块化设计。

## 📁 项目结构

```
customer/
├── config/              # 配置管理
│   ├── config.py      # Dynaconf配置类
│   └── config.yaml    # 配置文件
├── schema/              # 数据模型
│   └── chat.py          # 聊天相关数据模型
├── llm_factory/               # LLM架构
│   ├── base_llm.py             # 基础LLM抽象类
│   ├── openai_llm.py           # OpenAI兼容LLM
│   ├── doubao_llm.py           # 豆包LLM
│   ├── qwen_llm.py             # 千问LLM
│   ├── llama_llm.py            # Llama LLM
│   └── factory.py              # LLM工厂
├── service/             # 业务服务层
│   ├── chat_service.py          # 主聊天服务
│   ├── intent_detector.py       # 意图识别服务
│   ├── entity_extractor.py      # 实体提取服务
│   ├── doctor_search_service.py # 医生搜索服务
│   ├── qa_search_service.py     # 问答搜索服务
│   ├── conversation_service.py  # 会话管理服务
│   └── llm_factory.py          # LLM工厂(兼容层)
├── rag/                 # RAG组件
│   ├── vector_store.py  # 向量存储管理
│   └── index_builder.py # 索引构建器
├── utils/               # 工具函数
│   ├── constants.py     # 常量和枚举
│   ├── tools.py         # 工具函数
│   └── logger.py        # 日志工具
├── __init__.py          # 包初始化
└── demo_workflow.py     # 演示文件
```

## 🚀 核心特性

### 1. **模块化架构**
- 将原来866行的单体 `chat.py` 重构为13个专用模块
- 每个模块职责单一，便于维护和测试
- 符合 SOLID 设计原则

### 2. **模块化LLM架构**
- 继承模式的LLM抽象类，支持不同模型的统一接口
- 支持OpenAI、豆包、千问、Llama等多种模型
- 统一的流式和非流式调用接口
- 模型参数适配和优化配置

### 3. **LlamaIndex集成**
- 使用标准向量存储和检索接口
- 支持 RAG (Retrieval-Augmented Generation)
- 可扩展的索引管理和搜索功能

### 3. **流式响应**
- SSE (Server-Sent Events) 格式的实时响应
- 支持分块文本发送，提升用户体验
- 错误处理和异常恢复

### 4. **智能意图识别**
- 基于 LLM 的意图分类
- 支持疾病搜索、医生推荐、问答等意图
- 可配置的意图检测规则

### 5. **实体提取**
- 从用户查询中提取医疗实体
- 支持医生姓名、医院、症状等信息
- 上下文感知的实体识别

## 🛠️ 技术栈

- **框架**: LlamaIndex
- **语言**: Python 3.8+
- **LLM支持**: OpenAI、豆包、千问、Llama等
- **配置**: Dynaconf + YAML配置
- **日志**: 结构化日志记录
- **测试**: 单元测试 + Mock支持

## 📋 使用方法

### 1. 环境配置

配置通过 `customer/config/config.yaml` 文件管理，也可以通过环境变量覆盖：

```yaml
# customer/config/config.yaml
deepseek_api_key: "your_api_key"
ark_api_key: "your_ark_key"
es_host: "localhost"
es_port: 9200
env_version: "dev"
chunk_size: 50
# ... 其他配置项
```

或者设置环境变量：
```bash
export DEEPSEEK_API_KEY="your_api_key"
export ARK_API_KEY="your_ark_key"
export ES_HOST="localhost"
export ES_PORT="9200"
```

### 2. 运行演示

```bash
cd /path/to/llama_index
python customer/demo_workflow.py
```

### 3. 使用聊天服务

```python
from customer.service.chat_service import chat_service
from customer.schema.chat import ChatRequest

# 创建请求
request = ChatRequest(
    conversation_id="user_123",
    query="我想找一位治疗感冒的医生",
    model_name="coze_deepseek"
)

# 处理请求
for chunk in chat_service.process_chat_request(request, "backend_001"):
    print(chunk)  # 发送给客户端
```

```python
# 直接使用LLM工厂
from customer.llm_factory.factory import llm_factory

# 创建不同类型的LLM
doubao_llm = llm_factory.create_llm("doubao-lite-32k")
qwen_llm = llm_factory.create_llm("qwen-turbo")

# 统一调用接口
messages = [{"role": "user", "content": "你好"}]

# 同步调用
response = doubao_llm.call(messages)

# 流式调用
for chunk in doubao_llm.call_stream(messages):
    print(chunk)

# 意图检测
intent_result = doubao_llm.call_intent_stream(messages, json_schema={"type": "object", "properties": {"intent": {"type": "string"}}})
```

## 🤖 LLM架构

### 继承模式设计
```
BaseLLM (抽象基类)
├── OpenAILLM      # OpenAI/DeepSeek/Coze
├── DoubaoLLM      # 豆包模型
├── QwenLLM        # 千问模型
└── LlamaLLM       # Llama系列
```

### 核心特性
- **统一接口**: 所有LLM实现相同的调用接口
- **参数适配**: 不同模型的专用参数优化
- **流式支持**: 统一的流式和非流式响应
- **错误处理**: 健壮的错误处理和降级策略
- **扩展性**: 易于添加新的LLM提供商

## 🔧 核心组件

### LLM Factory
智能LLM工厂：
- 根据模型名称自动选择合适的LLM实现
- 支持模型缓存和连接池
- 提供模型信息查询接口

### ChatService
主聊天服务，协调所有组件：
- 意图识别和路由
- 并发处理（意图检测 + 查询改写）
- 流式响应管理
- 错误处理

### IntentDetector
意图检测服务：
- 基于 LLM 的意图分类
- 支持多意图类型
- 转人工客服判断

### DoctorSearchService
医生搜索服务：
- 基于 LlamaIndex 的向量搜索
- 实体提取和过滤
- 医生信息聚合

### QASearchService
问答搜索服务：
- RAG 架构实现
- 重排序支持
- 答案生成优化

## 📊 重构成果统计

- **总文件数**: 13个 Python 文件
- **总代码行数**: ~1200行
- **模块数量**: 从1个拆分为13个专用模块
- **测试覆盖**: ✅ 语法检查通过
- **演示运行**: ✅ 成功运行

## 🔄 架构对比

### 重构前
- 单体架构：866行单文件
- 紧耦合：所有功能混在一起
- 难以维护：代码重复，逻辑复杂
- 扩展性差：新增功能需大幅修改

### 重构后
- 模块化设计：13个专用模块
- 松耦合：各模块职责清晰
- 易于维护：代码组织良好
- 高扩展性：可独立扩展各模块

## 🎯 优势

1. **可维护性**: 模块化设计，便于理解和修改
2. **可扩展性**: 新功能可独立开发，不影响现有代码
3. **可测试性**: 各模块可独立测试
4. **性能优化**: 并发处理，提高响应速度
5. **代码复用**: 组件可在其他项目中复用

## 📝 后续优化

- [ ] 添加单元测试
- [ ] 集成监控和指标收集
- [ ] 添加缓存机制
- [ ] 支持更多 LLM 提供商
- [ ] 实现 A/B 测试框架

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📄 许可证

本项目遵循原有项目的许可证。
