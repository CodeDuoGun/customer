system_prompt =""" 
# Role: 医疗文献病种信息抽取专家

- description: 你将接收来自医疗资料（例如PDF文档解析的文本）的内容，从中提取疾病的结构化信息，并按固定JSON格式输出，便于构建病种知识库。

## Skills
1. 从非结构化医疗文本中精准提取疾病相关核心信息。
2. 按照统一的医学术语和顺序输出结构化结果。
3. 支持缺失信息自动填充空字符串“”。
4. 对治疗部分可区分西医治疗和中医治疗。
5. 提取鉴别诊断时可识别确切指标（数值、影像特征等）。

## Background
适用于医院知识管理系统、医学数据库建设、科研资料整理等场景。

## Goals
从输入的医疗资料中提取以下字段，并以标准 JSON 格式输出：
- 疾病名称
- 疾病简介（包含下列子字段）：
  1. 定义
  2. 所属系统疾病
  3. 病因
  4. 发病机制
  5. 流行病学
  6. 临床表现
  7. 疾病分期
  8. 鉴别诊断（含确切指标）
  9. 诊断
  10. 辅助检查介绍
  11. 治疗与预防（西医治疗 / 中医治疗 / 预防）
  12. 预后
  13. 并发症

## Rules
1. 输出必须为合法 JSON，符合给定 schema。
2. 缺失信息时填充空字符串。
3. 中医治疗与西医治疗分开展示。
4. 所有字段按 schema 顺序完整输出。
5. 所有内容应保持医学专业性和准确性。

## Workflows
1. 接收医疗文本（来自 PDF）。
2. 识别疾病名称。
3. 按字段顺序逐项提取信息，缺失则填空字符串“”。
4. 生成符合 JSON Schema 的结构化输出。

## Init
请提供一段医疗资料文本，我将输出符合 schema 的结构化疾病信息。
"""

disease_json_schema = {
  "type": "object",
  "properties": {
    "disease_name": {
      "type": "string",
      "description": "疾病名称"
    },
    "disease_summary": {
      "type": "object",
      "properties": {
        "definition": { "type": "string", "description": "疾病的定义, 若没有返回空字符串" },
        "system_category": { "type": "string", "description": "所属系统疾病, 若没有返回空字符串" },
        "etiology": { "type": "string", "description": "病因, 若没有返回空字符串" },
        "pathogenesis": { "type": "string", "description": "发病机制, 若没有返回空字符串" },
        "epidemiology": { "type": "string", "description": "流行病学特征, 若没有返回空字符串" },
        "clinical_manifestations": { "type": "string", "description": "临床表现, 若没有返回空字符串" },
        "staging": { "type": "string", "description": "疾病分期, 若没有返回空字符串" },
        "differential_diagnosis": { "type": "string", "description": "鉴别诊断及确切指标, 若没有返回空字符串" },
        "diagnosis": { "type": "string", "description": "诊断标准和方法, 若没有返回空字符串" },
        "auxiliary_examination": { "type": "string", "description": "辅助检查的详细介绍, 若没有返回空字符串" },
        "treatment_and_prevention": {
          "type": "object",
          "properties": {
            "western_medicine": { "type": "string", "description": "西医治疗方案, 若没有返回空字符串" },
            "traditional_chinese_medicine": { "type": "string", "description": "中医治疗方案, 若没有返回空字符串" },
            "prevention": { "type": "string", "description": "预防措施, 若没有返回空字符串" }
          },
          "required": ["western_medicine", "traditional_chinese_medicine", "prevention"]
        },
        "prognosis": { "type": "string", "description": "预后, 若没有返回空字符串" },
        "complications": { "type": "string", "description": "并发症, 若没有返回空字符串" }
      },
      "required": [
        "definition", "system_category", "etiology", "pathogenesis", "epidemiology",
        "clinical_manifestations", "staging", "differential_diagnosis", "diagnosis",
        "auxiliary_examination", "treatment_and_prevention", "prognosis", "complications"
      ]
    }
  },
  "required": ["disease_name", "disease_summary"]
}
