## 数据资产

`asserts` 目录下包含用于构建数据库所需的基本文献资料：
- `asserts/markdowns`：存放原始 PDF 文档经 OCR 解析后转换而成的 Markdown 文档
  - **功能**：构建领域知识库，做基于 SQL 的检索或基于 LLM 的问答、生成
- `asserts/notes`：LLM 生成的标签
  - **schema**:
```json
{    
    "authors": [作者列表, List],
    "year": [提出年份, Int],
    "research_directions": [研究方向, List[String]],
    "specific_task": [细分任务, List[String]],
    "summary": [一句话概括文章内容, String],
    "techniques": [主要技术, List[String]],
    "method": [方法简介, String]
    "title": [文章标题, String]
}
```
- `asserts/tech-notes`：LLM 生成的论文阅读笔记
  - **功能**：方便快速阅读和学习论文
- `asserts/nick-name`：LLM 自动标注文章所提方法的别名

> **Note**：以上数据的获取依赖 LLM 支持，使用模型为 Qwen3-Next-80B-A3B-Instruct