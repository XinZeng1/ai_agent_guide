# 核心要点
1. LCEL（管道式写法）的核心思想是什么？和平时写 Python 函数调用有什么区别？
2. LangChain 中的 Retriever 是什么？similarity 检索和 MMR 检索有什么区别？什么场景用 MMR？
3. 多轮对话 RAG 中"查询改写"这步为什么必不可少？如果不做改写，用户说"那个呢"会发生什么？
4. LangChain 帮你省了什么？没帮你省什么？
5. 回顾 Week1 整体：从手写 RAG 到用框架重构，这个学习路径的意义是什么？


# 练习

## Part1
学会使用 LangChain 的 RecursiveCharacterTextSplitter

### 理解
LangChain 封装了文本切片，`RecursiveCharacterTextSplitter` 会依次尝试按 `"\n\n" → "\n" → "。" → "，" → " " → ""` 来切割，优先保持语义完整性。中文场景需要在 separators 里加入中文标点。

和 Day3 手写的对比：框架省去了边界处理和 overlap 逻辑的手写，但 `chunk_size` 和 `overlap` 仍然需要根据业务调整——这是框架帮不了的。

### 注意
新版 LangChain（v0.2+）把 text_splitter 拆到了独立包，导入路径从 `langchain.text_splitter` 改为 `langchain_text_splitters`。


## Part2
用 LangChain + ChromaDB 快速构建向量索引

### 理解
`Chroma.from_documents()` 一行代码完成 文档→Embedding→存储。`as_retriever()` 把向量库转成 Retriever 对象，方便接入 LCEL Chain。

### MMR 检索
普通 similarity 检索按相似度排序，结果可能主题过于集中（如 Top-5 全是鸡蛋相关）。MMR（Maximal Marginal Relevance）在选中一个结果后，会降低与已选结果相似的候选分数，保证结果的多样性。

- `lambda_mult` 控制平衡：越小越多样，越大越接近普通检索（默认 0.5）
- `fetch_k`：先召回多少候选再用 MMR 挑选
- 适用场景：希望给 LLM 提供多角度参考信息时用 MMR，需要精准匹配单一主题时用 similarity

### 体会
知识库 chunk 天然分散在不同主题时，MMR 和普通检索的差异不明显。只有当同主题 chunk 密集时（如 6 条都是鸡蛋相关），MMR 的去重效果才显著。实际项目中知识库通常很大，同主题冗余是常态，MMR 的价值会更明显。


## Part3
理解 LCEL（LangChain Expression Language）的管道式写法

### 理解
LCEL 是 LangChain 的核心语法，用管道符 `|` 把组件串联，类似 Unix 管道。

```python
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | parser
)
```

关键组件：
- `RunnablePassthrough()`：把输入原样传递（question 不需要加工）
- `RunnableLambda(fn)`：包装自定义函数（把 Document 列表格式化成字符串）
- 字典结构 `{}`：让一个输入同时走两条路径（question 直传 + context 经过检索和格式化）

### 工程价值
- **组件可替换**：换 LLM（DeepSeek → GPT-4）只需改一个变量
- **自动支持**：`.stream()`（流式）、`.batch()`（批量）、`.ainvoke()`（异步）
- **代码简洁**：5 行 vs Day3 手写的 30 行，但底层逻辑完全相同


## Part4
实现多轮对话的 RAG（带上下文记忆）

### 理解
多轮对话 RAG 的核心问题：向量数据库不懂对话上下文。用户说"那如果没有冰箱呢"，向量库搜"那如果没有冰箱呢"，根本不知道"那"指什么，检索结果会完全错误。

解决方案：**查询改写（Contextualization）**，在检索前用 LLM 把带上下文的问题改写成独立查询：
- "那如果没有冰箱呢" → "鸡蛋在常温下能保存多久"
- "怎么判断它有没有坏" → "如何判断鸡蛋是否变质"

### 完整流程（两次 LLM 调用）

```
用户问："那如果没有冰箱呢"
  ↓
第1次 LLM：改写查询 → "鸡蛋在常温下能保存多久"  （为了检索）
  ↓
向量检索：用改写后的查询检索 → 命中正确文档
  ↓
第2次 LLM：基于 对话历史 + 检索结果 + 当前问题 → 生成回答  （和 bots 方式类似）
```

### 优化点
不是每次都需要改写。如果用户的问题本身就是独立的（如"苹果怎么保存"），可以跳过改写节省一次 LLM 调用。可以让改写 LLM 先判断是否需要改写，不需要则原样返回。

### 与 Day1 的关联
Day1 学到 LLM 是无状态的，多轮对话靠每次发送完整历史实现。Day5 的多轮对话 RAG 多了一个环节：历史不仅要给 LLM（生成用），还要用来改写查询（检索用）。对话越长，改写和生成两次 LLM 调用的 input_tokens 都会增长 → 需要 Memory 策略（滑动窗口/摘要压缩）。


## Part5
理解框架的价值和边界

### LangChain 帮了什么
- 组件封装：切片、Embedding、向量库操作一行搞定
- 接口统一：换 LLM/向量库只需改一个变量，不动 Chain 逻辑
- 生态集成：LangSmith 可视化追踪、LangServe 快速部署
- 开发提效：LCEL 管道式写法简洁，自动支持流式/批量/异步

### LangChain 帮不了什么
- Prompt 设计：框架提供了模板结构，但内容还是你自己写
- 切片策略选择：chunk_size、overlap、分隔符需要根据业务调
- 效果调优和评估：Hit Rate、MRR 这些需要自己建评测集跑
- 业务逻辑：什么时候该说"不知道"、什么时候允许 LLM 补充，这些是业务决策

### 什么时候用框架 vs 手写
- **框架**：快速原型、对接多种组件、团队协作
- **手写**：面试讲原理、极致性能、框架不支持的定制需求
- **最佳实践**：先手写理解原理 → 再用框架提效 → 需要时回到底层优化


# 核心要点回答

1. **LCEL 的核心思想**：用管道符 `|` 串联组件，前一个的输出自动传给下一个。和函数调用的区别：组件可替换、自动支持流式/批量/异步、代码更声明式。底层逻辑和手写完全一样。

2. **Retriever 是 LangChain 对检索器的统一抽象**，所有实现了 `invoke(query) → List[Document]` 的都是 Retriever。similarity 检索按相似度排序，结果可能主题集中；MMR 在相关性和多样性之间平衡，适合给 LLM 提供多角度参考。

3. **查询改写是因为向量数据库不懂对话上下文**。"那个呢"直接检索，向量库不知道"那个"指什么，会返回完全不相关的文档。改写成独立查询后才能正确检索。

4. **LangChain 帮了组件封装和接口统一，没帮 Prompt 设计、切片策略、效果调优**。框架是工具不是银弹。

5. **Week1 学习路径的意义**：先手写理解每一步原理（Day1-4），再用框架重构提效（Day5）。面试时能说"我先脱离框架手写了完整 RAG 流水线，再用 LangChain 重构"，比只说"我用了 LangChain"有深度得多。


# Week1 总结

```
Day1: LLM API 基础         → 理解"生成"
Day2: Embedding + 向量库    → 理解"检索"的基础
Day3: 手写 RAG              → 把检索和生成串起来
Day4: 混合检索 + Rerank     → 优化检索质量（召回→精排→生成三阶段）
Day5: LangChain 框架        → 用工具提效 + 多轮对话 RAG
```

### 技术栈
| 组件 | 选型 |
|------|------|
| LLM | DeepSeek-V3（兼容 OpenAI SDK） |
| Embedding | BAAI/bge-small-zh-v1.5（本地） |
| Rerank | BAAI/bge-reranker-base（本地 Cross-Encoder） |
| 向量数据库 | ChromaDB |
| 关键词检索 | rank_bm25 + jieba |
| 框架 | LangChain |

### 能力自检
- [x] 能手写完整 RAG 流程（不依赖框架）
- [x] 能解释 Embedding 原理和局限性
- [x] 能区分 Bi-Encoder 和 Cross-Encoder
- [x] 能用混合检索（向量 + BM25 + RRF）优化召回
- [x] 能用 Hit Rate / MRR 量化评估检索质量
- [x] 能用 LangChain LCEL 快速搭建 RAG Chain
- [x] 能实现带查询改写的多轮对话 RAG
- [x] 能讲清楚 RAG vs Fine-tuning 的区别和选型
- [x] 能讲清楚分层评估和调优方法论


# 面试高频考点

## LCEL 与框架类
1. **LCEL 的核心价值是什么？和普通函数调用有什么区别？**
   - 组件可替换、自动支持流式/批量/异步、声明式写法更简洁
2. **RunnablePassthrough 和 RunnableLambda 分别是什么？**
   - Passthrough 原样传递输入；Lambda 包装自定义函数。配合字典结构让一个输入走多条路径
3. **你为什么选 LangChain？它的优缺点？**
   - 优：统一接口、组件可替换、生态丰富
   - 缺：有抽象开销、更新频繁 API 不稳定、极致性能场景可能需要手写
4. **LangChain vs LlamaIndex 怎么选？**
   - LangChain：通用编排框架，擅长串联各种组件
   - LlamaIndex：专注数据索引和 RAG，开箱即用
   - 简单 RAG 用 LlamaIndex 更快，复杂编排用 LangChain 更灵活

## 多轮对话类
5. **多轮对话 RAG 和普通 RAG 的区别？多了什么步骤？**
   - 多了查询改写（Contextualization），因为向量数据库不懂对话上下文
6. **查询改写会增加一次 LLM 调用，怎么优化？**
   - 先判断是否需要改写（规则或 LLM 判断），不需要则跳过
   - 用更小/更快的模型做改写（不需要用主力模型）
7. **对话历史越来越长怎么办？**
   - 滑动窗口：只保留最近 N 轮
   - 摘要压缩：用 LLM 将长历史总结为一段 summary
   - 向量检索：把历史存入向量库，每次只检索相关片段
   - 混合策略：最近 3 轮保留原文 + 更早的用 summary

## MMR 类
8. **MMR 解决什么问题？lambda_mult 参数的作用？**
   - 避免检索结果主题过于集中，在相关性和多样性之间平衡
   - lambda_mult 越小越多样，越大越接近普通检索
9. **什么场景适合用 MMR，什么场景用普通 similarity？**
   - MMR：知识库同主题 chunk 多、需要多角度参考时
   - similarity：需要精准匹配单一主题、或知识库本身主题分散时

## 综合类
10. **用 3 分钟讲清楚你的 RAG 系统架构**
    - 离线侧：文档加载 → 清洗 → 切片 → 同时构建向量索引和 BM25 索引
    - 在线侧：（查询改写 →）混合检索 → Rerank → Prompt 拼接 → LLM 生成
    - 评估：Hit Rate / MRR 量化检索质量，分层评估逐层调优
11. **如果让你从零搭一个生产级 RAG，你会怎么做？**
    - 先手写跑通验证可行性 → 用 LangChain 重构提效 → 构建评测集量化效果 → 分层调优（切片→检索→Rerank→Prompt）→ 持久化（Elasticsearch + 向量库）→ 加多轮对话支持 → 上线后持续监控和迭代
