# 核心要点
1. Embedding 做了一件什么事？为什么它是 RAG 的基础？
2. 余弦相似度和欧氏距离有什么区别？项目中选哪个？
3. 向量数据库和 MySQL 这种传统数据库的本质区别是什么？
4. 你在练习4中发现了 Embedding 的哪些局限性？
   这些局限性在实际项目中怎么应对？（提示：Rerank、chunking）
5. 回顾你的食材管理项目：如果加一个"用户随便描述，就能找到相关食品安全知识"
   的功能，你现在知道怎么实现了吗？

# 练习
## Part1
看看文本变成向量后到底长什么样

### 理解
通过embedding把文本映射成固定长度的向量，向量维度取决于embedding模型。这些向量在向量空间里越接近，说明语义越相似。embedding后的向量人难以理解，但这却是非常重要的基础，后续的attention就是基于embedding后的向量加工处理的。embedding是静态语义，经过attention后就会变成动态语义，对于同个词，比如"苹果"，对应的向量就是A，但这个A是多个语义向量后平均的结果，类似（不恰当）于A = 0.5x苹果公司的苹果+0.5x可食用苹果的苹果，实际远远不止这些语义，类似这样的，经过attention(其他词)的影响后，会更接近具体语义的向量。

### 实验

### 结论
同一个词多次embedding后的向量一致，未变化，即embedding是静态语义，是平均语义，经过attention加工处理后才变成具体语义
embedding提供“语义起点”，同时提供“可计算空间”。


### 疑问
1. embedding 与 transformer 的关系
2. embedding背后的计算逻辑是什么每次同一个问题得到的向量内容一样



# 练习
## Part2
验证 "语义相近的文本向量距离也近"

### 理解
欧氏距离：绝对距离（位置+大小）
余弦相似度：方向相似
embedding 的特点：长度不稳定，scale不重要，方向才代表语义，那余弦相似度更符合“语义空间”的本质


### 实验

### 结论
在文本相似度，RAG检索，embedding搜索，attention机制（点积）场景下用余弦
在地里距离，数值异常检测中用欧氏
归一化后：欧氏距离近似余弦相似度
但在embedding中，若没有归一化，高维度的向量中，欧氏距离的公式就会导致平均化，且维度越高平均化越严重，这对于语义检索中却是致命的

### 疑问
余弦相似度&欧氏距离于embedding来说，是什么？什么场景使用


## Part3
搭建一个最简单的向量数据库，体验"存"和"查"

### 理解
向量数据库搭建通常有以下流程：原始数据 -> 清洗&切分 -> embedding(转向量) -> 存入向量数据库（带索引）
原始数据可以是网页，文档（PDF/Word），日志，FAQ，但原始数据通常会比较长，比较乱，容易导致语义不聚焦，检索精度下降，所以需要进行清洗和切割，然后embedding。通常结构长这样
```
{
  "id": "chunk1",
  "vector": [...],
  "text": "苹果公司成立于1976年",
  "metadata": {
    "source": "wiki",
    "type": "company"
  }
}
```
vector用于搜索，text（原文）用于返回，metadata用于过滤。
另外数据库要建立索引，不然检索会很慢，常见索引算法暴力搜索（精准但慢），IVF倒排文件（分桶），HNSW图结构（最常用），FAISS/Annoy(工具库)

### 实验

### 结论
embedding 是核心，决定语义空间质量，embdding好（清洗，切割）->检索准
chunking决定上限，太大容易模糊，太小会信息不完整
向量数据库搜索的本质是相似度排序，返回相似度最高的


## Part4
理解 Embedding 不是万能的，知道局限才能在中展现深度

### 理解
"知其然知其所以然"，光知道embedding将词/句向量化并通过余弦相似度找到相近语义不够，要知道是不是任何词，任何语义都能准确识别。中华文化博大精深，一次多义是很常见的，对于大部分embedding模型（我猜的），这一点都做得不好，因为embedding算法都是平均语义，且是静态的，不会根据上下文动态变化。也就容易出现词相同，语义不同，却有较高的相似度。除此之外，长文本embedding后容易丢失细节，一些关键信息被淡化了。还有一个容易忽视的就是否定句，以及双重否定表肯定这种，部分embedding模型对这类也处理的不好。

### 实验

### 结论
embedding是静态的，对于部分场景处理存在短板，这个可以针对性的处理，不是说随便选个embedding模型就一劳永逸，要知道embedding的不足之处，并针对性的优化调整，以达到最大化的识别。对于大文档，适当的切片。对于中文，选择适当的模型等等。


---

# 复盘纠正

## 一、写错的地方

### 1. Part1：混淆了「词级 Embedding」和「句级 Embedding」（核心错误）

原文写了"embedding是静态语义，经过attention后就会变成动态语义"，并用"苹果 = 0.5x苹果公司 + 0.5x水果苹果"来类比。
这里把两件事搅在一起了，需要区分清楚：

- **Transformer 内部的 Embedding Layer**（如 `nn.Embedding`）：这一层确实是静态的查找表，每个 token 对应一个固定向量，不感知上下文。经过多层 Self-Attention 后变成上下文相关的表示。"苹果 = 0.5x苹果公司 + 0.5x水果苹果"这个比喻只适用于这一层（以及 Word2Vec/GloVe 等老一代词向量模型）。

- **Sentence Transformer 输出的 Embedding**（实验中用的 `bge-small-zh-v1.5`）：这个模型内部**已经跑完了所有的 Attention 层**，输出的是整句的上下文感知向量，**不是静态的**。它已经根据上下文做了语义消歧。

结论：在 Day2 实验的语境下，说"embedding 是静态语义"是不准确的。`embed_model.encode("苹果很好吃")` 和 `embed_model.encode("苹果手机很好用")` 得到的向量是不同的——模型已经通过内部 Attention 区分了上下文。

### 2. Part1 结论推理有误

原文："同一个词多次embedding后的向量一致，未变化，即embedding是静态语义"

「同一输入多次 encode 结果一致」**不能**推导出「embedding 是静态语义」。结果一致是因为**模型推理是确定性的**（没有 temperature/采样），跟"静态 vs 动态"无关。就好比 `1 + 1` 每次算都等于 2，不代表加法是"静态"的。

### 3. Part2 错别字

「地里距离」→「地理距离」

### 4. Part3 错别字

`embdding` → `embedding`

### 5. Part3：FAISS/Annoy 分类不当

原文把 FAISS/Annoy 和索引算法（暴力搜索、IVF、HNSW）并列。FAISS 是**库**，它里面支持多种算法（Flat/IVF/HNSW/PQ 等）。应该拆开写：
- **索引算法**：暴力搜索（精准但慢）、IVF 倒排文件（分桶）、HNSW 图结构（最常用）、PQ 乘积量化（压缩存储）
- **工具库**：FAISS（Meta 开源，支持以上多种算法）、Annoy（Spotify 开源，基于树结构）

### 6. Part4 笔误 + 判断修正

- 「一次多义」→「一词多义」
- 「对于大部分embedding模型，这一点都做得不好，因为embedding算法都是平均语义，且是静态的」——这个判断有误。现代 Sentence Transformer（包括 BGE）是句级模型，内部已经过 Attention 处理，对一词多义的处理**比想象的好很多**。Part4 实验中「苹果很好吃 vs 苹果手机很好用」的相似度应该不高，可以回头看实验数据验证。真正做得不好的是老一代的 Word2Vec / GloVe 这类词级静态 Embedding。


## 二、没有写到位的地方

### 1. 四个 Part 的「实验」部分都是空白

跑了代码、有了结论，但缺少关键的实验数据记录。建议至少补上：
- Part2 的相似度排序结果和具体数值
- Part3 的 ChromaDB 查询 Top3 结果和 distance 值
- Part4 的否定语义对比值、一词多义对比值
- 这些数据是结论的论据，没有它们结论缺乏说服力

### 2. Part2 缺少数学直觉

"归一化后欧氏距离近似余弦相似度"这句可以更精确：
> 当向量 L2-归一化后（||v|| = 1），有 **||a - b||² = 2(1 - cos(a,b))**，所以欧氏距离和余弦相似度是**单调等价**的，不只是"近似"。

### 3. Part3 没回答核心要点第3题：向量数据库 vs 传统数据库的本质区别

核心区别：
- **传统数据库**：精确匹配（`WHERE name = 'xx'`），基于 B-Tree/Hash 索引
- **向量数据库**：近似最近邻（ANN），基于 HNSW/IVF 等索引，返回的是**相似度排序**而非精确匹配，存在召回率 vs 性能的 trade-off

### 4. Part3 缺少 ANN vs KNN 的讨论

向量数据库（除了暴力搜索）返回的都是**近似结果**（Approximate Nearest Neighbor），不保证返回真正的 Top-K。这是工程中的关键取舍——用精度换速度。通过参数（如 HNSW 的 `ef_search`、IVF 的 `nprobe`）控制精度-速度平衡。

### 5. 核心要点第5题没有作答

「如果给食材管理项目加一个语义搜索功能，怎么实现？」——直接关联实际应用，建议补上完整链路思路。


## 三、思考深度不够的地方

### 1. Chunking 需要展开

「太大容易模糊，太小会信息不完整」是对的，但不够深入：

- **常见策略**：固定大小（如 512 tokens）、按段落/句子切分、递归切分（LangChain 的 `RecursiveCharacterTextSplitter`）
- **Overlap 重叠**：相邻 chunk 有重叠区域（如 chunk_size=500, overlap=50），防止关键信息被切断在边界上
- **语义完整性**：按句号/段落分割 vs 按 token 数硬切，效果差很多
- **核心取舍**：小 chunk → 检索精准但上下文不足，大 chunk → 上下文丰富但语义模糊。这是 RAG 工程中最核心的调参项之一

### 2. 完全没有提 Rerank

核心要点第4题提示了 Rerank，但笔记里没有展开。Rerank 是 RAG 检索链路中非常重要的一环：

- **为什么需要**：Embedding 检索（双塔模型/Bi-Encoder）是"粗排"，速度快但精度有限；Rerank（交叉编码器/Cross-Encoder）是"精排"，把 query 和每个候选 document 拼接在一起输入模型，能更准确地判断相关性
- **原理区别**：Embedding 是双塔（query 和 doc 分别编码再比较），Rerank 是单塔（query + doc 拼接后一起编码），单塔精度更高但计算开销大
- **典型 pipeline**：Embedding 召回 Top-50 → Rerank 精排 → 取 Top-5 送给 LLM

### 3. 没有提混合检索（Hybrid Search）

纯向量检索的弱点是对**精确关键词匹配**不擅长（比如搜型号「iPhone 15 Pro Max」，向量检索可能返回语义相关但型号不对的结果）。实际项目中常用：

- **稀疏检索**（BM25/TF-IDF）：擅长关键词精确匹配
- **稠密检索**（Embedding）：擅长语义模糊匹配
- **混合检索**：两者结合，用 RRF（Reciprocal Rank Fusion）等算法合并排序结果

### 4. 没有讨论 Embedding 模型的选型标准

「选择适当的模型」需要展开：
- **MTEB 排行榜**（Massive Text Embedding Benchmark）：当前中文 Embedding 模型排行的参考标准
- **维度**：512 维 vs 1024 维 vs 1536 维——维度越高表达力越强，但存储和计算开销越大
- **最大输入长度**：BGE-small 是 512 tokens，一些模型支持 8192 tokens
- **多语言支持**：如果涉及跨语言场景，需要选 multilingual 模型


## 四、面试高频考点

### 基础概念类
1. **Embedding 的本质是什么？和 One-Hot 编码有什么区别？**
   - One-Hot 是稀疏的、正交的（任意两个词距离相同，无法表达语义关系）；Embedding 是稠密的、语义连续的（相似语义距离近）
2. **Word2Vec、GloVe 和 Sentence Transformer 的区别？**
   - Word2Vec/GloVe：词级，静态，同一个词不管出现在什么上下文中向量都一样
   - Sentence Transformer：句级，动态（上下文感知），同一个词在不同句子中向量不同
   - 面试官常用来考对"静态 vs 动态 Embedding"的理解深度
3. **余弦相似度和欧氏距离什么时候等价？**
   - L2 归一化后单调等价：`||a-b||² = 2(1 - cos(a,b))`

### RAG 工程类
4. **RAG 的完整链路是什么？每一步可能出问题的地方在哪？**
   - 文档加载 → Chunking → Embedding → 索引存储 → Query Embedding → 检索 → Rerank → Prompt 组装 → LLM 生成
   - 每一步都可能是瓶颈：chunk 太大/太小、Embedding 模型不合适、Top-K 设太小、没有 Rerank、Prompt 没有引导好格式等
5. **Chunking 策略怎么选？chunk_size 和 overlap 怎么定？**
   - 结合文档类型、模型上下文窗口、检索精度需求综合考虑
   - 通常 chunk_size 200-1000 tokens，overlap 10%-20%，需要实验调优
6. **为什么需要 Rerank？双塔和交叉编码器的区别？**
   - 双塔（Bi-Encoder）：query 和 doc 分别编码，速度快，适合粗排
   - 交叉编码器（Cross-Encoder）：query + doc 拼接编码，精度高，适合精排
7. **纯向量检索有什么不足？怎么解决？**
   - 对精确关键词匹配弱 → 引出混合检索（BM25 + Dense）和 RRF 融合

### 向量数据库类
8. **HNSW 的基本原理是什么？为什么快？**
   - 分层可导航小世界图（Hierarchical Navigable Small World）
   - 通过多层跳表结构实现 O(log N) 的近似最近邻搜索
   - 上层稀疏（大步跳转定位区域），下层稠密（精细搜索）
9. **ANN 和 KNN 的区别？ANN 的 trade-off 是什么？**
   - KNN：精确最近邻，必须和所有点比较，O(N)
   - ANN：近似最近邻，用召回率换速度
   - 通过参数（如 HNSW 的 `ef_search`、IVF 的 `nprobe`）控制精度-速度平衡
10. **向量数据库选型：ChromaDB vs Milvus vs Pinecone vs Qdrant？**
    - ChromaDB：轻量，适合原型验证和小规模场景
    - Milvus：分布式，适合大规模生产环境
    - Pinecone：全托管 SaaS，免运维
    - Qdrant：Rust 实现，性能好，API 友好

### 深度追问类
11. **Embedding 模型怎么选？你了解 MTEB 吗？**
    - MTEB（Massive Text Embedding Benchmark）是当前主流的 Embedding 模型评测基准
    - 选型要看具体任务（检索、分类、聚类）、语言、维度、速度
12. **高维空间中，为什么所有点之间的欧氏距离趋于相同？**
    - 维度灾难（Curse of Dimensionality）：随维度增加，最近点和最远点之间的距离差占比趋近于 0，欧氏距离的区分度下降
13. **如果检索召回的内容不相关，你会从哪些环节排查？**
    - Chunking 粒度是否合适 → Embedding 模型质量 → 检索参数（Top-K、distance 阈值）→ 是否需要 Rerank → Query 是否需要改写/扩展
