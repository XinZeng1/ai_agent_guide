"""
=============================================================
  AI工程师转型 - 第1周第4天：动手实践
  主题：RAG进阶 — 重排序（Rerank）与混合检索
=============================================================

使用前准备：
  pip install openai chromadb sentence-transformers numpy rank_bm25 jieba

今天的目标：
  昨天你搭了一个能跑的RAG，但检索质量还有提升空间。
  今天学两个关键优化手段：
  1. BM25关键词检索 + 向量检索 = 混合检索（取长补短）
  2. 重排序（Rerank）：对初步检索结果做二次精排

  这两个是面试中区分"入门"和"进阶"的分水岭。

为什么需要优化？
  向量检索的问题：语义相近但细节不匹配（如"苹果保存"匹配到"香蕉保存"）
  关键词检索的问题：同义词搜不到（如搜"保鲜"找不到"保存"）
  → 混合检索：两者互补
  → Rerank：在混合结果上再做一轮精排，把最相关的排到最前面
"""

import os
import token
import numpy as np
import jieba
from collections import Counter
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

# ========================================
# 🔧 配置
# ========================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "你的API Key")

llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

print("✅ Embedding模型加载完成")

# ========================================
# 准备数据：复用昨天的食品安全知识库
# ========================================
KNOWLEDGE_BASE = [
    "新鲜鸡蛋在冰箱冷藏环境（0-4摄氏度）下通常可以保存30到45天左右。在常温环境下（25度以上），保存时间会大大缩短，一般只有10到15天。鸡蛋表面有一层保护膜，清洗后保护膜被破坏，保质期会明显缩短，所以不建议提前清洗。",
    "判断鸡蛋是否变质有几种方法。最简单的是水浮法：将鸡蛋轻轻放入水中，新鲜鸡蛋会沉入水底平躺，存放较久的会一端翘起，完全变质的会浮在水面上。也可以摇晃鸡蛋：新鲜的摇晃时无声，不新鲜的会有明显晃荡感。",
    "存放鸡蛋的最佳方式是大头朝上放置，因为气室在大头一端，这样可以减缓蛋黄上浮接触蛋壳的速度。不建议放在冰箱门上，因为开关门的温度变化会影响保鲜效果，最好放在冰箱内部靠里的位置。",
    "巴氏消毒奶（鲜奶）开封后应在2-3天内饮用完毕，未开封的按包装上的保质期存放（通常7-15天，需冷藏）。UHT牛奶（超高温灭菌奶）未开封可在常温下保存6个月左右，但开封后同样需要冷藏并在3天内饮完。",
    "酸奶在冰箱中通常可以保存到保质期后一周左右，但口感和活菌数量会逐渐下降。自制酸奶因为没有工业级密封，保存时间更短，建议3天内食用。",
    "奶酪的保存时间差异很大：软质奶酪（如马苏里拉）开封后1周内食用；硬质奶酪（如帕玛森）可以保存数月。出现霉点的硬质奶酪可以切掉霉变部分继续食用，但软质奶酪出现霉变应整块丢弃。",
    "生肉（猪牛羊鸡）在冰箱冷藏（0-4度）可保存1-3天，冷冻（-18度以下）可保存3-6个月。绞肉因为接触空气面积大，变质更快，冷藏不超过1天。",
    "海鲜类是所有食品中最容易变质的。新鲜鱼类冷藏不超过2天，虾类冷藏不超过1天。购买后应尽快处理或冷冻。解冻肉类和海鲜的最佳方式是放在冰箱冷藏室缓慢解冻。",
    "解冻后的肉类和海鲜不应二次冷冻，因为解冻过程中已经有细菌繁殖，再次冷冻只是让细菌休眠而非杀死它们，下次解冻后细菌数量会更多。",
    "米饭煮熟后在常温下不应超过2小时，冰箱冷藏可保存1-2天。重新加热时必须彻底加热到75度以上，避免蜡样芽孢杆菌中毒。",
    "剩菜剩饭应在烹饪后2小时内放入冰箱，冷藏保存不超过3天。叶菜类不建议二次加热，肉类和根茎类可以。",
    "蜂蜜是唯一不会变质的天然食品，但必须密封保存。结晶是正常现象，不影响品质。掺假蜂蜜可能含有糖浆，可通过拉丝测试和水溶测试鉴别。",
    "速冻食品应在-18度以下保存，解冻后应尽快食用。微波解冻最安全，常温解冻容易滋生细菌。反复解冻冷冻会严重影响口感和安全性。",
    "食用油开封后应在3个月内用完，存放在阴凉避光处。油脂氧化会产生有害物质，闻到哈喇味说明已经变质，不应继续食用。",
    "豆腐开封后冷藏可保存3-5天，需每天换水。变质的豆腐会发黏、变黄、有异味。冻豆腐可以保存更久但口感会变化。",
    "苹果在冰箱冷藏（0-4度）可以保存1-2个月。苹果会释放乙烯气体，应与其他水果分开存放，否则会加速其他水果成熟变质。",
    "香蕉不适合放冰箱，低温会让香蕉皮变黑。最佳保存方式是挂起来通风存放，避免挤压。已经变黑的香蕉果肉如果没有异味仍可食用。",
]


# ========================================
# 练习1：BM25关键词检索
# ========================================
def exercise_1_bm25():
    """
    【目标】理解BM25（传统关键词检索）的工作原理

    【面试考点】
    BM25是经典的信息检索算法，基于词频（TF）和逆文档频率（IDF）：
    - TF：一个词在文档中出现越多次，文档越可能相关
    - IDF：一个词在所有文档中越罕见，它的区分能力越强
    - BM25 在 TF-IDF 基础上加了长度归一化和饱和函数

    BM25 vs 向量检索：
    - BM25：精确匹配关键词，快速，不需要GPU
    - 向量：语义匹配，能找到同义词，但可能不精确
    """
    print("=" * 60)
    print("练习1：BM25 关键词检索")
    print("=" * 60)

    # 中文需要先分词（jieba）
    tokenized_docs = [list(jieba.cut(doc)) for doc in KNOWLEDGE_BASE]

    # 构建BM25索引
    bm25 = BM25Okapi(tokenized_docs)

    # 测试查询
    queries = [
        "鸡蛋保存",         # 直接包含关键词
        "蛋怎么存放",       # 同义词替换
        "过期食品判断",     # 更抽象的表达
        "乙烯气体",         # 专业术语精确匹配
    ]

    for query in queries:
        tokenized_query = list(jieba.cut(query))
        scores = bm25.get_scores(tokenized_query)

        # 获取Top-3
        top_indices = np.argsort(scores)[::-1][:3]

        print(f"\n🔍 查询: 「{query}」 (分词: {tokenized_query})")
        for rank, idx in enumerate(top_indices):
            print(f"   Top{rank+1} [分数:{scores[idx]:.4f}] {KNOWLEDGE_BASE[idx][:50]}...")

    print("\n💡 观察要点：")
    print("  - 「鸡蛋保存」直接匹配关键词，效果很好")
    print("  - 「蛋怎么存放」里'蛋'和'存放'未必能匹配'鸡蛋'和'保存'")
    print("  - 「乙烯气体」这种专业术语，BM25比向量检索更精准！")
    print("  - BM25的弱点：依赖分词质量，无法理解语义相似")

    return bm25, tokenized_docs


# ========================================
# 练习2：混合检索（Hybrid Search）
# ========================================
def exercise_2_hybrid_search(bm25, tokenized_docs):
    """
    【目标】把向量检索和BM25结合起来，取长补短

    【面试考点】
    混合检索的核心思路：
    1. 分别用向量检索和BM25检索，各拿Top-K个结果
    2. 用 RRF（Reciprocal Rank Fusion）合并排名
    3. 最终排名 = 综合两种方法的结果

    RRF公式：score(d) = Σ 1/(k + rank_i(d))
    其中 k 通常取60，rank_i(d) 是文档d在第i个检索方法中的排名

    为什么用RRF而不是直接加分数？
    因为BM25分数和向量距离的量纲不同，不能直接相加
    RRF只用排名，绕过了量纲问题
    """
    print("\n" + "=" * 60)
    print("练习2：混合检索（向量 + BM25）")
    print("=" * 60)

    # 构建向量索引
    chroma_client = chromadb.Client()

    class LocalEmbeddingFunction:
        def __call__(self, input):
            return embed_model.encode(input).tolist()
        
        def embed_query(self, input):
            return self.__call__(input)

    try:
        chroma_client.delete_collection("hybrid_search")
    except:
        pass

    collection = chroma_client.create_collection(
        name="hybrid_search",
        embedding_function=LocalEmbeddingFunction(),
    )
    collection.add(
        documents=KNOWLEDGE_BASE,
        ids=[f"doc_{i}" for i in range(len(KNOWLEDGE_BASE))],
    )

    def hybrid_search(query, top_k=5, rrf_k=60):
        """
        混合检索：向量 + BM25，用RRF融合排名
        """
        # --- 向量检索 ---
        vec_results = collection.query(query_texts=[query], n_results=top_k)
        vec_ids = vec_results["ids"][0]  # ['doc_0', 'doc_3', ...]
        vec_doc_indices = [int(id.split("_")[1]) for id in vec_ids]

        # --- BM25检索 ---
        tokenized_query = list(jieba.cut(query))
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k].tolist()

        # --- RRF融合 ---
        rrf_scores = {}

        # 向量检索的排名贡献
        for rank, doc_idx in enumerate(vec_doc_indices):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (rrf_k + rank + 1)

        # BM25的排名贡献
        for rank, doc_idx in enumerate(bm25_top_indices):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (rrf_k + rank + 1)

        # 按RRF分数排序
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results, vec_doc_indices, bm25_top_indices

    # 测试对比
    test_queries = [
        "蛋怎么存放比较好",        # 同义词场景：BM25弱，向量强
        "乙烯气体对水果的影响",     # 专业术语场景：BM25强，向量可能也行
        "食物变质有哈喇味",         # 口语+专业术语混合
    ]

    for query in test_queries:
        sorted_results, vec_top, bm25_top = hybrid_search(query)

        print(f"\n{'─'*50}")
        print(f"🔍 查询: 「{query}」")
        print(f"{'─'*50}")

        print(f"\n  📊 向量检索 Top-3: {vec_top[:3]}")
        for idx in vec_top[:3]:
            print(f"     doc_{idx}: {KNOWLEDGE_BASE[idx][:40]}...")

        print(f"\n  📊 BM25检索 Top-3: {bm25_top[:3]}")
        for idx in bm25_top[:3]:
            print(f"     doc_{idx}: {KNOWLEDGE_BASE[idx][:40]}...")

        print(f"\n  🏆 混合检索（RRF融合）Top-3:")
        for rank, (doc_idx, score) in enumerate(sorted_results[:3]):
            in_vec = "✅" if doc_idx in vec_top[:5] else "❌"
            in_bm25 = "✅" if doc_idx in bm25_top[:5] else "❌"
            print(f"     Top{rank+1} [RRF:{score:.6f}] [向量{in_vec}|BM25{in_bm25}] {KNOWLEDGE_BASE[doc_idx][:40]}...")

    print("\n💡 观察要点：")
    print("  - 混合检索的结果是否比单独任一方法更全面？")
    print("  - 「蛋怎么存放」向量检索能找到'鸡蛋保存'，BM25可能找不到")
    print("  - 「乙烯气体」BM25精确命中，向量检索可能不那么精确")
    print("  - RRF融合让两种方法的优势互补 → 这就是混合检索的价值")

    return collection


# ========================================
# 练习3：重排序（Rerank）
# ========================================
def exercise_3_rerank(collection):
    """
    【目标】理解 Rerank 的原理和作用

    【面试考点】
    Rerank的核心思想：
    1. 初步检索（向量/BM25/混合）拿到一批候选文档（比如Top-20）
    2. 用一个更精细的模型（Cross-Encoder）逐一判断"查询-文档"的相关性
    3. 根据精细分数重新排序，取Top-K

    为什么不直接用Cross-Encoder做检索？
    - Cross-Encoder需要把query和每个文档拼接后逐对打分
    - 10万个文档就要跑10万次模型，太慢了
    - 所以先用"粗检索"（快但不精确）→ 再用"精排序"（慢但精确）
    - 这是经典的「召回-排序」两阶段架构

    Bi-Encoder（向量检索） vs Cross-Encoder（Rerank）：
    - Bi-Encoder：query和doc分别编码，用余弦相似度比较 → 快
    - Cross-Encoder：query和doc拼在一起输入模型，输出相关性分数 → 准
    """
    print("\n" + "=" * 60)
    print("练习3：重排序（Rerank）— 让检索结果更精确")
    print("=" * 60)

    # 加载Cross-Encoder模型（用于Rerank）
    print("正在加载Rerank模型（首次需要下载，约100MB）...")
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    print("✅ Rerank模型加载完成\n")

    def search_with_rerank(query, initial_top_k=10, final_top_k=3):
        """
        两阶段检索：粗检索 → 精排序
        """
        # Stage 1：粗检索（向量检索，快速拿到候选集）
        results = collection.query(query_texts=[query], n_results=initial_top_k)
        candidate_docs = results["documents"][0]
        candidate_ids = results["ids"][0]
        distances = results["distances"][0]

        # Stage 2：Rerank（Cross-Encoder精排）
        # Cross-Encoder需要 [query, doc] 的配对输入
        pairs = [[query, doc] for doc in candidate_docs]
        rerank_scores = reranker.predict(pairs)

        # 按Rerank分数重新排序
        ranked = sorted(
            zip(candidate_docs, candidate_ids, distances, rerank_scores),
            key=lambda x: x[3],  # 按rerank分数排
            reverse=True
        )

        return ranked[:final_top_k], candidate_docs, distances

    # 测试
    test_queries = [
        "海鲜解冻后再冻一次可以吗",
        "怎么知道鸡蛋坏没坏",
        "吃剩的饭菜第二天还能吃吗",
    ]

    for query in test_queries:
        print(f"\n{'─'*50}")
        print(f"🔍 查询: 「{query}」")
        print(f"{'─'*50}")

        reranked, orig_docs, orig_dists = search_with_rerank(query)

        print(f"\n  📊 向量检索原始排名 Top-3：")
        for i in range(min(3, len(orig_docs))):
            print(f"     [{i+1}] [距离:{orig_dists[i]:.4f}] {orig_docs[i][:50]}...")

        print(f"\n  🏆 Rerank后排名 Top-3：")
        for i, (doc, doc_id, dist, score) in enumerate(reranked):
            # 找这个文档在原始排名中的位置
            orig_rank = orig_docs.index(doc) + 1 if doc in orig_docs else "?"
            move = f"原第{orig_rank}名"
            print(f"     [{i+1}] [Rerank分:{score:.4f}] ({move}) {doc[:50]}...")

    print("\n💡 观察要点：")
    print("  - Rerank后的排名和原始向量检索排名是否有变化？")
    print("  - 变化大 → 说明向量检索的粗排不够精确，Rerank修正了")
    print("  - 变化小 → 说明向量检索对这个query已经足够好了")
    print("  - 实际项目中：initial_top_k 通常取 10-20，final_top_k 取 3-5")

    return reranker


# ========================================
# 练习4：完整RAG优化流水线
# ========================================
def exercise_4_full_optimized_rag(collection, bm25, reranker):
    """
    【目标】把混合检索 + Rerank + LLM生成串在一起

    【面试加分项】
    能讲出"为什么要用两阶段"以及"每个环节的trade-off"：
    - 召回阶段追求高 Recall（宁可多不可漏）
    - 排序阶段追求高 Precision（精准排到前面）
    - 生成阶段追求 Faithfulness（忠实于检索到的内容）
    """
    print("\n" + "=" * 60)
    print("练习4：完整优化版RAG 🔥")
    print("=" * 60)

    def optimized_rag(query, show_process=True):
        """
        完整的优化RAG流水线：
        用户提问 → 混合检索(召回) → Rerank(精排) → 构造Prompt → LLM生成
        """

        if show_process:
            print(f"\n📌 Step 1: 混合检索（召回阶段）")

        # === 向量检索 ===
        vec_results = collection.query(query_texts=[query], n_results=10)
        vec_docs = vec_results["documents"][0]
        vec_ids = [int(id.split("_")[1]) for id in vec_results["ids"][0]]

        # === BM25检索 ===
        tokenized_query = list(jieba.cut(query))
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_top = np.argsort(bm25_scores)[::-1][:10].tolist()

        # === RRF融合 ===
        rrf_scores = {}
        for rank, idx in enumerate(vec_ids):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(bm25_top):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (60 + rank + 1)

        # 取Top-10候选
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        candidate_docs = [KNOWLEDGE_BASE[idx] for idx, _ in sorted_candidates]

        if show_process:
            print(f"   混合检索召回 {len(candidate_docs)} 个候选文档")

        # === Rerank精排 ===
        if show_process:
            print(f"\n📌 Step 2: Rerank（精排阶段）")

        pairs = [[query, doc] for doc in candidate_docs]
        rerank_scores = reranker.predict(pairs)

        ranked = sorted(
            zip(candidate_docs, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )

        # 取Top-3
        top_docs = [doc for doc, score in ranked[:3]]

        if show_process:
            for i, (doc, score) in enumerate(ranked[:3]):
                print(f"   Top{i+1} [分数:{score:.4f}] {doc[:40]}...")

        # === 构造Prompt并生成 ===
        if show_process:
            print(f"\n📌 Step 3: LLM生成")

        context = "\n\n---\n\n".join(top_docs)
        prompt = f"""你是一位专业的食品安全顾问。请严格根据以下参考资料回答用户的问题。

要求：
1. 只根据参考资料中的信息回答，不要编造内容
2. 如果参考资料中没有相关信息，请明确告知
3. 回答简洁实用，包含具体数字和操作建议

参考资料：
{context}

用户问题：{query}

请回答："""

        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )

        answer = response.choices[0].message.content
        usage = response.usage

        if show_process:
            print(f"   Token消耗: 输入={usage.prompt_tokens}, 输出={usage.completion_tokens}")

        return answer

    # 测试
    questions = [
        "我买了三文鱼，解冻吃了一半，剩下的能再冻起来吗？",
        "鸡蛋到底应该怎么放冰箱？放门上行不行？",
        "炒菜剩了很多，明天带饭还能吃吗？叶子菜呢？",
    ]

    for q in questions:
        print(f"\n{'='*50}")
        print(f"👤 {q}")
        print(f"{'='*50}")
        answer = optimized_rag(q)
        print(f"\n🤖 回答:\n{answer}")

    print("\n💡 完整流水线总结（面试时这样讲）：")
    print("  1.「召回」混合检索（向量+BM25+RRF），保证不遗漏 → 高Recall")
    print("  2.「精排」Cross-Encoder Rerank，把最相关的排到前面 → 高Precision")
    print("  3.「生成」LLM基于精排后的Top-K文档生成回答 → 高Faithfulness")
    print("  每个阶段都有明确的目标，这就是工程化的RAG系统")


# ========================================
# 练习5：检索质量评估
# ========================================
def exercise_5_evaluation(collection, bm25, reranker):
    """
    【目标】学会用简单指标评估RAG检索质量

    【面试考点】
    RAG的评估维度：
    1. 检索质量：检索到的文档是否包含回答问题所需的信息
       - Hit Rate（命中率）：Top-K中是否包含正确文档
       - MRR（Mean Reciprocal Rank）：正确文档排在第几位
    2. 生成质量：
       - Faithfulness（忠实度）：回答是否基于检索到的内容
       - Relevance（相关度）：回答是否切题
       - 这部分通常用LLM-as-Judge来评估（让另一个LLM打分）
    """
    print("\n" + "=" * 60)
    print("练习5：检索质量评估 — 怎么知道你的RAG好不好")
    print("=" * 60)

    # 构建评估数据集：问题 + 期望命中的文档索引
    eval_dataset = [
        {"query": "鸡蛋放冰箱能保存多少天", "relevant_doc_indices": [0]},
        {"query": "怎么判断鸡蛋有没有坏", "relevant_doc_indices": [1]},
        {"query": "牛奶开封后能放多久", "relevant_doc_indices": [3]},
        {"query": "解冻的肉能不能再冷冻", "relevant_doc_indices": [8]},
        {"query": "苹果怎么保存不会坏", "relevant_doc_indices": [15]},
        {"query": "剩饭剩菜怎么处理", "relevant_doc_indices": [10, 11]},
        {"query": "食用油什么时候算变质", "relevant_doc_indices": [13]},
        {"query": "香蕉能放冰箱吗", "relevant_doc_indices": [16]},
    ]

    def evaluate_retrieval(search_fn, method_name, top_k=3):
        """评估检索方法的 Hit Rate 和 MRR"""
        hit_count = 0
        mrr_sum = 0

        for item in eval_dataset:
            query = item["query"]
            relevant = set(item["relevant_doc_indices"])

            # 获取检索结果的文档索引
            result_indices = search_fn(query, top_k)

            # Hit Rate: Top-K中是否包含至少一个正确文档
            hit = bool(relevant & set(result_indices))
            if hit:
                hit_count += 1

            # MRR: 第一个正确文档的排名的倒数
            for rank, idx in enumerate(result_indices):
                if idx in relevant:
                    mrr_sum += 1.0 / (rank + 1)
                    break

        hit_rate = hit_count / len(eval_dataset)
        mrr = mrr_sum / len(eval_dataset)
        return hit_rate, mrr

    # 定义三种检索方法
    def vector_search(query, top_k):
        results = collection.query(query_texts=[query], n_results=top_k)
        return [int(id.split("_")[1]) for id in results["ids"][0]]

    def bm25_search(query, top_k):
        tokenized = list(jieba.cut(query))
        scores = bm25.get_scores(tokenized)
        return np.argsort(scores)[::-1][:top_k].tolist()

    def hybrid_rerank_search(query, top_k):
        # 混合召回
        vec_results = collection.query(query_texts=[query], n_results=10)
        vec_ids = [int(id.split("_")[1]) for id in vec_results["ids"][0]]

        tokenized = list(jieba.cut(query))
        bm25_scores_arr = bm25.get_scores(tokenized)
        bm25_top = np.argsort(bm25_scores_arr)[::-1][:10].tolist()

        rrf = {}
        for rank, idx in enumerate(vec_ids):
            rrf[idx] = rrf.get(idx, 0) + 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(bm25_top):
            rrf[idx] = rrf.get(idx, 0) + 1.0 / (60 + rank + 1)

        candidates = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:10]
        candidate_docs = [(idx, KNOWLEDGE_BASE[idx]) for idx, _ in candidates]

        # Rerank
        pairs = [[query, doc] for _, doc in candidate_docs]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
        return [idx for (idx, _), _ in ranked[:top_k]]

    # 评估
    methods = [
        ("向量检索", vector_search),
        ("BM25检索", bm25_search),
        ("混合检索+Rerank", hybrid_rerank_search),
    ]

    print(f"\n评估数据集: {len(eval_dataset)} 个问题")
    print(f"评估指标: Hit Rate@3 和 MRR@3\n")

    print(f"{'方法':<20} {'Hit Rate@3':>12} {'MRR@3':>12}")
    print("─" * 46)

    for name, fn in methods:
        hit_rate, mrr = evaluate_retrieval(fn, name)
        bar = "█" * int(hit_rate * 20)
        print(f"{name:<20} {hit_rate:>10.1%}   {mrr:>10.4f}  {bar}")

    print(f"\n💡 指标解读：")
    print(f"  Hit Rate@3：Top-3中命中正确文档的比例 → 越高越好（目标>90%）")
    print(f"  MRR@3：正确文档平均排在第几位的倒数 → 越接近1越好")
    print(f"")
    print(f"  如果混合+Rerank比单独向量或BM25好 → 说明优化有效")
    print(f"  如果差不多 → 说明这个数据集太小/太简单，区分度不够")
    print(f"  实际项目中，评估数据集应该有50-100个以上才有统计意义")


# ========================================
# 🎯 运行入口
# ========================================
if __name__ == "__main__":
    print("🚀 AI工程师转型 - Day4 RAG进阶\n")

    exercises = {
        "1": "BM25关键词检索",
        "2": "混合检索（向量+BM25）",
        "3": "重排序（Rerank）",
        "4": "完整优化版RAG 🔥",
        "5": "检索质量评估",
        "all": "运行全部练习",
    }

    print("选择要运行的练习：")
    for key, name in exercises.items():
        print(f"  {key}. {name}")

    choice = input("\n请输入编号 (1/2/3/4/5/all): ").strip()

    # 初始化共享资源
    bm25_obj = None
    tokenized = None
    coll = None
    reranker_obj = None

    def ensure_bm25():
        global bm25_obj, tokenized
        if bm25_obj is None:
            tokenized = [list(jieba.cut(doc)) for doc in KNOWLEDGE_BASE]
            bm25_obj = BM25Okapi(tokenized)
        return bm25_obj, tokenized

    def ensure_collection():
        global coll
        if coll is None:
            client = chromadb.Client()
            try:
                client.delete_collection("hybrid_search")
            except:
                pass

            class EF:
                def __call__(self, input):
                    return embed_model.encode(input).tolist()
                
                def embed_query(self, input):
                    return self.__call__(input)

            coll = client.create_collection(name="hybrid_search", embedding_function=EF())
            coll.add(documents=KNOWLEDGE_BASE, ids=[f"doc_{i}" for i in range(len(KNOWLEDGE_BASE))])
        return coll

    def ensure_reranker():
        global reranker_obj
        if reranker_obj is None:
            print("正在加载Rerank模型...")
            reranker_obj = CrossEncoder("BAAI/bge-reranker-base")
            print("✅ Rerank模型加载完成")
        return reranker_obj

    if choice == "1":
        exercise_1_bm25()
    elif choice == "2":
        b, t = ensure_bm25()
        exercise_2_hybrid_search(b, t)
    elif choice == "3":
        ensure_collection()
        exercise_3_rerank(coll)
    elif choice == "4":
        b, t = ensure_bm25()
        ensure_collection()
        ensure_reranker()
        exercise_4_full_optimized_rag(coll, bm25_obj, reranker_obj)
    elif choice == "5":
        b, t = ensure_bm25()
        ensure_collection()
        ensure_reranker()
        exercise_5_evaluation(coll, bm25_obj, reranker_obj)
    elif choice == "all":
        bm25_obj, tokenized = exercise_1_bm25()
        print("\n" + "🔹" * 30 + "\n")
        coll = ensure_collection()
        exercise_2_hybrid_search(bm25_obj, tokenized)
        print("\n" + "🔹" * 30 + "\n")
        reranker_obj = exercise_3_rerank(coll)
        print("\n" + "🔹" * 30 + "\n")
        exercise_4_full_optimized_rag(coll, bm25_obj, reranker_obj)
        print("\n" + "🔹" * 30 + "\n")
        exercise_5_evaluation(coll, bm25_obj, reranker_obj)

    print("\n\n" + "=" * 60)
    print("✅ Day4 练习完成！")
    print("=" * 60)
    print("""
📝 今日思考题：

1. BM25和向量检索各自的优缺点是什么？为什么要混合使用？

2. RRF（Reciprocal Rank Fusion）的思路是什么？
   为什么不直接把BM25分数和向量距离加起来？

3. Rerank用的Cross-Encoder和Embedding用的Bi-Encoder有什么区别？
   为什么不直接用Cross-Encoder做全量检索？

4. Hit Rate 和 MRR 分别衡量什么？
   在你的食材管理项目中，哪个指标更重要？

5. 完整说一遍优化版RAG的流程：
   召回（混合检索） → 精排（Rerank） → 生成（LLM）
   每个阶段的目标分别是什么？

明天的内容：LangChain框架 — 用框架重写RAG，对比手写的差异 🔧
""")

