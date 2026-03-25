"""
=============================================================
  AI工程师转型 - 第1周第2天：动手实践
  主题：Embedding 与向量数据库（ChromaDB）
  API：DeepSeek（暂不提供Embedding） → 使用本地模型替代
=============================================================

使用前准备：
  pip install chromadb sentence-transformers numpy

说明：
  DeepSeek目前没有公开的Embedding API，所以我们用开源的
  sentence-transformers 在本地跑Embedding模型。
  这反而是好事——你能更直观地理解Embedding的工作原理，
  而且时说"我用过本地Embedding模型"比只调API更加分。

  首次运行会自动下载模型（约100MB），之后就走本地缓存了。
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

# ========================================
# 🔧 加载本地Embedding模型
# ========================================
print("正在加载Embedding模型（首次需要下载，约100MB）...")
# BAAI/bge-small-zh-v1.5 是中文效果很好的轻量Embedding模型
# 输出维度：512维
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print("✅ 模型加载完成\n")


# ========================================
# 练习1：直观感受 Embedding
# ========================================
def exercise_1_what_is_embedding():
    """
    【目标】看看文本变成向量后到底长什么样
    
    【考点】
    - Embedding 把任意长度的文本映射为固定长度的向量
    - 语义相似的文本，向量也相似（距离近）
    - 这是所有语义搜索、RAG 的基础
    """
    print("=" * 60)
    print("练习1：文本 → 向量，到底发生了什么？")
    print("=" * 60)

    text = "苹果放冰箱能保存多久"

    # 文本 → 向量
    vector = embed_model.encode(text)

    print(f"\n原始文本: 「{text}」")
    print(f"向量维度: {vector.shape[0]} 维")
    print(f"向量前20个数: {vector[:20].round(4)}")
    print(f"向量数值范围: [{vector.min():.4f}, {vector.max():.4f}]")
    print(f"向量的数据类型: {vector.dtype}")

    print("========同类问题，对比向量差异======")

    text = "香蕉放冰箱能保存多久"

    # 文本 → 向量
    vector = embed_model.encode(text)

    print(f"\n原始文本: 「{text}」")
    print(f"向量维度: {vector.shape[0]} 维")
    print(f"向量前20个数: {vector[:20].round(4)}")
    print(f"向量数值范围: [{vector.min():.4f}, {vector.max():.4f}]")
    print(f"向量的数据类型: {vector.dtype}")

    print("\n💡 关键理解：")
    print("  - 这段文字被压缩成了一个512维的数字数组")
    print("  - 每个维度捕捉了某种语义特征（但不是人类能直接理解的）")
    print("  - 不管输入多长的文本，输出都是512维（由模型决定）")
    print("  - 你之前做食材管理时调用DeepSeek，内部也是先做Embedding再处理的")


# ========================================
# 练习2：语义相似度计算
# ========================================
def exercise_2_similarity():
    """
    【目标】验证 "语义相近的文本向量距离也近"
    
    【考点】
    - 余弦相似度：衡量两个向量方向的一致性，范围 [-1, 1]
      - 1 = 完全相同方向（最相似）
      - 0 = 正交（不相关）
      - -1 = 完全相反
    - 欧氏距离：两点间的直线距离，越小越相似
    - 实际项目中余弦相似度用得最多
    """
    print("\n" + "=" * 60)
    print("练习2：语义相似度 — 哪些文本'距离'更近？")
    print("=" * 60)

    # 基准文本
    query = "苹果怎么保存比较好"

    # 对比文本：有语义相近的，也有不相关的
    candidates = [
        "苹果的储存方法",
        "苹果的储存方法和注意事项",     # 语义相近
        "水果的储存方法和注意事项",     # 语义相近
        "苹果放冰箱冷藏能保鲜多久",     # 非常相近
        "如何延长蔬菜水果的保质期",      # 相关
        "苹果公司今年的股价走势如何",     # "苹果"歧义
        "今天的天气真不错",             # 完全不相关
        "香蕉应该怎样存放才不会变黑",    # 相关（不同水果，同类问题）
    ]

    # 计算所有文本的Embedding
    query_vec = embed_model.encode(query)
    candidate_vecs = embed_model.encode(candidates)

    print(f"\n基准文本: 「{query}」\n")
    print(f"{'对比文本':<25} {'余弦相似度':>10} {'欧氏距离':>10}")
    print("-" * 50)

    results = []
    for text, vec in zip(candidates, candidate_vecs):
        # 余弦相似度：手动计算，让你理解公式
        cos_sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
        # 欧氏距离
        euc_dist = np.linalg.norm(query_vec - vec)
        results.append((text, cos_sim, euc_dist))

    # 按余弦相似度排序
    results.sort(key=lambda x: x[1], reverse=True)

    for text, cos_sim, euc_dist in results:
        bar = "█" * int(cos_sim * 30)  # 简单的可视化
        print(f"「{text}」")
        print(f"  余弦相似度: {cos_sim:.4f} {bar}")
        print(f"  欧氏距离:   {euc_dist:.4f}")
        print()

    print("💡 观察要点：")
    print("  - 「苹果放冰箱冷藏能保鲜多久」是否排在最前面？")
    print("  - 「苹果公司的股价」和「苹果怎么保存」相似度如何？")
    print("    → Embedding模型能区分'苹果'的不同含义吗？")
    print("  - 「香蕉怎样存放」虽然说的是不同水果，相似度是否还挺高？")
    print("  - 这就是'语义搜索'的核心：不靠关键词匹配，靠意思接近")


# ========================================
# 练习3：使用 ChromaDB 向量数据库
# ========================================
def exercise_3_chromadb():
    """
    【目标】搭建一个最简单的向量数据库，体验"存"和"查"
    
    【考点】
    向量数据库的核心操作：
    1. 创建集合（类似传统数据库的"表"）
    2. 添加文档（自动计算Embedding并存储）
    3. 查询（自动计算查询Embedding → 找最近的K个 → 返回结果）
    
    ChromaDB 是最轻量的向量数据库，适合学习和小项目。
    生产环境可以用 Milvus（大规模）或 Pinecone（云服务）。
    """
    print("\n" + "=" * 60)
    print("练习3：ChromaDB 向量数据库实战")
    print("=" * 60)

    # --- 第1步：初始化ChromaDB ---
    # 内存模式，重启后数据会丢失。生产环境用 PersistentClient
    chroma_client = chromadb.Client()

    # 自定义Embedding函数，让ChromaDB使用我们的本地模型
    class LocalEmbeddingFunction:
        def __call__(self, input):
            return embed_model.encode(input).tolist()

        def embed_query(self, input):
            return self.__call__(input)

    # 创建一个集合（类似数据库里的"表"）
    collection = chroma_client.create_collection(
        name="food_safety",
        embedding_function=LocalEmbeddingFunction(),
    )
    print("✅ 创建集合: food_safety")

    # --- 第2步：添加文档 ---
    # 模拟一个食品安全知识库
    documents = [
        "新鲜鸡蛋在冰箱中可以保存30-45天，常温下只能保存10-15天。判断鸡蛋是否变质可以用水浮法：新鲜鸡蛋沉底，变质鸡蛋会浮起来。",
        "牛奶开封后应在2-3天内饮用完毕，未开封的可以保存到保质期。UHT牛奶（超高温灭菌）未开封可常温保存6个月。",
        "米饭煮熟后在常温下不应超过2小时，冰箱冷藏可保存1-2天。重新加热时必须彻底加热到75度以上，避免蜡样芽孢杆菌中毒。",
        "苹果在冰箱冷藏（0-4度）可以保存1-2个月。苹果会释放乙烯气体，应与其他水果分开存放，否则会加速其他水果成熟。",
        "三文鱼等生鱼片购买后应在当天食用，冷藏不超过24小时。冷冻可保存2-3个月，解冻后不可二次冷冻。",
        "剩菜剩饭应在烹饪后2小时内放入冰箱，冷藏保存不超过3天。叶菜类不建议二次加热，肉类和根茎类可以。",
        "蜂蜜是唯一不会变质的天然食品，但必须密封保存。结晶是正常现象，不影响品质。掺假蜂蜜可能含有糖浆。",
        "速冻食品应在-18度以下保存，解冻后应尽快食用。微波解冻最安全，常温解冻容易滋生细菌。",
        "豆腐开封后冷藏可保存3-5天，需每天换水。变质的豆腐会发黏、变黄、有异味。",
        "食用油开封后应在3个月内用完，存放在阴凉避光处。油脂氧化会产生有害物质，闻到哈喇味说明已经变质。",
    ]

    # 每个文档需要一个唯一ID
    ids = [f"doc_{i}" for i in range(len(documents))]

    # 还可以存元数据（metadata），方便后续过滤
    metadatas = [
        {"category": "蛋类", "risk_level": "中"},
        {"category": "乳制品", "risk_level": "高"},
        {"category": "主食", "risk_level": "高"},
        {"category": "水果", "risk_level": "低"},
        {"category": "海鲜", "risk_level": "高"},
        {"category": "剩菜", "risk_level": "高"},
        {"category": "调味品", "risk_level": "低"},
        {"category": "速冻食品", "risk_level": "中"},
        {"category": "豆制品", "risk_level": "中"},
        {"category": "调味品", "risk_level": "中"},
    ]

    # 添加到集合（ChromaDB会自动调用Embedding函数计算向量）
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"✅ 已添加 {len(documents)} 条文档到向量数据库\n")

    # --- 第3步：语义查询 ---
    print("--- 查询测试 ---\n")

    queries = [
        "鸡蛋放了很久还能不能吃",
        "生鱼片买回来怎么保存",
        "做多了饭菜吃不完怎么办",
        "油闻起来有点奇怪",
    ]

    for query in queries:
        print(f"🔍 查询: 「{query}」")

        results = collection.query(
            query_texts=[query],
            n_results=3,  # 返回最相关的3条
        )

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            # ChromaDB 返回的 distance 是 L2距离（越小越相似）
            print(f"  Top{i+1} [距离:{dist:.4f}] [{meta['category']}]")
            print(f"       {doc[:60]}...")

        print()

    # --- 第4步：带过滤的查询 ---
    print("--- 带元数据过滤的查询 ---\n")
    print("🔍 查询: 「什么食物容易变质」 + 过滤: risk_level=高")

    filtered_results = collection.query(
        query_texts=["什么食物容易变质需要注意"],
        n_results=3,
        where={"risk_level": "高"},  # 只在高风险食品中搜索
    )

    for i, (doc, meta) in enumerate(zip(
        filtered_results["documents"][0],
        filtered_results["metadatas"][0],
    )):
        print(f"  Top{i+1} [{meta['category']}] {doc[:60]}...")

    print("\n💡 观察要点：")
    print("  - 「鸡蛋放了很久还能不能吃」能匹配到鸡蛋保存的文档吗？")
    print("    用户的表述和文档的表述完全不同，但语义相近")
    print("  - 「油闻起来有点奇怪」能找到食用油变质的内容吗？")
    print("    这就是语义搜索的威力：用户不需要知道精确的关键词")
    print("  - metadata 过滤让你可以在语义搜索基础上加业务逻辑")
    print("  - 这10条文档就是一个最小版的 RAG 知识库！")


# ========================================
# 练习4：Embedding 的局限性探索
# ========================================
def exercise_4_limitations():
    """
    【目标】理解 Embedding 不是万能的，知道局限才能在中展现深度
    
    【加分点】
    - 不同Embedding模型效果差异很大
    - 跨语言、专业术语可能效果差
    - 长文本Embedding会丢失细节（信息被压缩到固定维度）
    - 这些局限是RAG需要"重排序（Rerank）"的原因之一
    """
    print("\n" + "=" * 60)
    print("练习4：Embedding 的局限性探索")
    print("=" * 60)

    # 测试1：否定语义
    print("\n--- 测试1：Embedding 能理解否定吗？ ---")
    texts = [
        "这个食物可以吃",
        "这个食物不可以吃",
        "这个食物能食用",
    ]
    vecs = embed_model.encode(texts)

    sim_pos = np.dot(vecs[0], vecs[2]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[2]))
    sim_neg = np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))

    print(f"「可以吃」 vs 「能食用」 相似度: {sim_pos:.4f} （应该很高）")
    print(f"「可以吃」 vs 「不可以吃」相似度: {sim_neg:.4f} （语义相反，但相似度可能也很高！）")
    print("  → 很多Embedding模型对否定词的处理不够好，这是已知局限")

    # 测试2：长文本 vs 短文本
    print("\n--- 测试2：文本长度对Embedding的影响 ---")
    short = "鸡蛋保存方法"
    long_text = (
        "鸡蛋是日常生活中最常见的食材之一。新鲜鸡蛋在冰箱冷藏环境下"
        "通常可以保存30到45天左右。在常温环境下，保存时间会大大缩短，"
        "一般只有10到15天。判断鸡蛋新鲜度可以使用水浮法：将鸡蛋放入水中，"
        "新鲜鸡蛋会沉入水底平躺，不太新鲜的会一端翘起，变质的会浮在水面上。"
        "此外，还可以通过摇晃鸡蛋听声音来判断：新鲜鸡蛋摇晃时没有声音，"
        "不新鲜的会有明显的晃荡感。保存鸡蛋时，建议大头朝上放置，因为气室"
        "在大头一端，这样可以减缓蛋黄上浮接触蛋壳的速度。"
    )
    query = "怎么保存鸡蛋"

    q_vec = embed_model.encode(query)
    short_vec = embed_model.encode(short)
    long_vec = embed_model.encode(long_text)

    sim_short = np.dot(q_vec, short_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(short_vec))
    sim_long = np.dot(q_vec, long_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(long_vec))

    print(f"查询: 「{query}」")
    print(f"vs 短文本 ({len(short)}字): 相似度 {sim_short:.4f}")
    print(f"vs 长文本 ({len(long_text)}字): 相似度 {sim_long:.4f}")
    print("  → 长文本包含更多信息，但Embedding压缩后可能丢失细节")
    print("  → 这就是为什么RAG需要把文档切成小块（chunking），而不是整篇做Embedding")

    # 测试3：同义词 vs 同字不同义
    print("\n--- 测试3：一词多义挑战 ---")
    pairs = [
        ("苹果很好吃", "苹果手机很好用", "同字不同义"),
        ("番茄炒蛋", "西红柿炒鸡蛋", "同义不同字"),
        ("冰箱里的牛奶过期了", "冰柜中的鲜奶到期了", "近义改写"),
    ]

    for text_a, text_b, desc in pairs:
        vec_a, vec_b = embed_model.encode([text_a, text_b])
        sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        print(f"「{text_a}」 vs 「{text_b}」")
        print(f"  [{desc}] 相似度: {sim:.4f}")

    print("\n💡 关键认识（加分项）：")
    print("  1. Embedding不是完美的，否定语义、一词多义都是挑战")
    print("  2. 这就是为什么RAG系统需要Rerank（重排序）来修正检索结果")
    print("  3. 选择合适的Embedding模型很重要（中文场景用BGE/M3E等中文模型）")
    print("  4. 文档切片（chunking）策略直接影响检索质量——后天会深入学")


# ========================================
# 🎯 运行入口
# ========================================
if __name__ == "__main__":
    print("🚀 AI工程师转型 - Day2 Embedding与向量数据库\n")

    exercises = {
        "1": ("文本→向量的直观感受", exercise_1_what_is_embedding),
        "2": ("语义相似度计算", exercise_2_similarity),
        "3": ("ChromaDB向量数据库实战", exercise_3_chromadb),
        "4": ("Embedding的局限性探索", exercise_4_limitations),
        "all": ("运行全部练习", None),
    }

    print("选择要运行的练习：")
    for key, (name, _) in exercises.items():
        print(f"  {key}. {name}")

    choice = input("\n请输入编号 (1/2/3/4/all): ").strip()

    if choice == "all":
        for key in ["1", "2", "3", "4"]:
            exercises[key][1]()
            print("\n" + "🔹" * 30 + "\n")
    elif choice in exercises and exercises[choice][1]:
        exercises[choice][1]()
    else:
        print("无效输入，请输入 1/2/3/4/all")

    print("\n\n" + "=" * 60)
    print("✅ Day2 练习完成！")
    print("=" * 60)
    print("""
📝 今日思考题（用自己的话写下来）：

1. Embedding 做了一件什么事？为什么它是 RAG 的基础？
2. 余弦相似度和欧氏距离有什么区别？项目中选哪个？
3. 向量数据库和 MySQL 这种传统数据库的本质区别是什么？
4. 你在练习4中发现了 Embedding 的哪些局限性？
   这些局限性在实际项目中怎么应对？（提示：Rerank、chunking）
5. 回顾你的食材管理项目：如果加一个"用户随便描述，就能找到相关食品安全知识"
   的功能，你现在知道怎么实现了吗？

🔗 和昨天的关联：
   昨天学的 LLM API 负责"生成回答"
   今天学的 Embedding + 向量数据库 负责"找到相关知识"
   明天把两者串起来 → 就是完整的 RAG 系统！

明天的内容：RAG核心流程 — 把检索和生成串成完整链路 🔗
""")
