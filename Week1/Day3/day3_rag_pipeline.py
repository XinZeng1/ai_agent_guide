"""
=============================================================
  AI工程师转型 - 第1周第3天：动手实践
  主题：RAG核心流程 — 从零搭建完整的检索增强生成系统
=============================================================

使用前准备：
  pip install openai chromadb sentence-transformers

今天的目标：
  不借助任何RAG框架（不用LangChain/LlamaIndex），用纯Python
  把RAG的每一步都手写一遍。这样你才能真正理解每个环节在做什么，
  面试时画出流程图讲解时才有底气。

  明天再用LlamaIndex框架重写，到时你就能体会框架帮你省了什么。

整体流程（面试必画的图）：

  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ 文档加载  │ →  │ 文本切片  │ →  │ Embedding │ →  │ 存入向量库 │
  └─────────┘    └──────────┘    └──────────┘    └──────────┘
                                                      ↓
  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ 生成回答  │ ←  │ 构造Prompt│ ←  │ 检索Top-K │ ←  │ 用户提问  │
  └─────────┘    └──────────┘    └──────────┘    └──────────┘
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

# ========================================
# 🔧 配置
# ========================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "你的API Key")

llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

print("✅ 模型加载完成\n")


# ========================================
# 练习1：文档加载与文本切片（Chunking）
# ========================================
def exercise_1_chunking():
    """
    【目标】理解为什么要切片，以及不同切片策略的影响

    【面试高频考点】
    Q：RAG中为什么要对文档做切片？
    A：三个原因 ——
      1. Embedding模型有输入长度限制（通常512 tokens）
      2. 长文本Embedding会丢失细节（昨天练习4验证过）
      3. 检索时需要返回"精确相关"的片段，整篇文档太粗糙

    Q：chunk_size 和 overlap 怎么设置？
    A：没有万能值，需要根据场景调：
      - chunk_size 太小 → 上下文不完整，检索到的片段没有足够信息
      - chunk_size 太大 → 不够精确，可能混入不相关内容
      - overlap 的作用 → 防止关键信息被切断在两个chunk的边界
      - 常见起步值：chunk_size=500字, overlap=50字
    """
    print("=" * 60)
    print("练习1：文本切片（Chunking）— RAG的第一步")
    print("=" * 60)

    # 模拟一篇较长的食品安全文档
    document = """食品保存完全指南

第一章：蛋类保存

鸡蛋是日常生活中最常见的食材之一。新鲜鸡蛋在冰箱冷藏环境（0-4摄氏度）下通常可以保存30到45天左右。在常温环境下（25度以上），保存时间会大大缩短，一般只有10到15天。鸡蛋表面有一层保护膜，清洗后保护膜被破坏，保质期会明显缩短，所以不建议提前清洗。

判断鸡蛋是否变质有几种方法。最简单的是水浮法：将鸡蛋轻轻放入水中，新鲜鸡蛋会沉入水底平躺，存放较久的会一端翘起，完全变质的会浮在水面上。也可以摇晃鸡蛋：新鲜的摇晃时无声，不新鲜的会有明显晃荡感。打开后，蛋黄完整挺立说明新鲜，蛋黄散开或有异味则不应食用。

存放鸡蛋的最佳方式是大头朝上放置，因为气室在大头一端，这样可以减缓蛋黄上浮接触蛋壳的速度。不建议放在冰箱门上，因为开关门的温度变化会影响保鲜效果，最好放在冰箱内部靠里的位置。

第二章：乳制品保存

牛奶是高营养但也容易变质的食品。巴氏消毒奶（鲜奶）开封后应在2-3天内饮用完毕，未开封的按包装上的保质期存放（通常7-15天，需冷藏）。UHT牛奶（超高温灭菌奶）未开封可在常温下保存6个月左右，但开封后同样需要冷藏并在3天内饮完。

酸奶在冰箱中通常可以保存到保质期后一周左右，但口感和活菌数量会逐渐下降。自制酸奶因为没有工业级密封，保存时间更短，建议3天内食用。

奶酪的保存时间差异很大：软质奶酪（如马苏里拉）开封后1周内食用；硬质奶酪（如帕玛森）可以保存数月。出现霉点的硬质奶酪可以切掉霉变部分继续食用，但软质奶酪出现霉变应整块丢弃。

第三章：肉类与海鲜保存

生肉（猪牛羊鸡）在冰箱冷藏（0-4度）可保存1-3天，冷冻（-18度以下）可保存3-6个月。绞肉因为接触空气面积大，变质更快，冷藏不超过1天。

海鲜类是所有食品中最容易变质的。新鲜鱼类冷藏不超过2天，虾类冷藏不超过1天。购买后应尽快处理或冷冻。解冻肉类和海鲜的最佳方式是放在冰箱冷藏室缓慢解冻（需要提前一晚），其次是流水解冻。微波解冻虽然快但容易导致受热不均。绝对不要在常温下解冻，这会让表面温度升高滋生细菌，而内部还是冻着的。

解冻后的肉类和海鲜不应二次冷冻，因为解冻过程中已经有细菌繁殖，再次冷冻只是让细菌"休眠"而非杀死它们，下次解冻后细菌数量会更多。"""

    print(f"原始文档长度: {len(document)} 字\n")

    # --- 切片方法1：固定长度切片 ---
    def chunk_by_size(text, chunk_size=200, overlap=50):
        """最基础的切片方法：按字符数切割，带重叠"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - overlap  # 下一个chunk从overlap处开始
        return [c for c in chunks if c]  # 去掉空chunk

    # --- 切片方法2：按段落切片 ---
    def chunk_by_paragraph(text, max_size=300):
        """
        按自然段落切片，更尊重文档结构。
        如果单个段落超过max_size，再做二次切割。
        """
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # 如果当前chunk加上这段不超过限制，就合并
            if len(current_chunk) + len(para) <= max_size:
                current_chunk += ("\n" + para if current_chunk else para)
            else:
                # 保存当前chunk，开始新的
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # 对比两种方法
    chunks_fixed = chunk_by_size(document, chunk_size=200, overlap=50)
    chunks_para = chunk_by_paragraph(document, max_size=300)

    print("--- 方法1：固定长度切片 (size=200, overlap=50) ---")
    print(f"切出 {len(chunks_fixed)} 个chunk\n")
    for i, chunk in enumerate(chunks_fixed[:3]):  # 只展示前3个
        print(f"  Chunk {i}: [{len(chunk)}字] {chunk[:80]}...")
        if i < 2:
            # 展示overlap部分
            overlap_text = chunks_fixed[i][-50:]
            next_start = chunks_fixed[i + 1][:50]
            if overlap_text.strip() and next_start.strip():
                print(f"         ↕ 重叠区域: 「...{overlap_text[-30:]}」")
    print()

    print("--- 方法2：按段落切片 (max_size=300) ---")
    print(f"切出 {len(chunks_para)} 个chunk\n")
    for i, chunk in enumerate(chunks_para[:3]):
        print(f"  Chunk {i}: [{len(chunk)}字] {chunk[:80]}...")
    print()

    print("💡 两种方法对比：")
    print("  固定长度：简单粗暴，可能把一句话切成两半")
    print("  按段落：  尊重文档结构，但如果段落很长还是需要二次切割")
    print("  实际项目中常用：RecursiveCharacterTextSplitter（LangChain提供）")
    print("  它会依次按 段落→句子→字符 来切割，兼顾结构和长度\n")

    # 返回切片结果供后续练习使用
    return chunks_para, document


# ========================================
# 练习2：构建索引（Embedding → 向量库）
# ========================================
def exercise_2_build_index(chunks):
    """
    【目标】把切片后的文档存入向量数据库，构建检索索引

    【面试考点】
    索引构建 = 切片 + Embedding + 存储
    这个过程是离线做的（一次性），查询时不需要重复
    """
    print("=" * 60)
    print("练习2：构建向量索引")
    print("=" * 60)

    chroma_client = chromadb.Client()

    class LocalEmbeddingFunction:
        def __call__(self, input):
            return embed_model.encode(input).tolist()

        def embed_query(self, input):
            return self.__call__(input)

    # 如果集合已存在，先删除
    try:
        chroma_client.delete_collection("food_safety_rag")
    except:
        pass

    collection = chroma_client.create_collection(
        name="food_safety_rag",
        embedding_function=LocalEmbeddingFunction(),
    )

    # 存入所有chunk
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

    print(f"✅ 已将 {len(chunks)} 个chunk存入向量数据库")
    print(f"   每个chunk会被自动转为 {embed_model.get_sentence_embedding_dimension()} 维向量")

    # 快速验证
    test_query = "鸡蛋怎么判断有没有坏"
    results = collection.query(query_texts=[test_query], n_results=2)

    print(f"\n验证查询: 「{test_query}」")
    for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
        print(f"  Top{i+1} [距离:{dist:.4f}] {doc[:60]}...")

    print("\n💡 这一步完成后，RAG的索引侧就搭好了")
    print("   接下来是查询侧：用户提问 → 检索 → 生成")

    return collection


# ========================================
# 练习3：完整RAG流程（核心！）
# ========================================
def exercise_3_full_rag(collection):
    """
    【目标】串联检索和生成，跑通完整的RAG

    【这是面试最常被要求手写/讲解的内容】
    完整流程：
    1. 接收用户问题
    2. 用Embedding检索向量库，拿到Top-K相关文档片段
    3. 把检索到的片段拼接到Prompt中（作为"上下文"）
    4. 调用LLM，让它基于上下文回答问题
    5. 返回答案
    """
    print("\n" + "=" * 60)
    print("练习3：完整的RAG流程 🔥（今天最重要的练习）")
    print("=" * 60)

    def rag_query(question, top_k=3, show_process=True):
        """
        一个完整的RAG查询函数。
        面试时你需要能手写出这个函数的核心逻辑。
        """

        # === Step 1: 检索 ===
        results = collection.query(
            query_texts=[question],
            n_results=top_k,
        )

        retrieved_docs = results["documents"][0]
        distances = results["distances"][0]

        if show_process:
            print(f"\n📌 Step 1 - 检索到 {len(retrieved_docs)} 个相关片段：")
            for i, (doc, dist) in enumerate(zip(retrieved_docs, distances)):
                print(f"   [{i+1}] 距离={dist:.4f} | {doc[:50]}...")

        # === Step 2: 构造Prompt ===
        # 这个Prompt模板是RAG的核心 —— 把检索到的文档作为"上下文"
        context = "\n\n---\n\n".join(retrieved_docs)

        prompt = f"""你是一个专业的食品安全顾问。请根据以下参考资料回答用户的问题。

要求：
1. 只根据参考资料中的信息回答，不要编造
2. 如果参考资料中没有相关信息，请明确告知用户
3. 回答要简洁实用，给出具体的建议

参考资料：
{context}

用户问题：{question}

请回答："""

        if show_process:
            print(f"\n📌 Step 2 - 构造Prompt（总长度 {len(prompt)} 字）")
            print(f"   上下文部分占 {len(context)} 字")

        # === Step 3: 调用LLM生成回答 ===
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # RAG场景用低temperature，保证回答基于上下文
            max_tokens=500,
        )

        answer = response.choices[0].message.content
        usage = response.usage

        if show_process:
            print(f"\n📌 Step 3 - LLM生成回答")
            print(f"   Token消耗: 输入={usage.prompt_tokens}, 输出={usage.completion_tokens}")

        return answer

    # --- 测试几个问题 ---
    test_questions = [
        "鸡蛋买回来怎么放才能保存最久？",
        "剩下的牛奶明天还能喝吗？",
        "冷冻的虾解冻后能不能再放回冰箱冻起来？",
        "奶酪上长了一点霉还能吃吗？",
    ]

    for q in test_questions:
        print(f"\n{'='*50}")
        print(f"👤 用户问题: {q}")
        print(f"{'='*50}")
        answer = rag_query(q)
        print(f"\n🤖 RAG回答:\n{answer}")

    print("\n\n💡 观察要点：")
    print("  - 回答是否确实基于检索到的文档内容？（忠实度）")
    print("  - 如果问一个文档中没有的问题，模型会怎么回应？")
    print("  - 注意token消耗：上下文越长，input tokens越多 → 成本越高")

    return rag_query


# ========================================
# 练习4：RAG vs 直接问LLM 的对比
# ========================================
def exercise_4_rag_vs_llm(rag_query_fn):
    """
    【目标】直观对比有RAG和没RAG的回答质量差异

    【面试考点】
    Q：RAG的作用是什么？
    A：解决LLM的三个核心问题 ——
      1. 知识截止：LLM训练数据有截止日期，RAG能注入最新知识
      2. 幻觉问题：LLM可能编造信息，RAG让它基于真实文档回答
      3. 领域知识：通用LLM不了解你的私有数据，RAG能接入企业知识库

    Q：RAG和微调（Fine-tuning）的区别？
    A：
      - RAG：不改模型，改输入（把知识塞进Prompt）→ 灵活、实时更新
      - 微调：改模型参数 → 效果可能更好，但成本高、更新不便
      - 实际项目中通常先试RAG，效果不够再考虑微调
    """
    print("\n" + "=" * 60)
    print("练习4：RAG vs 直接问LLM — 谁回答得更好？")
    print("=" * 60)

    comparison_questions = [
        "鸡蛋放冰箱最好大头朝上还是朝下？为什么？",
        "解冻后的海鲜为什么不能二次冷冻？",
        "巴氏奶和UHT奶保存方式有什么不同？",
    ]

    for q in comparison_questions:
        print(f"\n{'─'*50}")
        print(f"❓ 问题: {q}")
        print(f"{'─'*50}")

        # 方式A：直接问LLM（无RAG）
        direct_response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是食品安全顾问，简洁回答问题。"},
                {"role": "user", "content": q}
            ],
            temperature=0,
            max_tokens=300,
        )
        direct_answer = direct_response.choices[0].message.content

        # 方式B：RAG
        rag_answer = rag_query_fn(q, show_process=False)

        print(f"\n🅰️ 直接问LLM（无RAG）:")
        print(f"   {direct_answer[:200]}...")

        print(f"\n🅱️ RAG回答（基于知识库）:")
        print(f"   {rag_answer[:200]}...")

    print("\n💡 对比分析：")
    print("  - 对于通用知识（鸡蛋保存），两者可能差不多")
    print("  - 但RAG的回答能溯源到具体文档，更可信")
    print("  - 如果知识库里有你公司特有的食品标准，直接问LLM就完全答不上了")
    print("  - 这就是RAG的核心价值：让LLM能回答它原本不知道的事情")


# ========================================
# 练习5：Prompt模板的影响
# ========================================
def exercise_5_prompt_engineering(collection):
    """
    【目标】体会不同的Prompt模板对RAG输出质量的影响

    【面试加分点】
    RAG中的Prompt Engineering不只是"把文档塞进去"那么简单。
    Prompt模板的设计直接影响：
    - 回答是否忠实于文档（不编造）
    - 回答的格式和详细程度
    - 模型是否会说"我不知道"（而不是硬编）
    """
    print("\n" + "=" * 60)
    print("练习5：Prompt模板对RAG质量的影响")
    print("=" * 60)

    question = "绞肉和普通肉块在保存上有什么区别？"

    # 先检索
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n\n".join(results["documents"][0])

    # 模板A：简单粗暴
    prompt_a = f"根据以下内容回答问题。\n{context}\n\n问题：{question}"

    # 模板B：详细指令
    prompt_b = f"""你是一位食品安全专家。请严格根据以下参考资料回答用户的问题。

规则：
1. 只使用参考资料中的信息，不要添加额外知识
2. 如果资料中没有相关信息，回答"根据现有资料无法回答此问题"
3. 用要点形式回答，每个要点包含具体的数字和建议
4. 在回答末尾标注信息来源（引用了哪段资料）

参考资料：
{context}

用户问题：{question}"""

    # 模板C：角色扮演 + 结构化输出
    prompt_c = f"""你是一位在超市工作20年的食品安全检验员。一位顾客问了你以下问题，
请用通俗易懂的方式回答，就像面对面跟顾客解释一样。

回答格式：
【简短结论】一句话总结
【详细解释】为什么会这样
【实用建议】具体怎么做

参考知识（基于你的专业知识）：
{context}

顾客问题：{question}"""

    templates = [
        ("A: 简单模板", prompt_a),
        ("B: 详细指令模板", prompt_b),
        ("C: 角色扮演+结构化模板", prompt_c),
    ]

    for name, prompt in templates:
        print(f"\n--- {name} ---")
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
        )
        print(response.choices[0].message.content)

    print("\n💡 观察要点：")
    print("  - 模板A的回答是否可能包含文档中没有的信息？（幻觉风险）")
    print("  - 模板B的「不要添加额外知识」指令是否有效？")
    print("  - 模板C的结构化格式是否让回答更清晰？")
    print("  - 哪个模板最适合你的食材管理项目？为什么？")
    print("  - Prompt Engineering 是 RAG 质量的重要一环，别只关注检索")


# ========================================
# 🎯 运行入口
# ========================================
if __name__ == "__main__":
    print("🚀 AI工程师转型 - Day3 RAG完整流程\n")

    exercises = {
        "1": "文本切片（Chunking）",
        "2": "构建向量索引",
        "3": "完整RAG流程 🔥",
        "4": "RAG vs 直接问LLM",
        "5": "Prompt模板对比",
        "all": "运行全部练习",
    }

    print("选择要运行的练习：")
    for key, name in exercises.items():
        print(f"  {key}. {name}")

    choice = input("\n请输入编号 (1/2/3/4/5/all): ").strip()

    # 练习之间有依赖关系，所以需要按顺序准备数据
    chunks = None
    collection = None
    rag_fn = None

    def ensure_chunks():
        global chunks
        if chunks is None:
            chunks, _ = exercise_1_chunking()
        return chunks

    def ensure_collection():
        global collection
        if collection is None:
            ensure_chunks()
            collection = exercise_2_build_index(chunks)
        return collection

    def ensure_rag():
        global rag_fn
        if rag_fn is None:
            ensure_collection()
            rag_fn = exercise_3_full_rag(collection)
        return rag_fn

    if choice == "1":
        exercise_1_chunking()
    elif choice == "2":
        ensure_chunks()
        exercise_2_build_index(chunks)
    elif choice == "3":
        ensure_collection()
        exercise_3_full_rag(collection)
    elif choice == "4":
        ensure_rag()
        exercise_4_rag_vs_llm(rag_fn)
    elif choice == "5":
        ensure_collection()
        exercise_5_prompt_engineering(collection)
    elif choice == "all":
        chunks, _ = exercise_1_chunking()
        print("\n" + "🔹" * 30 + "\n")
        collection = exercise_2_build_index(chunks)
        print("\n" + "🔹" * 30 + "\n")
        rag_fn = exercise_3_full_rag(collection)
        print("\n" + "🔹" * 30 + "\n")
        exercise_4_rag_vs_llm(rag_fn)
        print("\n" + "🔹" * 30 + "\n")
        exercise_5_prompt_engineering(collection)
    else:
        print("无效输入")

    print("\n\n" + "=" * 60)
    print("✅ Day3 练习完成！")
    print("=" * 60)
    print("""
📝 今日思考题（这些是面试必考题，用自己的话写下来）：

1. 完整描述RAG的流程：从用户提问到返回答案，经过了哪些步骤？
   （提示：能不能不看代码，在纸上画出流程图？）

2. 为什么要做文本切片？chunk_size太大和太小分别有什么问题？

3. RAG的Prompt模板中，为什么要加"如果资料中没有相关信息请明确告知"
   这样的指令？如果不加会怎样？

4. RAG和直接问LLM相比，优势是什么？劣势呢？
   （提示：延迟、成本、准确性、实时性）

5. RAG和Fine-tuning的区别是什么？什么时候该用哪个？

🏗️ 项目关联：
   你今天手写的这个RAG系统，就是你食材管理项目的"知识引擎"。
   之前你可能只是把用户问题直接扔给DeepSeek，现在你知道怎么
   让它基于你自己的知识库回答了。

   周末的综合实践，就是把这个RAG系统接入你的食材管理项目。

明天的内容：RAG进阶 — 重排序（Rerank）与混合检索 🔍
  → 解决今天RAG系统中"检索不够精确"的问题
""")
