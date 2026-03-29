"""
=============================================================
  AI工程师转型 - 第1周第5天：动手实践
  主题：LangChain框架入门 — 用框架重写RAG
=============================================================

使用前准备：
  pip install langchain langchain-openai langchain-community langchain-chroma
  pip install chromadb sentence-transformers

今天的目标：
  前两天你手写了完整的RAG流水线，理解了每一步的原理。
  今天用LangChain框架重写，体会：
  1. 框架帮你省了什么（封装、抽象、组合）
  2. 框架没帮你省什么（Prompt设计、切片策略、评估）
  3. LCEL（LangChain Expression Language）的管道式写法

  同时理解 LangChain vs LlamaIndex 的定位差异：
  - LangChain：通用编排框架，擅长把各种组件串起来
  - LlamaIndex：专注数据索引和RAG，在RAG场景下更开箱即用
"""

import os
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# ========================================
# 🔧 配置
# ========================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "你的API Key")

# LangChain中使用DeepSeek：它兼容OpenAI接口，所以直接用ChatOpenAI
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0,
)

# Embedding模型：LangChain封装了HuggingFace的模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
)

print("✅ 模型加载完成\n")

# ========================================
# 原始文档（和前几天一样的食品安全知识）
# ========================================
RAW_DOCUMENT = """食品保存完全指南

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

解冻后的肉类和海鲜不应二次冷冻，因为解冻过程中已经有细菌繁殖，再次冷冻只是让细菌休眠而非杀死它们，下次解冻后细菌数量会更多。

第四章：其他常见食品

米饭煮熟后在常温下不应超过2小时，冰箱冷藏可保存1-2天。重新加热时必须彻底加热到75度以上，避免蜡样芽孢杆菌中毒。

剩菜剩饭应在烹饪后2小时内放入冰箱，冷藏保存不超过3天。叶菜类不建议二次加热，肉类和根茎类可以。

苹果在冰箱冷藏（0-4度）可以保存1-2个月。苹果会释放乙烯气体，应与其他水果分开存放，否则会加速其他水果成熟变质。

香蕉不适合放冰箱，低温会让香蕉皮变黑。最佳保存方式是挂起来通风存放，避免挤压。已经变黑的香蕉果肉如果没有异味仍可食用。

食用油开封后应在3个月内用完，存放在阴凉避光处。油脂氧化会产生有害物质，闻到哈喇味说明已经变质，不应继续食用。"""


# ========================================
# 练习1：LangChain文本切片
# ========================================
def exercise_1_text_splitting():
    """
    【目标】学会使用LangChain的RecursiveCharacterTextSplitter

    【面试考点】
    RecursiveCharacterTextSplitter的工作方式：
    它会依次尝试按以下分隔符切割：
      "\n\n" → "\n" → " " → ""
    优先按段落切，段落太长就按句子切，再不行就按字符切。
    这比固定长度切片更"聪明"，因为它尽量保持语义完整性。
    """
    print("=" * 60)
    print("练习1：LangChain 文本切片")
    print("=" * 60)

    # --- 你之前手写的方式 vs LangChain封装 ---

    # LangChain的递归字符切片器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,       # 每个chunk最大300字符
        chunk_overlap=50,     # 相邻chunk重叠50字符
        separators=["\n\n", "\n", "。", "，", " ", ""],  # 中文友好的分隔符
        length_function=len,  # 用字符数计算长度
    )

    # 切片
    chunks = splitter.split_text(RAW_DOCUMENT)

    print(f"原始文档: {len(RAW_DOCUMENT)} 字")
    print(f"切片结果: {len(chunks)} 个chunk\n")

    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: [{len(chunk)}字]")
        print(f"    开头: {chunk[:50]}...")
        print(f"    结尾: ...{chunk[-50:]}")
        print()
        if i >= 4:
            print(f"  ... 还有 {len(chunks) - 5} 个chunk")
            break

    # 也可以直接切成Document对象（带metadata）
    docs = splitter.create_documents(
        texts=[RAW_DOCUMENT],
        metadatas=[{"source": "食品保存指南", "type": "reference"}],
    )
    print(f"\n作为Document对象: {len(docs)} 个")
    print(f"  第一个Document的metadata: {docs[0].metadata}")
    print(f"  第一个Document的内容前50字: {docs[0].page_content[:50]}...")

    print("\n💡 对比你Day3手写的切片方法：")
    print("  - RecursiveCharacterTextSplitter 会优先按段落和句子切割")
    print("  - 你不用自己处理边界问题，框架帮你做了")
    print("  - 但你需要针对中文设置合适的 separators（加入中文标点）")
    print("  - chunk_size 和 overlap 仍然需要你根据业务场景调整")

    return docs


# ========================================
# 练习2：用LangChain构建向量索引
# ========================================
def exercise_2_vector_store(docs):
    """
    【目标】用LangChain + ChromaDB快速构建向量索引

    【面试考点】
    LangChain封装了向量数据库的操作：
    - from_documents()：一行代码完成 切片→Embedding→存储
    - as_retriever()：把向量库转成Retriever对象，方便接入Chain
    """
    print("\n" + "=" * 60)
    print("练习2：LangChain + ChromaDB 构建索引")
    print("=" * 60)

    # 一行代码搞定：文档 → Embedding → 存入ChromaDB
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="food_safety_lc",
    )

    print(f"✅ 向量索引构建完成，包含 {len(docs)} 个文档\n")

    # 转为Retriever（这是LangChain的核心抽象之一）
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # 也可以用 "mmr"（最大边际相关性）
        search_kwargs={"k": 3},
    )

    # 测试检索
    print("--- 测试检索 ---\n")
    test_queries = ["鸡蛋怎么保存", "海鲜能不能二次冷冻"]

    for query in test_queries:
        print(f"🔍 查询: 「{query}」")
        results = retriever.invoke(query)
        for i, doc in enumerate(results):
            print(f"   Top{i+1}: {doc.page_content[:50]}...")
        print()

    # --- MMR检索：增加结果多样性 ---
    print("--- MMR检索（增加多样性） ---\n")
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10},  # 先取10个再用MMR挑3个
    )

    query = "食物保存注意事项"
    print(f"🔍 查询: 「{query}」")
    print(f"\n  普通检索 Top-3:")
    for i, doc in enumerate(retriever.invoke(query)):
        print(f"   [{i+1}] {doc.page_content[:50]}...")

    print(f"\n  MMR检索 Top-3（更多样）:")
    for i, doc in enumerate(mmr_retriever.invoke(query)):
        print(f"   [{i+1}] {doc.page_content[:50]}...")

    print("\n💡 MMR（Maximal Marginal Relevance）的作用：")
    print("  普通检索可能返回3段都是关于鸡蛋的内容（高相似但重复）")
    print("  MMR会在相关性和多样性之间平衡，避免结果过于集中")
    print("  这在RAG中很有用：你希望给LLM提供多角度的参考信息")

    return vectorstore, retriever


# ========================================
# 练习3：LCEL链式表达（今天最重要的概念）
# ========================================
def exercise_3_lcel_chain(retriever):
    """
    【目标】理解LCEL（LangChain Expression Language）的管道式写法

    【面试考点】
    LCEL是LangChain的核心语法：
    - 用管道符 | 把组件串联：prompt | llm | output_parser
    - 每个组件实现 invoke() 方法
    - 前一个组件的输出自动传给下一个组件的输入
    - 类似Unix管道：cat file | grep "error" | wc -l

    这不只是语法糖，它让你能：
    - 轻松切换组件（换LLM、换检索器）
    - 自动支持流式输出、批量处理、异步调用
    - 方便调试（每个环节的输入输出都可追踪）
    """
    print("\n" + "=" * 60)
    print("练习3：LCEL — LangChain的管道式写法 🔥")
    print("=" * 60)

    # --- Step 1：最简单的Chain ---
    print("--- 3.1 最简单的Chain: prompt | llm | parser ---\n")

    simple_prompt = ChatPromptTemplate.from_template("用一句话解释：{topic}")
    parser = StrOutputParser()  # 把ChatMessage转为纯字符串

    # LCEL管道：prompt → llm → parser
    simple_chain = simple_prompt | llm | parser

    # 调用
    result = simple_chain.invoke({"topic": "什么是RAG"})
    print(f"  输入: topic='什么是RAG'")
    print(f"  输出: {result}\n")

    # --- Step 2：RAG Chain ---
    print("--- 3.2 RAG Chain: retriever → prompt → llm → parser ---\n")

    # RAG专用Prompt模板
    rag_prompt = ChatPromptTemplate.from_template("""你是一位专业的食品安全顾问。请根据以下参考资料回答问题。

要求：
1. 只根据参考资料回答，不要编造
2. 如果资料中没有相关信息，明确告知
3. 回答简洁实用

参考资料：
{context}

问题：{question}

回答：""")

    def format_docs(docs):
        """把检索到的Document列表格式化为字符串"""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # RAG Chain（这就是LCEL的精髓）
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | parser
    )

    # 测试
    test_questions = [
        "鸡蛋放冰箱门上可以吗？",
        "酸奶过期一周了还能喝吗？",
        "冻虾解冻后吃不完怎么办？",
    ]

    for q in test_questions:
        print(f"  👤 {q}")
        answer = rag_chain.invoke(q)
        print(f"  🤖 {answer}\n")

    print("💡 LCEL的价值：")
    print("  对比你Day3手写的rag_query函数，LCEL版本：")
    print("  - 代码更简洁（5行 vs 30行）")
    print("  - 组件可替换（换个retriever或llm只需改一个变量）")
    print("  - 自动支持 .stream()（流式输出）和 .batch()（批量处理）")
    print("  - 但底层逻辑和你手写的完全一样：检索→拼Prompt→生成")

    return rag_chain


# ========================================
# 练习4：带记忆的RAG对话
# ========================================
def exercise_4_conversational_rag(retriever):
    """
    【目标】实现多轮对话的RAG（带上下文记忆）

    【面试考点】
    单轮RAG的问题：用户说"那个呢"，系统不知道"那个"是什么。
    需要把对话历史也考虑进去：
    1. 把历史对话 + 当前问题 → 改写为独立的检索查询
    2. 用改写后的查询去检索
    3. 把检索结果 + 历史对话 + 当前问题 → 生成回答

    这个"查询改写"步骤叫做 Contextualization（上下文化）
    """
    print("\n" + "=" * 60)
    print("练习4：多轮对话RAG（带记忆）")
    print("=" * 60)

    # Step 1：查询改写链
    # 把带上下文的问题改写为独立的检索查询
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "根据对话历史，将用户最新的问题改写为一个独立的、不依赖上下文的检索查询。只输出改写后的查询，不要其他内容。"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    # Step 2：RAG回答链
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位专业的食品安全顾问。根据参考资料和对话历史回答用户问题。
只根据参考资料回答，不要编造。如果不知道，明确说明。"""),
        MessagesPlaceholder("chat_history"),
        ("human", """参考资料：
{context}

问题：{input}"""),
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # 完整的对话RAG函数
    def conversational_rag(question, chat_history):
        """
        支持多轮对话的RAG。

        关键步骤：
        1. 如果有历史对话 → 先改写问题
        2. 用改写后的问题检索
        3. 基于检索结果 + 历史 + 当前问题生成回答
        """
        # Step 1：改写问题（如果有对话历史）
        if chat_history:
            standalone_query = contextualize_chain.invoke({
                "chat_history": chat_history,
                "input": question,
            })
            print(f"  🔄 改写后的查询: 「{standalone_query}」")
        else:
            standalone_query = question

        # Step 2：检索
        retrieved_docs = retriever.invoke(standalone_query)
        context = format_docs(retrieved_docs)

        # Step 3：生成回答
        answer_chain = answer_prompt | llm | StrOutputParser()
        answer = answer_chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "input": question,
        })

        return answer

    # 模拟多轮对话
    print("\n--- 模拟多轮对话 ---\n")

    chat_history = []

    conversations = [
        "鸡蛋在冰箱里能放多久？",
        "那如果没有冰箱呢？",           # 需要理解"那"指鸡蛋
        "怎么判断它有没有坏？",          # 需要理解"它"指鸡蛋
        "牛奶呢？开封后能放几天？",       # 话题切换到牛奶
        "如果是自制酸奶呢？",            # 需要理解上下文是乳制品
    ]

    for question in conversations:
        print(f"👤 用户: {question}")
        answer = conversational_rag(question, chat_history)
        print(f"🤖 助手: {answer}\n")

        # 更新对话历史
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

    print("💡 关键理解：")
    print("  - 「那如果没有冰箱呢」→ 改写为「鸡蛋在常温下能保存多久」")
    print("  - 「怎么判断它有没有坏」→ 改写为「如何判断鸡蛋是否变质」")
    print("  - 没有查询改写，这些追问根本检索不到正确文档")
    print("  - 这就是为什么生产级RAG都需要 Contextualization 步骤")
    print("  - 对话历史越长，token消耗越大 → 需要做历史压缩/摘要")


# ========================================
# 练习5：LangChain vs 手写 — 全面对比
# ========================================
def exercise_5_comparison():
    """
    【目标】理解框架的价值和边界

    【面试加分点】
    不要只说"我用了LangChain"，要能说清楚：
    - 框架帮你做了什么（组件封装、接口统一、生态集成）
    - 框架没帮你做什么（Prompt设计、业务逻辑、效果调优）
    - 什么时候用框架，什么时候手写
    """
    print("\n" + "=" * 60)
    print("练习5：LangChain vs 手写 — 什么时候该用框架？")
    print("=" * 60)

    comparison = [
        {
            "aspect": "文本切片",
            "handwritten": "自己写循环，处理边界和overlap",
            "langchain": "RecursiveCharacterTextSplitter一行搞定",
            "verdict": "框架省力，但切片策略仍需你自己设计",
        },
        {
            "aspect": "向量存储",
            "handwritten": "手动调用Embedding → 手动存入ChromaDB",
            "langchain": "Chroma.from_documents() 一行搞定",
            "verdict": "框架封装了重复代码",
        },
        {
            "aspect": "RAG链路",
            "handwritten": "写函数：检索→拼Prompt→调LLM",
            "langchain": "LCEL: retriever | prompt | llm | parser",
            "verdict": "LCEL更简洁，但逻辑一样",
        },
        {
            "aspect": "多轮对话",
            "handwritten": "手动管理对话列表，手动拼历史",
            "langchain": "MessagesPlaceholder + 查询改写链",
            "verdict": "框架提供了更规范的模式",
        },
        {
            "aspect": "换LLM/向量库",
            "handwritten": "改代码中的API调用",
            "langchain": "改一个变量（ChatOpenAI → ChatAnthropic）",
            "verdict": "框架的最大价值：接口统一，组件可替换",
        },
        {
            "aspect": "Prompt设计",
            "handwritten": "自己写字符串模板",
            "langchain": "ChatPromptTemplate（本质还是你自己写）",
            "verdict": "框架帮不了你，Prompt设计是手艺活",
        },
        {
            "aspect": "调试排查",
            "handwritten": "print大法",
            "langchain": "LangSmith可视化追踪",
            "verdict": "框架生态的优势",
        },
        {
            "aspect": "性能优化",
            "handwritten": "自己控制每个环节",
            "langchain": "框架有抽象开销，部分场景性能略差",
            "verdict": "极致性能场景可能需要手写",
        },
    ]

    print(f"\n{'方面':<12} {'手写':^28} {'LangChain':^28} {'结论'}")
    print("─" * 100)
    for item in comparison:
        print(f"\n📋 {item['aspect']}")
        print(f"   手写:     {item['handwritten']}")
        print(f"   框架:     {item['langchain']}")
        print(f"   → {item['verdict']}")

    print(f"""
\n💡 什么时候用框架，什么时候手写？

  用LangChain：
    ✅ 快速原型开发（先跑通再优化）
    ✅ 需要对接多种LLM/向量库/工具的项目
    ✅ 团队协作（统一的抽象降低沟通成本）

  手写：
    ✅ 面试！你必须能脱离框架讲清楚原理
    ✅ 对性能有极致要求的生产环境
    ✅ 框架不支持的定制化需求

  最佳实践：
    先手写理解原理 → 再用框架提效 → 需要时回到底层优化
    （你这几天的学习路径就是这样设计的）

  面试话术：
    "我先不用框架手写了完整的RAG流水线来理解每个环节，
     然后用LangChain重构来提高开发效率。框架帮我统一了
     组件接口，但Prompt设计和检索策略这些核心逻辑还是
     需要自己把握。"
""")


# ========================================
# 🎯 运行入口
# ========================================
if __name__ == "__main__":
    print("🚀 AI工程师转型 - Day5 LangChain框架\n")

    exercises = {
        "1": "LangChain文本切片",
        "2": "向量索引构建 + MMR检索",
        "3": "LCEL链式表达（RAG Chain）🔥",
        "4": "多轮对话RAG（带记忆）",
        "5": "LangChain vs 手写 对比总结",
        "all": "运行全部练习",
    }

    print("选择要运行的练习：")
    for key, name in exercises.items():
        print(f"  {key}. {name}")

    choice = input("\n请输入编号 (1/2/3/4/5/all): ").strip()

    docs = None
    vectorstore = None
    retriever = None
    rag_chain = None

    def ensure_docs():
        nonlocal docs
        if docs is None:
            docs = exercise_1_text_splitting()
        return docs

    def ensure_retriever():
        nonlocal vectorstore, retriever
        if retriever is None:
            ensure_docs()
            vectorstore, retriever = exercise_2_vector_store(docs)
        return retriever

    if choice == "1":
        exercise_1_text_splitting()
    elif choice == "2":
        ensure_docs()
        exercise_2_vector_store(docs)
    elif choice == "3":
        ensure_retriever()
        exercise_3_lcel_chain(retriever)
    elif choice == "4":
        ensure_retriever()
        exercise_4_conversational_rag(retriever)
    elif choice == "5":
        exercise_5_comparison()
    elif choice == "all":
        docs = exercise_1_text_splitting()
        print("\n" + "🔹" * 30 + "\n")
        vectorstore, retriever = exercise_2_vector_store(docs)
        print("\n" + "🔹" * 30 + "\n")
        rag_chain = exercise_3_lcel_chain(retriever)
        print("\n" + "🔹" * 30 + "\n")
        exercise_4_conversational_rag(retriever)
        print("\n" + "🔹" * 30 + "\n")
        exercise_5_comparison()

    print("\n\n" + "=" * 60)
    print("✅ Day5 练习完成！第一周工作日学习完成！🎉")
    print("=" * 60)
    print("""
📝 今日思考题：

1. LCEL（管道式写法）的核心思想是什么？
   和你平时写Python函数调用有什么区别？

2. LangChain中的Retriever是什么？
   similarity检索和MMR检索有什么区别？什么场景用MMR？

3. 多轮对话RAG中"查询改写"这步为什么必不可少？
   如果不做改写，用户说"那个呢"会发生什么？

4. LangChain帮你省了什么？没帮你省什么？
   （面试最爱问：你为什么选LangChain？它的优缺点？）

5. 现在回顾你的食材管理项目：
   如果用LangChain重构食品安全问答功能，
   哪些部分会更简洁？哪些部分仍然需要你自己设计？

🏗️ 第一周总结：
  Day1: LLM API基础 → 理解"生成"
  Day2: Embedding + 向量库 → 理解"检索"
  Day3: 手写RAG → 把两者串起来
  Day4: 混合检索 + Rerank → 优化检索质量
  Day5: LangChain框架 → 用工具提效

  你现在已经能手写RAG、也能用框架快速搭建RAG了。
  周末的任务：把这些整合到你的食材管理项目中！

🗓️ 周末综合实践预告：
  把食材管理项目的问答功能升级为完整的RAG系统：
  1. 收集食品安全文档作为知识库
  2. 实现完整的RAG链路（切片→索引→混合检索→Rerank→生成）
  3. 支持多轮对话
  4. 准备一份架构图，练习用3分钟讲清楚整个系统
""")
