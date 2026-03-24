"""
=============================================================
  AI工程师转型 - 第1周第1天：动手实践
  主题：理解LLM API的核心参数与多轮对话
  API：DeepSeek（兼容OpenAI SDK）
=============================================================

使用前准备：
1. gu
2. 设置你的DeepSeek API Key（去 platform.deepseek.com 获取）

运行方式：
  python day1_llm_basics.py

代码分为4个练习，建议按顺序做，每个练习做完想一想"为什么"。
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# ========================================
# 🔧 配置区 - 把你的API Key填在这里
# ========================================
API_KEY = os.getenv("DEEPSEEK_API_KEY", "你的API Key")
MODEL = "deepseek-chat"  # DeepSeek-V3


def _to_messages(dicts):
    """dict 列表 → LangChain 消息对象"""
    _MAP = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    return [_MAP[m["role"]](content=m["content"]) for m in dicts]


def _create_llm(temperature=1.0, max_tokens=1024):
    """创建一个 ChatOpenAI 实例（DeepSeek 兼容 OpenAI 接口）"""
    return ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url="https://api.deepseek.com",
        max_tokens=max_tokens,
        temperature=temperature,
    )


def call_llm(messages, temperature=1.0, max_tokens=1024):
    """
    流式调用 LLM，token 逐个实时打印。

    参数：
    - messages: 对话历史，每条消息有 role 和 content
    - temperature: 控制随机性，0=确定性输出，2=非常随机
    - max_tokens: 限制回复长度（token ≈ 1-2个中文字）
    """
    model = _create_llm(temperature=temperature, max_tokens=max_tokens)
    full_text = ""
    for chunk in model.stream(_to_messages(messages)):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_text += chunk.content
    print()
    return full_text


# ========================================
# 练习1：理解 temperature 参数
# ========================================
def exercise_1_temperature():
    """
    【目标】直观感受 temperature 对输出的影响
    
    【考点】
    temperature 控制的是 softmax 输出的概率分布：
    - temperature=0：几乎总是选概率最高的token → 输出确定、保守
    - temperature=1：正常采样 → 输出多样、自然
    - temperature=2：概率分布被拉平 → 输出随机、可能不连贯
    
    【思考题】什么业务场景该用低temperature？什么场景用高temperature？
    提示：代码生成 vs 创意写作
    """
    print("=" * 60)
    print("练习1：temperature 参数对比")
    print("=" * 60)

    prompt = "用两句话描述什么是人工智能"
    messages = [{"role": "user", "content": prompt}]

    for temp in [0, 1, 2]:
        print(f"\n--- temperature = {temp} ---")
        # 同一个prompt调用3次，观察输出是否一样
        for i in range(3):
            print(f"  第{i+1}次: ", end="")
            call_llm(messages, temperature=temp, max_tokens=200)
    
    print("\n💡 观察要点：")
    print("  - temperature=0 时，3次输出是否完全一样？")
    print("  - temperature 越高，输出是否越不同、越'发散'？")
    print("  - 想一想：你之前做食品安全问答，应该用多少 temperature？为什么？")


# ========================================
# 练习2：理解 system prompt 的作用
# ========================================
def exercise_2_system_prompt():
    """
    【目标】理解 system prompt 如何塑造模型行为
    
    【考点】
    messages 中的三种角色：
    - system：设定模型的身份和行为规则（对用户不可见）
    - user：用户的输入
    - assistant：模型的回复（多轮对话中需要传回去）
    
    system prompt 是 AI 应用开发中最重要的"可调参数"之一。
    你之前在食材管理中调用DeepSeek，本质上就是在写system prompt。
    """
    print("\n" + "=" * 60)
    print("练习2：system prompt 的威力")
    print("=" * 60)

    user_question = "苹果放冰箱一个月了还能吃吗？"

    # 场景A：没有 system prompt
    messages_a = [
        {"role": "user", "content": user_question}
    ]

    # 场景B：专业食品安全顾问
    messages_b = [
        {"role": "system", "content": (
            "你是一位专业的食品安全顾问，拥有10年食品检验经验。"
            "回答时需要：1）给出明确建议 2）解释食品变质的科学原理 "
            "3）提供安全判断方法。语气专业但通俗易懂。"
        )},
        {"role": "user", "content": user_question}
    ]

    # 场景C：逗比美食博主
    messages_c = [
        {"role": "system", "content": (
            "你是一个搞笑的美食博主，说话风格夸张幽默，喜欢用比喻和emoji。"
            "但在食品安全问题上依然给出正确建议，只是表达方式很搞笑。"
        )},
        {"role": "user", "content": user_question}
    ]

    scenarios = [
        ("A: 无system prompt", messages_a),
        ("B: 食品安全顾问", messages_b),
        ("C: 搞笑博主", messages_c),
    ]

    for name, msgs in scenarios:
        print(f"\n--- {name} ---")
        call_llm(msgs, temperature=0.7, max_tokens=500)

    print("\n💡 观察要点：")
    print("  - 同一个问题，不同的 system prompt 如何改变回答的风格和深度？")
    print("  - 这就是为什么 AI 应用开发中，prompt engineering 如此重要")
    print("  - 想一想：你的食材管理项目，system prompt 还能怎么优化？")


# ========================================
# 练习3：理解 max_tokens 与 token 计费
# ========================================
def exercise_3_max_tokens():
    """
    【目标】理解 token 的概念和 max_tokens 的实际影响
    
    【考点】
    - Token 不等于字符。英文大约 1 word ≈ 1.3 token，中文约 1 字 ≈ 1-2 token
    - max_tokens 限制的是「生成」的token数，不包括输入
    - 总消耗 = input_tokens（你发给模型的）+ output_tokens（模型生成的）
    - 这直接影响成本和延迟！AI应用必须关注token用量
    """
    print("\n" + "=" * 60)
    print("练习3：max_tokens 与 token 消耗")
    print("=" * 60)

    prompt = "详细解释一下RAG（检索增强生成）技术的原理和应用场景"
    messages = [{"role": "user", "content": prompt}]

    for max_tok in [50, 200, 1000]:
        print(f"\n--- max_tokens = {max_tok} ---")
        model = _create_llm(temperature=0, max_tokens=max_tok)
        response = model.invoke(_to_messages(messages))
        content = response.content
        usage = response.usage_metadata
        finish_reason = response.response_metadata.get("finish_reason", "")

        print(f"  回复内容（前100字）: {content[:100]}...")
        print(f"  回复是否被截断: {'是 ⚠️' if finish_reason == 'length' else '否 ✅'}")
        print(f"  输入token数: {usage['input_tokens']}")
        print(f"  输出token数: {usage['output_tokens']}")
        print(f"  总token数:   {usage['total_tokens']}")

    print("\n💡 观察要点：")
    print("  - max_tokens=50 时，回答是不是被截断了？")
    print("  - 输入token数三次是否一样？（应该一样，因为prompt没变）")
    print("  - 实际项目中，你需要根据场景合理设置 max_tokens")
    print("    比如：分类任务只需要10 tokens，报告生成可能需要2000+")


# ========================================
# 练习4：多轮对话 - 理解上下文管理
# ========================================
def exercise_4_multi_turn():
    """
    【目标】理解多轮对话的实现原理
    
    【考点 - 非常重要！】
    LLM 是无状态的！每次调用都是独立的。
    "记住上下文" 的方式是：每次调用都把完整的对话历史发过去。
    
    这意味着：
    1. 对话越长，input_tokens 越多 → 成本越高、速度越慢
    2. 超过上下文窗口长度 → 必须截断或总结历史
    3. 这就是为什么需要 Memory 机制（第2周会学到）
    
    这个知识点直接关联面试题「Agent的实现逻辑」——
    Agent的每一步推理都需要带上之前所有的思考和观察结果。
    """
    print("\n" + "=" * 60)
    print("练习4：多轮对话 - LLM如何「记住」上下文")
    print("=" * 60)

    # 模拟一段关于食材的多轮对话
    conversation = [
        {"role": "system", "content": "你是一个食品安全助手，简洁回答问题。"},
    ]

    # 预设的对话轮次
    user_inputs = [
        "鸡蛋放冰箱能保存多久？",
        "如果是已经煮熟的呢？",          # 测试：模型能否理解"的"指的是鸡蛋
        "那如果我周一煮的，周五还能吃吗？",  # 测试：模型能否结合前面的保存时间判断
    ]

    model = _create_llm(temperature=0, max_tokens=500)
    for i, user_input in enumerate(user_inputs, 1):
        print(f"\n--- 第{i}轮 ---")
        print(f"  👤 用户: {user_input}")

        # 添加用户消息到对话历史
        conversation.append({"role": "user", "content": user_input})

        # 调用LLM（注意：每次都发送完整的 conversation）
        response = model.invoke(_to_messages(conversation))
        assistant_reply = response.content
        usage = response.usage_metadata

        print(f"  🤖 助手: {assistant_reply}")
        print(f"  📊 本轮token消耗: 输入={usage['input_tokens']}, 输出={usage['output_tokens']}")

        # 把助手回复也加入对话历史（这步是关键！）
        conversation.append({"role": "assistant", "content": assistant_reply})

    # 对比：如果不带历史会怎样？
    print("\n\n--- 对比：不带上下文，直接问第3个问题 ---")
    isolated_messages = [
        {"role": "system", "content": "你是一个食品安全助手，简洁回答问题。"},
        {"role": "user", "content": "那如果我周一煮的，周五还能吃吗？"},
    ]
    print(f"  👤 用户: 那如果我周一煮的，周五还能吃吗？")
    print(f"  🤖 助手: ", end="")
    call_llm(isolated_messages, temperature=0, max_tokens=500)

    print("\n💡 观察要点：")
    print("  - 带上下文时，模型知道「的」指鸡蛋，「周一煮的」能结合保存时间判断")
    print("  - 不带上下文时，模型不知道你在说什么")
    print("  - 注意每轮的 input token 数是否在递增？这就是多轮对话的成本问题")
    print("  - 这也解释了为什么 Agent 的推理步骤越多，token 消耗越大")


# ========================================
# 🎯 运行入口
# ========================================
if __name__ == "__main__":
    print("🚀 AI工程师转型 - Day1 动手实践")
    print("📌 确保你已经设置了 DEEPSEEK_API_KEY 或在代码中填写了 API Key")
    print()

    exercises = {
        "1": ("temperature 参数对比", exercise_1_temperature),
        "2": ("system prompt 的威力", exercise_2_system_prompt),
        "3": ("max_tokens 与 token 消耗", exercise_3_max_tokens),
        "4": ("多轮对话与上下文管理", exercise_4_multi_turn),
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
    print("✅ Day1 练习完成！")
    print("=" * 60)
    print("""
📝 做完后请思考并用自己的话写下答案：

1. temperature 控制的是什么？什么场景用高/低值？
2. system prompt 在 AI 应用中扮演什么角色？
3. token 是什么？为什么要关注 token 消耗？
4. 多轮对话是怎么实现的？LLM 本身有记忆吗？
5. 以上这些知识，跟你之前做的食材管理项目有什么关联？

明天的内容：Embedding 与向量数据库 —— RAG 的第一块拼图 🧩
""")