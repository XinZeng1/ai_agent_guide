"""
=============================================================
  AI工程师转型 - 第2周第4天：动手实践
  主题：LangChain Agent — 用框架重写Agent
=============================================================

使用前准备：
  pip install langchain langchain-openai langchain-community

今天的目标：
  前三天你手写了Agent的每个环节，今天用LangChain重写，
  体会框架封装了什么，以及框架的Agent和你手写的有什么区别。
"""

import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv
load_dotenv()
# ========================================
# 🔧 配置
# ========================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "你的API Key")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0,
)


# ========================================
# 练习1：用@tool装饰器定义工具
# ========================================
def exercise_1_define_tools():
    """
    【目标】学会LangChain中定义工具的方式

    LangChain中定义工具有三种方式：
    1. @tool 装饰器（最简单，推荐）
    2. StructuredTool.from_function()
    3. 继承BaseTool类（最灵活）

    装饰器方式最常用，函数的docstring会自动变成工具描述。
    """
    print("=" * 60)
    print("练习1：LangChain 工具定义")
    print("=" * 60)

    @tool
    def get_shelf_life(food_name: str) -> str:
        """查询食品的保质期信息。传入食品名称，返回冷藏和常温下的保存时间。"""
        db = {
            "鸡蛋": "冷藏30-45天，常温10-15天。大头朝上放冰箱内侧",
            "牛奶": "巴氏奶开封后冷藏2-3天，UHT奶未开封常温6个月",
            "三文鱼": "冷藏不超过24小时，冷冻2-3个月",
            "苹果": "冷藏1-2个月，与其他水果分开存放",
        }
        return db.get(food_name, f"暂无「{food_name}」的保质期数据")

    @tool
    def check_food_safety(food_name: str, days: int, method: str) -> str:
        """判断食品是否还安全可食用。传入食品名称、已存放天数、存放方式（冷藏/常温）。"""
        limits = {
            ("鸡蛋", "冷藏"): 45, ("鸡蛋", "常温"): 15,
            ("牛奶", "冷藏"): 3, ("三文鱼", "冷藏"): 1,
        }
        limit = limits.get((food_name, method))
        if limit is None:
            return f"暂无「{food_name}」在{method}条件下的安全数据"
        safe = days <= limit
        return json.dumps({
            "食品": food_name, "安全": safe,
            "已存放": f"{days}天", "上限": f"{limit}天",
            "建议": "可以食用" if safe else "建议丢弃",
        }, ensure_ascii=False)

    @tool
    def calc_nutrition(food_name: str, grams: float) -> str:
        """计算指定重量食品的营养成分。传入食品名称和重量（克）。"""
        data = {
            "鸡蛋": {"蛋白质g": 13.3, "热量kcal": 144, "脂肪g": 8.8},
            "牛奶": {"蛋白质g": 3.0, "热量kcal": 54, "脂肪g": 3.2},
        }
        base = data.get(food_name)
        if not base:
            return f"暂无「{food_name}」的营养数据"
        result = {k: round(v * grams / 100, 1) for k, v in base.items()}
        result["食品"] = food_name
        result["重量"] = f"{grams}g"
        return json.dumps(result, ensure_ascii=False)

    tools = [get_shelf_life, check_food_safety, calc_nutrition]

    # 看看LangChain帮我们生成了什么
    print("\n已定义的工具：\n")
    for t in tools:
        print(f"  🔧 {t.name}")
        print(f"     描述: {t.description}")
        print(f"     参数Schema: {json.dumps(t.args, ensure_ascii=False, indent=6)}")
        print()

    print("💡 对比昨天手写的JSON Schema定义：")
    print("   @tool装饰器自动从函数签名和docstring生成Schema")
    print("   省去了手写JSON Schema的繁琐，但本质是一样的")

    return tools


# ========================================
# 练习2：创建Tool-Calling Agent
# ========================================
def exercise_2_create_agent(tools):
    """
    【目标】用LangChain创建标准的Tool-Calling Agent

    LangChain的Agent类型：
    - create_tool_calling_agent：推荐，使用模型原生的Function Calling
    - create_react_agent：使用ReAct Prompt模板
    - create_structured_chat_agent：使用结构化输出
    """
    print("\n" + "=" * 60)
    print("练习2：创建LangChain Agent")
    print("=" * 60)

    # 定义Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的食品安全助手。使用可用的工具来帮助回答用户的问题，给出简洁实用的建议。"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),  # Agent的"草稿纸"，记录中间步骤
    ])

    # 创建Agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 用AgentExecutor包装（它负责运行Agent的循环）
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,    # 打印每一步的详细信息
        max_iterations=5, # 最大迭代次数
        handle_parsing_errors=True,  # 自动处理解析错误
    )

    print("✅ Agent创建完成\n")
    print("组成部分：")
    print("  - LLM: DeepSeek-Chat")
    print("  - Tools: 3个食品安全工具")
    print("  - Prompt: 带chat_history和agent_scratchpad")
    print("  - AgentExecutor: 负责运行循环，max_iterations=5\n")

    # 测试
    questions = [
        "鸡蛋放冰箱30天了还能吃吗？",
        "帮我算一下两个鸡蛋（120克）的营养成分",
    ]

    for q in questions:
        print(f"\n{'━'*50}")
        print(f"👤 {q}")
        print(f"{'━'*50}")
        result = agent_executor.invoke({"input": q})
        print(f"\n🤖 最终回答: {result['output']}")

    return agent_executor


# ========================================
# 练习3：带记忆的Agent
# ========================================
def exercise_3_agent_with_memory(tools):
    """
    【目标】让Agent支持多轮对话

    AgentExecutor + chat_history = 带记忆的Agent
    """
    print("\n" + "=" * 60)
    print("练习3：带记忆的多轮对话Agent")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是食品安全助手。使用工具回答问题，简洁实用。"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # 手动管理对话历史
    chat_history = []

    conversations = [
        "鸡蛋放冰箱能放多久？",
        "那如果已经放了40天呢？",   # 需要理解"那"指鸡蛋
        "它的营养成分怎么样？一个大概60克", # 需要理解"它"指鸡蛋
    ]

    print()
    for q in conversations:
        print(f"👤 {q}")
        result = agent_executor.invoke({
            "input": q,
            "chat_history": chat_history,
        })
        answer = result["output"]
        print(f"🤖 {answer}\n")

        # 更新对话历史
        chat_history.extend([
            HumanMessage(content=q),
            AIMessage(content=answer),
        ])

    print("💡 对比你昨天手写的多轮Agent：")
    print("   LangChain用chat_history + MessagesPlaceholder自动处理")
    print("   你不用手动拼接对话历史到prompt中")


# ========================================
# 练习4：AgentExecutor参数详解
# ========================================
def exercise_4_executor_params(tools):
    """
    【目标】理解AgentExecutor的关键参数

    这些参数在面试中可能不会直接考，
    但说出来证明你真正用过Agent框架，不是只看了文档。
    """
    print("\n" + "=" * 60)
    print("练习4：AgentExecutor 关键参数")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是食品安全助手。"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    params = {
        "verbose": "是否打印每步详情。开发时True，生产时False",
        "max_iterations": "最大循环次数。防止无限循环，通常设5-10",
        "max_execution_time": "最大执行时间(秒)。超时强制停止",
        "handle_parsing_errors": "LLM输出格式不对时自动处理",
        "return_intermediate_steps": "返回中间步骤（用于调试和日志）",
        "early_stopping_method": "'force'直接停止 / 'generate'让LLM总结已有信息",
    }

    print("\nAgentExecutor 关键参数：\n")
    for param, desc in params.items():
        print(f"  📋 {param}")
        print(f"     {desc}\n")

    # 测试 return_intermediate_steps
    print("--- 测试 return_intermediate_steps ---\n")
    executor = AgentExecutor(
        agent=agent, tools=tools, verbose=False,
        return_intermediate_steps=True,
    )

    result = executor.invoke({"input": "鸡蛋放冰箱20天了还安全吗？"})

    print(f"最终回答: {result['output']}\n")
    print(f"中间步骤 ({len(result['intermediate_steps'])} 步):")
    for i, (action, observation) in enumerate(result["intermediate_steps"]):
        print(f"  Step {i+1}:")
        print(f"    Tool: {action.tool}")
        print(f"    Input: {action.tool_input}")
        print(f"    Output: {observation[:80]}...")
        print()

    print("💡 intermediate_steps的用途：")
    print("   1. 调试：看Agent的每步决策是否合理")
    print("   2. 日志：记录完整的推理链路")
    print("   3. 监控：统计工具调用频率和耗时")
    print("   4. 类似你昨天手写Agent中的trace功能")


# ========================================
# 练习5：LangChain Agent vs 手写Agent
# ========================================
def exercise_5_comparison():
    """
    【目标】系统对比，形成面试话术
    """
    print("\n" + "=" * 60)
    print("练习5：框架Agent vs 手写Agent — 总结对比")
    print("=" * 60)

    print("""
  ┌────────────────┬───────────────────────────────┬───────────────────────────────┐
  │     维度        │ 手写Agent（Day2-3）            │ LangChain Agent（今天）        │
  ├────────────────┼───────────────────────────────┼───────────────────────────────┤
  │ 工具定义       │ 手写JSON Schema               │ @tool装饰器自动生成            │
  │ 推理循环       │ 手写while循环+解析             │ AgentExecutor封装              │
  │ 错误处理       │ 自己写try/catch               │ handle_parsing_errors          │
  │ 循环保护       │ 自己写max_steps               │ max_iterations                  │
  │ 对话记忆       │ 手动管理messages列表           │ chat_history + Placeholder     │
  │ 调试          │ print / 自己写trace            │ verbose=True / LangSmith       │
  │ 可替换性       │ 换LLM要改代码                 │ 换一行（ChatOpenAI→其他）      │
  │ 代码量        │ ~100行                        │ ~30行                          │
  │ 理解深度       │ 每个细节都清楚                │ 被封装了，要看源码才知道       │
  │ 灵活性        │ 完全自由                      │ 受框架约束                     │
  └────────────────┴───────────────────────────────┴───────────────────────────────┘

  面试话术：
  "我先用纯Python手写了ReAct Agent，包括Prompt模板、输出解析、
   工具执行循环、错误处理、循环检测等。这让我深入理解了Agent的
   每个环节。之后用LangChain重构，提高了开发效率——工具定义用
   @tool装饰器，循环用AgentExecutor，调试用verbose和LangSmith。
   
   我认为手写的价值在于理解原理，框架的价值在于工程效率。
   生产中我会用框架快速搭建，遇到框架解决不了的问题时
   回到底层手写定制化逻辑。"
""")


# ========================================
# 🎯 运行入口
# ========================================
if __name__ == "__main__":
    print("🚀 AI工程师转型 - Week2 Day4 LangChain Agent\n")

    exercises = {
        "1": "LangChain工具定义",
        "2": "创建Tool-Calling Agent",
        "3": "带记忆的多轮Agent",
        "4": "AgentExecutor参数详解",
        "5": "框架vs手写 总结对比",
        "all": "运行全部练习",
    }

    print("选择要运行的练习：")
    for key, name in exercises.items():
        print(f"  {key}. {name}")

    choice = input("\n请输入编号 (1/2/3/4/5/all): ").strip()

    tools = None
    executor = None

    def ensure_tools():
        nonlocal tools
        if tools is None:
            tools = exercise_1_define_tools()
        return tools

    if choice == "1": exercise_1_define_tools()
    elif choice == "2":
        tools = ensure_tools()
        exercise_2_create_agent(tools)
    elif choice == "3":
        tools = ensure_tools()
        exercise_3_agent_with_memory(tools)
    elif choice == "4":
        tools = ensure_tools()
        exercise_4_executor_params(tools)
    elif choice == "5": exercise_5_comparison()
    elif choice == "all":
        tools = exercise_1_define_tools()
        print("\n" + "🔹" * 30 + "\n")
        exercise_2_create_agent(tools)
        print("\n" + "🔹" * 30 + "\n")
        exercise_3_agent_with_memory(tools)
        print("\n" + "🔹" * 30 + "\n")
        exercise_4_executor_params(tools)
        print("\n" + "🔹" * 30 + "\n")
        exercise_5_comparison()

    print("\n" + "=" * 60)
    print("✅ Week2 Day4 完成！")
    print("=" * 60)
    print("""
📝 今日思考题：

1. @tool装饰器做了什么？它自动生成的JSON Schema从哪里来？
2. AgentExecutor的agent_scratchpad是什么？为什么需要它？
3. verbose=True时看到的输出，对应你手写Agent中的哪些部分？
4. 框架Agent和手写Agent各自的优劣是什么？什么时候用哪个？

明天：Agent调试与可观测性 + 第二周综合实战 🔍
""")
