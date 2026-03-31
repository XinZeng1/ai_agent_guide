"""
=============================================================
  AI工程师转型 - 第2周第5天：动手实践
  主题：Agent调试与可观测性 + 第2周面试题全攻略
=============================================================

使用前准备：
  pip install langchain langchain-openai langchain-community

今天的目标：
  1. 学会调试Agent（出了问题怎么排查）
  2. 理解Agent在生产环境的关键问题
  3. 系统性准备面试中Agent相关的所有问题
"""

import os
import json
import time
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv
load_dotenv()
# ========================================
# 🔧 配置
# ========================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "你的API Key")
llm = ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY,
                 base_url="https://api.deepseek.com", temperature=0)


# ========================================
# 工具定义
# ========================================
@tool
def get_shelf_life(food_name: str) -> str:
    """查询食品保质期。传入食品名称。"""
    db = {"鸡蛋": "冷藏30-45天", "牛奶": "开封冷藏2-3天", "苹果": "冷藏1-2月"}
    return db.get(food_name, f"无{food_name}数据")

@tool
def check_safety(food_name: str, days: int, method: str) -> str:
    """判断食品安全性。传入食品名称、天数、存放方式(冷藏/常温)。"""
    limits = {("鸡蛋", "冷藏"): 45, ("牛奶", "冷藏"): 3, ("苹果", "冷藏"): 60}
    limit = limits.get((food_name, method))
    if not limit:
        return "无数据"
    return json.dumps({"安全": days <= limit, "上限": f"{limit}天"}, ensure_ascii=False)

@tool
def calc_nutrition(food_name: str, grams: float) -> str:
    """计算食品营养。传入食品名称和克数。"""
    data = {"鸡蛋": {"蛋白质g": 13.3, "热量kcal": 144}}
    base = data.get(food_name)
    if not base:
        return "无数据"
    return json.dumps({k: round(v * grams / 100, 1) for k, v in base.items()}, ensure_ascii=False)

ALL_TOOLS = [get_shelf_life, check_safety, calc_nutrition]


# ========================================
# 练习1：自定义Callback — Agent可观测性
# ========================================
def exercise_1_callbacks():
    """
    【目标】用Callback监控Agent的每一步行为

    【面试加分点】
    生产环境中Agent的可观测性至关重要：
    - 每步的决策是否合理
    - 工具调用耗时多少
    - Token消耗是多少
    - 哪一步出了错
    """
    print("=" * 60)
    print("练习1：自定义Callback — 监控Agent行为")
    print("=" * 60)

    class AgentMonitor(BaseCallbackHandler):
        """自定义的Agent监控器"""

        def __init__(self):
            self.steps = []
            self.start_time = None
            self.tool_times = {}

        def on_chain_start(self, serialized, inputs, **kwargs):
            self.start_time = time.time()
            print(f"\n  🚀 Agent启动")

        def on_tool_start(self, serialized, input_str, **kwargs):
            tool_name = serialized.get("name", "unknown")
            self.tool_times[tool_name] = time.time()
            print(f"  🔧 调用工具: {tool_name}")
            print(f"     输入: {input_str}")

        def on_tool_end(self, output, **kwargs):
            # 找最近开始的工具
            for name, start in self.tool_times.items():
                elapsed = time.time() - start
                print(f"     输出: {output[:80]}")
                print(f"     耗时: {elapsed:.3f}s")
                self.steps.append({"tool": name, "time": elapsed})
                break

        def on_chain_end(self, outputs, **kwargs):
            if self.start_time:
                total = time.time() - self.start_time
                if total > 0.5:  # 只在最外层chain打印
                    print(f"\n  📊 执行统计:")
                    print(f"     总耗时: {total:.2f}s")
                    print(f"     工具调用: {len(self.steps)}次")
                    for s in self.steps:
                        print(f"       - {s['tool']}: {s['time']:.3f}s")

    # 创建Agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是食品安全助手，使用工具回答问题。"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)

    monitor = AgentMonitor()
    executor = AgentExecutor(
        agent=agent, tools=ALL_TOOLS,
        verbose=False,
        callbacks=[monitor],
    )

    print("\n--- 带监控的Agent执行 ---")
    result = executor.invoke({"input": "鸡蛋放了20天还能吃吗？帮我算一个鸡蛋(60克)的营养"})
    print(f"\n  🤖 回答: {result['output']}")

    print(f"""
\n💡 Callback的作用：
   1. 不侵入Agent代码就能监控行为
   2. 可以接入日志系统（ELK/Prometheus）
   3. 可以实现告警（某工具耗时超阈值）
   4. LangSmith本质上也是通过Callback实现的
""")


# ========================================
# 练习2：常见Agent问题排查
# ========================================
def exercise_2_debugging():
    """
    【目标】知道Agent出问题时怎么排查

    常见问题：
    1. Agent不调用工具 → 工具描述不清楚
    2. Agent调用了错误的工具 → 工具描述有歧义
    3. Agent陷入循环 → Prompt设计问题
    4. Agent回答不基于工具结果 → system prompt没约束好
    """
    print("\n" + "=" * 60)
    print("练习2：Agent常见问题与排查")
    print("=" * 60)

    # 问题1：工具描述太差导致不调用
    print("\n--- 问题1：工具描述对调用行为的影响 ---\n")

    @tool
    def bad_tool(x: str) -> str:
        """处理数据"""  # 描述太模糊
        return f"已处理: {x}"

    @tool
    def good_tool(food_name: str) -> str:
        """查询指定食品在冰箱冷藏和常温环境下的保质期天数。输入食品名称如"鸡蛋"、"牛奶"等。"""
        return f"{food_name}: 冷藏30天"

    print("  ❌ 差的工具描述: '处理数据'")
    print("     → LLM不知道什么时候该用它")
    print()
    print("  ✅ 好的工具描述: '查询指定食品在冰箱冷藏和常温环境下的保质期天数。")
    print("                    输入食品名称如"鸡蛋"、"牛奶"等。'")
    print("     → LLM清楚知道何时使用、传什么参数")

    # 问题2：prompt对回答质量的影响
    print("\n\n--- 问题2：System Prompt对Agent行为的影响 ---\n")

    bad_prompt = ChatPromptTemplate.from_messages([
        ("system", "回答问题"),  # 太简单
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    good_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一个专业的食品安全助手。\n"
            "规则：\n"
            "1. 必须使用工具获取数据，不要凭记忆回答\n"
            "2. 如果工具返回无数据，明确告知用户\n"
            "3. 回答要包含具体数字和实用建议\n"
            "4. 不确定时说不确定，不要编造"
        )),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    print("  ❌ 差的Prompt: '回答问题'")
    print("     → Agent可能不用工具直接编造答案")
    print()
    print("  ✅ 好的Prompt: 明确角色 + 必须用工具 + 不准编造")
    print("     → Agent行为可控，回答质量有保障")

    # 问题3：调试清单
    print(f"""
\n--- Agent调试清单 ---

  Agent不调用工具：
    □ 检查工具描述是否清楚
    □ 检查tool_choice是否设置正确
    □ 检查system prompt是否要求使用工具
    □ 试试verbose=True看LLM的决策过程

  Agent调用了错误的工具：
    □ 检查工具描述是否有歧义
    □ 多个工具的描述是否有重叠
    □ 参数定义是否清晰（特别是enum类型）

  Agent陷入循环：
    □ 检查max_iterations是否设置
    □ 工具返回的错误信息是否有指导性
    □ Prompt是否明确了何时应该停止

  Agent回答不基于工具结果：
    □ System prompt是否约束了"只基于工具结果回答"
    □ temperature是否设太高（Agent场景建议0-0.3）
    □ 工具返回的信息是否足够回答问题
""")


# ========================================
# 练习3：Agent的Token消耗分析
# ========================================
def exercise_3_token_analysis():
    """
    【目标】理解Agent的token成本，学会优化

    Agent是token消耗大户：
    - 每步都要发完整的消息历史
    - 工具定义本身就占不少token
    - 多轮工具调用，token累积很快
    """
    print("\n" + "=" * 60)
    print("练习3：Agent Token消耗分析")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是食品安全助手。"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    executor = AgentExecutor(
        agent=agent, tools=ALL_TOOLS,
        verbose=False, return_intermediate_steps=True,
    )

    questions = [
        ("简单（单工具）", "鸡蛋能放多久？"),
        ("中等（两步）", "鸡蛋放冰箱30天了还安全吗？"),
        ("复杂（多步）", "鸡蛋放了20天安全吗？如果安全帮我算两个鸡蛋120克的营养"),
    ]

    print(f"\n{'难度':<15} {'工具调用次数':>10} {'中间步骤':>8}")
    print("─" * 40)

    for label, q in questions:
        result = executor.invoke({"input": q})
        steps = len(result["intermediate_steps"])
        tool_calls = sum(1 for _ in result["intermediate_steps"])
        print(f"{label:<15} {tool_calls:>8}次   {steps:>6}步")

    print(f"""
\n💡 Token优化策略：
   1. 工具描述精简：不需要的细节删掉，每个工具节省几十token
   2. 减少工具数量：只注册当前场景需要的工具
   3. Prompt精简：避免过长的system prompt
   4. 历史截断：长对话中只保留最近N轮
   5. 工具结果精简：返回JSON而非长文本
""")


# ========================================
# 练习4：第2周面试题全攻略
# ========================================
def exercise_4_interview_prep():
    """
    把你上次面试的Agent相关题目全部覆盖，
    每道题给出结构化的回答框架。
    """
    print("\n" + "=" * 60)
    print("练习4：面试题全攻略 🔥（背下来）")
    print("=" * 60)

    qa_pairs = [
        {
            "q": "Agent是什么？",
            "a": (
                "Agent是一个以LLM为核心的自主决策系统。它包含四个组成部分：\n"
                "LLM（决策大脑）、Tools（外部工具）、Planning（任务规划）、Memory（记忆）。\n"
                "与普通LLM调用的区别：普通LLM是一问一答，Agent是目标驱动的循环——\n"
                "感知→思考→行动→观察，直到目标完成。"
            ),
        },
        {
            "q": "ChatGPT是不是Agent？",
            "a": (
                "分层回答：基础ChatGPT不是Agent，它只做对话。\n"
                "但ChatGPT + Plugins/Code Interpreter具有Agent特征——能调用工具、\n"
                "根据结果决定下一步。\n"
                "不过它还不是完全的Agent：缺少持久记忆、缺少主动性、\n"
                "缺少长期目标驱动。它正在从对话系统向Agent演进。"
            ),
        },
        {
            "q": "ChatGPT和Agent的区别？",
            "a": (
                "ChatGPT本质是对话系统：接收输入→生成回复→结束。\n"
                "Agent是自主系统：接收目标→规划步骤→调用工具→观察结果→\n"
                "决定下一步→...→完成目标。\n"
                "关键区别：Agent能调用外部工具、能多步推理、有目标驱动的循环。"
            ),
        },
        {
            "q": "Agent的实现逻辑是什么？",
            "a": (
                "以ReAct为例：\n"
                "1. 用户输入目标\n"
                "2. LLM思考（Thought）：分析需要什么信息\n"
                "3. LLM选择工具（Action）：返回JSON格式的调用指令\n"
                "4. 代码执行工具，获取结果（Observation）\n"
                "5. 结果返回给LLM，继续思考\n"
                "6. 重复2-5直到LLM认为可以给出最终答案（Final Answer）\n"
                "核心是一个 Thought→Action→Observation 的循环。"
            ),
        },
        {
            "q": "Agent调用后返回内容是什么类型？如何解决并发调用问题？",
            "a": (
                "返回类型：结构化JSON（函数名 + 参数），不是执行结果！\n"
                "LLM只「建议」调什么工具和参数，你的代码负责实际执行。\n\n"
                "并发问题：用DAG（有向无环图）编排。\n"
                "1. 分析工具调用间的依赖关系\n"
                "2. 无依赖的 → asyncio.gather 并发执行\n"
                "3. 有依赖的 → 按拓扑排序，等上一层完成再执行下一层\n"
                "4. 兼顾了正确性（依赖顺序）和效率（最大化并发）"
            ),
        },
        {
            "q": "Agent如何使用工具/MCP？",
            "a": (
                "Function Calling方式：\n"
                "1. 用JSON Schema定义工具（名称、描述、参数）\n"
                "2. 工具定义随请求发给LLM\n"
                "3. LLM返回调用指令（JSON格式）\n"
                "4. 你的代码解析JSON → 执行对应函数 → 返回结果\n"
                "5. 结果作为Observation发回LLM\n\n"
                "MCP方式（更标准化）：\n"
                "Host（你的应用）→ Client（MCP客户端）→ Server（工具服务）\n"
                "MCP统一了工具接入协议，类似USB统一了外设接口。"
            ),
        },
        {
            "q": "ReAct是什么？",
            "a": (
                "ReAct = Reasoning + Acting，是Agent最经典的推理框架。\n"
                "核心循环：Thought（思考）→ Action（调用工具）→ Observation（观察结果）\n"
                "反复循环直到能给出Final Answer。\n\n"
                "对比CoT（只推理不行动）：CoT依赖LLM记忆，可能编造。\n"
                "ReAct通过工具获取真实数据，更准确。\n"
                "优势：准确、可追踪、可调试。\n"
                "局限：token消耗大、可能陷入循环、速度较慢。"
            ),
        },
    ]

    for i, qa in enumerate(qa_pairs, 1):
        print(f"\n{'━'*50}")
        print(f"  Q{i}: {qa['q']}")
        print(f"{'━'*50}")
        print(f"  {qa['a']}")

    print(f"""
\n\n📋 面试准备建议：

  1. 每道题练习两个版本：
     - 30秒简答版（电梯演讲）
     - 3分钟详答版（深入展开，配例子）

  2. 准备画图讲解：
     - Agent的四大组件图
     - ReAct的循环流程图
     - Function Calling的消息流转图
     - DAG编排的依赖图

  3. 准备代码讲解：
     - "我手写过ReAct Agent，核心是一个while循环..."
     - "Function Calling的消息有三种role: user/assistant/tool..."
     - "并发用asyncio.gather，依赖用拓扑排序..."

  4. 结合你的项目：
     - "在我的食材管理项目中，Agent集成了保质期查询、
        安全判断、营养计算三个工具..."
""")


# ========================================
# 练习5：周末综合实践规划
# ========================================
def exercise_5_weekend_plan():
    """
    规划周末的综合实践
    """
    print("\n" + "=" * 60)
    print("练习5：周末综合实践规划")
    print("=" * 60)

    print("""
📋 周末目标：构建「智能食材助手Agent」

  把第1周的RAG + 第2周的Agent整合成一个完整系统：

  ┌────────────────────────────────────────────────────┐
  │                 智能食材助手Agent                     │
  │                                                      │
  │  ┌─────────────────────────────────────────┐        │
  │  │            ReAct 推理循环                 │        │
  │  │  Thought → Action → Observation → ...   │        │
  │  └──────────────┬──────────────────────────┘        │
  │                 │                                    │
  │    ┌────────────┼────────────┐                      │
  │    ↓            ↓            ↓                      │
  │  ┌──────┐  ┌──────┐  ┌──────────┐                  │
  │  │ RAG  │  │安全   │  │营养计算  │                  │
  │  │知识库 │  │判断   │  │         │                  │
  │  └──────┘  └──────┘  └──────────┘                  │
  │    ↑                                                │
  │  向量数据库(食品安全文档)                              │
  └────────────────────────────────────────────────────┘

  具体步骤：

  1. RAG知识库（复用第1周成果）
     - 食品安全文档 → 切片 → Embedding → ChromaDB
     - 包装成一个工具：rag_search(query)

  2. Agent工具集
     - rag_search: RAG检索食品安全知识
     - check_safety: 判断食品安全性
     - calc_nutrition: 计算营养成分
     - get_recipe: 推荐菜谱

  3. ReAct Agent
     - 可以用手写版或LangChain版
     - 支持多轮对话
     - 有完整的推理日志

  4. 准备面试讲解
     - 画系统架构图
     - 准备3分钟项目讲解话术
     - 准备常见追问的回答

  预计用时：4-6小时
""")


# ========================================
# 🎯 运行入口
# ========================================
if __name__ == "__main__":
    print("🚀 AI工程师转型 - Week2 Day5 Agent调试与面试准备\n")

    exercises = {
        "1": "Callback监控Agent行为",
        "2": "Agent常见问题排查",
        "3": "Token消耗分析",
        "4": "面试题全攻略 🔥",
        "5": "周末综合实践规划",
        "all": "运行全部练习",
    }

    print("选择要运行的练习：")
    for key, name in exercises.items():
        print(f"  {key}. {name}")

    choice = input("\n请输入编号 (1/2/3/4/5/all): ").strip()

    if choice == "1": exercise_1_callbacks()
    elif choice == "2": exercise_2_debugging()
    elif choice == "3": exercise_3_token_analysis()
    elif choice == "4": exercise_4_interview_prep()
    elif choice == "5": exercise_5_weekend_plan()
    elif choice == "all":
        for fn in [exercise_1_callbacks, exercise_2_debugging,
                    exercise_3_token_analysis, exercise_4_interview_prep,
                    exercise_5_weekend_plan]:
            fn()
            print("\n" + "🔹" * 30 + "\n")

    print("\n" + "=" * 60)
    print("✅ Week2 Day5 完成！第二周学习完成！🎉")
    print("=" * 60)
    print("""
🏗️ 第2周总结：

  Day1: Agent基础 → 理解Agent的定义和四大组成
  Day2: ReAct Agent → 手写Agent（最重要的一天）
  Day3: Function Calling → 工具调用协议和并发处理
  Day4: LangChain Agent → 用框架重写
  Day5: 调试+面试 → 工程化考量和面试准备

  现在回看你上次的面试题：
  ✅ Agent是什么 → Day1
  ✅ ChatGPT是不是Agent → Day1
  ✅ Agent的实现逻辑 → Day2 (ReAct)
  ✅ Agent返回内容类型 → Day3 (JSON)
  ✅ 并发调用问题 → Day3 (DAG编排)
  ✅ Agent如何使用工具/MCP → Day3 (Function Calling)
  ✅ ReAct是什么 → Day2

  这些题你现在应该都能流利回答了！💪

🗓️ 下周预告（第3周：Agent进阶 + MCP）：
  - DAG编排框架（LangGraph）
  - Hierarchical多Agent架构
  - MCP协议详解
  - 多Agent协作系统
""")
