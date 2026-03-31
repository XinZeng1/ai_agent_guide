"""
=============================================================
  AI工程师转型 - 第2周第3天：动手实践
  主题：Function Calling深入 + 并发工具调用
=============================================================

使用前准备：
  pip install openai asyncio

今天的目标：
  1. 深入理解Function Calling的完整协议
  2. 解决面试题：有前后依赖关系的工具调用如何处理
  3. 学会用asyncio并发调用无依赖的工具
"""

import os
import json
import asyncio
import time
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
# ========================================
# 🔧 配置
# ========================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "你的API Key")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
async_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"


# ========================================
# 工具集
# ========================================
def get_shelf_life(food_name: str) -> str:
    time.sleep(0.1)  # 模拟网络延迟
    db = {
        "鸡蛋": {"冷藏天数": 45, "常温天数": 15, "建议": "大头朝上放冰箱内侧"},
        "牛奶": {"冷藏天数": 3, "常温天数": 0, "建议": "开封后尽快饮用"},
        "三文鱼": {"冷藏天数": 1, "常温天数": 0, "建议": "当天食用或冷冻"},
        "苹果": {"冷藏天数": 60, "常温天数": 14, "建议": "与其他水果分开存放"},
    }
    return json.dumps(db.get(food_name, {"error": f"无{food_name}数据"}), ensure_ascii=False)

def check_safety(food_name: str, days: int, method: str) -> str:
    time.sleep(0.1)
    limits = {("鸡蛋", "冷藏"): 45, ("牛奶", "冷藏"): 3, ("三文鱼", "冷藏"): 1}
    limit = limits.get((food_name, method))
    if limit is None:
        return json.dumps({"error": "无数据"}, ensure_ascii=False)
    return json.dumps({"安全": days <= limit, "剩余天数": max(0, limit - days)}, ensure_ascii=False)

def calc_nutrition(food_name: str, grams: float) -> str:
    time.sleep(0.1)
    data = {"鸡蛋": {"蛋白质g": 13.3, "热量kcal": 144}, "牛奶": {"蛋白质g": 3.0, "热量kcal": 54}}
    base = data.get(food_name)
    if not base:
        return json.dumps({"error": f"无{food_name}数据"}, ensure_ascii=False)
    return json.dumps({k: round(v * grams / 100, 1) for k, v in base.items()}, ensure_ascii=False)

def get_recipe(ingredients: list) -> str:
    time.sleep(0.2)  # 模拟较慢的API
    recipes = {
        frozenset(["鸡蛋"]): "推荐：水煮蛋、蒸蛋羹、煎蛋",
        frozenset(["鸡蛋", "牛奶"]): "推荐：法式吐司、蛋奶布丁、奶香炒蛋",
    }
    key = frozenset(ingredients)
    for k, v in recipes.items():
        if k.issubset(key) or key.issubset(k):
            return v
    return f"用{', '.join(ingredients)}可以做：炒菜、煮汤等"

TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "get_shelf_life", "description": "查询食品保质期",
        "parameters": {"type": "object", "properties": {"food_name": {"type": "string"}}, "required": ["food_name"]}}},
    {"type": "function", "function": {
        "name": "check_safety", "description": "判断食品安全性",
        "parameters": {"type": "object", "properties": {
            "food_name": {"type": "string"}, "days": {"type": "integer"},
            "method": {"type": "string", "enum": ["冷藏", "常温"]}}, "required": ["food_name", "days", "method"]}}},
    {"type": "function", "function": {
        "name": "calc_nutrition", "description": "计算食品营养",
        "parameters": {"type": "object", "properties": {
            "food_name": {"type": "string"}, "grams": {"type": "number"}}, "required": ["food_name", "grams"]}}},
    {"type": "function", "function": {
        "name": "get_recipe", "description": "根据食材推荐菜谱",
        "parameters": {"type": "object", "properties": {
            "ingredients": {"type": "array", "items": {"type": "string"}}}, "required": ["ingredients"]}}},
]

TOOL_MAP = {
    "get_shelf_life": get_shelf_life,
    "check_safety": check_safety,
    "calc_nutrition": calc_nutrition,
    "get_recipe": get_recipe,
}


# ========================================
# 练习1：Function Calling完整协议
# ========================================
def exercise_1_full_protocol():
    """
    【目标】理解Function Calling的完整消息流转

    【面试考点】
    完整的消息流：
    1. User消息 → 发给LLM（带tools定义）
    2. LLM返回 assistant消息（含tool_calls字段）
    3. 你执行函数，结果作为 tool消息 发回
    4. LLM看到tool结果，生成最终回答

    注意：tool_calls可能包含多个调用（并行调用场景）
    """
    print("=" * 60)
    print("练习1：Function Calling 完整消息流转")
    print("=" * 60)

    question = "鸡蛋放冰箱30天了还能吃吗？顺便告诉我两个鸡蛋的营养"

    messages = [
        {"role": "system", "content": "你是食品安全助手，用工具回答问题。"},
        {"role": "user", "content": question},
    ]

    print(f"\n👤 Step 1 - 用户消息: {question}")
    print(f"   (同时发送了{len(TOOLS_SCHEMA)}个工具定义)\n")

    # 第一次调用：LLM决定调用哪些工具
    response = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS_SCHEMA,
        tool_choice="auto", temperature=0,
    )

    assistant_msg = response.choices[0].message

    print(f"🤖 Step 2 - LLM返回 (finish_reason={response.choices[0].finish_reason}):")
    if assistant_msg.tool_calls:
        print(f"   tool_calls包含 {len(assistant_msg.tool_calls)} 个调用：")
        for tc in assistant_msg.tool_calls:
            print(f"   ├─ id: {tc.id}")
            print(f"   ├─ function.name: {tc.function.name}")
            print(f"   ├─ function.arguments: {tc.function.arguments}")
            print(f"   └─ (注意：这是JSON字符串，不是执行结果！)")
            print()

    # 把assistant消息加入历史
    messages.append(assistant_msg)

    # 执行每个工具调用
    print(f"🔧 Step 3 - 你的代码执行工具：")
    for tc in assistant_msg.tool_calls:
        fn_name = tc.function.name
        fn_args = json.loads(tc.function.arguments)
        result = TOOL_MAP[fn_name](**fn_args)
        print(f"   {fn_name}({fn_args}) → {result}")

        # 每个工具结果作为tool消息返回
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,  # 必须和tool_call的id对应！
            "content": result,
        })

    print(f"\n📤 Step 4 - 把tool结果发回给LLM：")
    print(f"   (发送了{len(messages)}条消息，包含原始问题+assistant+tool结果)")

    # 第二次调用：LLM基于工具结果生成最终回答
    final_response = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0, max_tokens=500,
    )

    print(f"\n🤖 Step 5 - LLM最终回答:")
    print(f"   {final_response.choices[0].message.content}")

    print(f"""
\n💡 完整协议要点（面试必知）：
   1. tool_calls中的每个调用都有唯一的id
   2. tool消息的tool_call_id必须和对应的id匹配
   3. LLM可能一次返回多个tool_calls（并行调用意图）
   4. 你的代码可以并行执行这些调用，也可以串行执行
   5. finish_reason="tool_calls"表示LLM想调用工具
      finish_reason="stop"表示LLM给出了最终回答
""")


# ========================================
# 练习2：有依赖关系的工具调用（面试核心题）
# ========================================
def exercise_2_dependent_calls():
    """
    【目标】解决面试题：有前后依赖关系时如何处理并发调用

    【面试考点】
    Q：Agent有多个工具调用，有些有前后依赖关系，怎么处理？
    A：用DAG（有向无环图）来编排：
      1. 分析工具调用之间的依赖关系
      2. 没有依赖的可以并行执行
      3. 有依赖的按拓扑排序串行执行

    例子：
      "鸡蛋放了30天还能吃吗？能吃的话帮我推荐食谱和算营养。"

      依赖关系：
        check_safety ─┬→ get_recipe     (只有安全才推荐)
                      └→ calc_nutrition  (只有安全才算营养)

      执行策略：
        先执行 check_safety
        → 如果安全，并行执行 get_recipe 和 calc_nutrition
        → 如果不安全，直接告诉用户不要吃
    """
    print("\n" + "=" * 60)
    print("练习2：依赖关系分析与执行编排")
    print("=" * 60)

    question = "冰箱里有鸡蛋放了30天、牛奶开了2天。帮我看看哪些还能吃，能吃的帮我算下营养（两个鸡蛋120克，一杯牛奶250克），再推荐个食谱。"

    print(f"\n❓ {question}\n")

    # --- 分析依赖关系 ---
    print("📊 依赖关系分析：")
    print()
    print("  check_safety(鸡蛋, 30天) ──┐")
    print("  check_safety(牛奶, 2天)  ──┤")
    print("      (这两个互不依赖，可以并行)  │")
    print("                               ↓")
    print("      [等待安全检查结果]")
    print("                               ↓")
    print("  calc_nutrition(鸡蛋, 120g) ──┐  (只对安全的食品)")
    print("  calc_nutrition(牛奶, 250g) ──┤  (可以并行)")
    print("  get_recipe([安全的食材])   ──┘")
    print("                               ↓")
    print("      [汇总所有结果，生成最终回答]")

    # --- 按DAG执行 ---
    print(f"\n\n🔄 按DAG顺序执行：\n")

    # Layer 1：并行检查安全性
    print("  Layer 1（并行）: 安全检查")
    start = time.time()
    safety_egg = check_safety("鸡蛋", 30, "冷藏")
    safety_milk = check_safety("牛奶", 2, "冷藏")
    layer1_time = time.time() - start

    print(f"    鸡蛋: {safety_egg}")
    print(f"    牛奶: {safety_milk}")
    print(f"    耗时: {layer1_time:.3f}s (串行，实际可并行)")

    # 判断哪些安全
    safe_foods = []
    egg_safe = json.loads(safety_egg).get("安全", False)
    milk_safe = json.loads(safety_milk).get("安全", False)
    if egg_safe:
        safe_foods.append("鸡蛋")
    if milk_safe:
        safe_foods.append("牛奶")

    print(f"\n    安全的食品: {safe_foods}")

    # Layer 2：对安全的食品并行执行后续操作
    if safe_foods:
        print(f"\n  Layer 2（并行）: 营养计算 + 菜谱推荐")
        start = time.time()

        results = {}
        if egg_safe:
            results["鸡蛋营养"] = calc_nutrition("鸡蛋", 120)
        if milk_safe:
            results["牛奶营养"] = calc_nutrition("牛奶", 250)
        results["菜谱"] = get_recipe(safe_foods)

        layer2_time = time.time() - start

        for k, v in results.items():
            print(f"    {k}: {v}")
        print(f"    耗时: {layer2_time:.3f}s (串行，实际可并行)")

    print(f"""
\n💡 面试回答"如何处理有依赖关系的并发调用"：

   "首先分析工具调用之间的依赖关系，构建一个DAG（有向无环图）。
   然后按拓扑排序分层执行：
   
   1. 同一层的调用互不依赖 → 可以并行执行（用asyncio）
   2. 不同层之间有依赖 → 必须等上一层完成才能开始下一层
   3. 某些调用的结果会影响后续调用是否需要执行（条件分支）
   
   这就是DAG编排的核心思想，LangGraph就是基于这个原理实现的。"
""")


# ========================================
# 练习3：asyncio并发工具调用
# ========================================
def exercise_3_async_tools():
    """
    【目标】用asyncio实现真正的并发工具调用

    【面试考点】
    Python并发 + AI应用的结合：
    - LLM API调用是IO密集型 → 用asyncio协程最合适
    - 多个无依赖的工具调用 → asyncio.gather并发执行
    - 有依赖的 → await上一步结果后再执行下一步
    """
    print("\n" + "=" * 60)
    print("练习3：asyncio并发工具调用")
    print("=" * 60)

    # 异步版本的工具
    async def async_get_shelf_life(food_name):
        await asyncio.sleep(0.3)  # 模拟300ms网络延迟
        return get_shelf_life(food_name)

    async def async_check_safety(food_name, days, method):
        await asyncio.sleep(0.3)
        return check_safety(food_name, days, method)

    async def async_calc_nutrition(food_name, grams):
        await asyncio.sleep(0.3)
        return calc_nutrition(food_name, grams)

    async def async_get_recipe(ingredients):
        await asyncio.sleep(0.5)  # 模拟更慢的API
        return get_recipe(ingredients)

    # --- 串行执行 ---
    async def serial_execution():
        print("\n  📊 串行执行（一个接一个）：")
        start = time.time()

        r1 = await async_check_safety("鸡蛋", 30, "冷藏")
        r2 = await async_check_safety("牛奶", 2, "冷藏")
        r3 = await async_calc_nutrition("鸡蛋", 120)
        r4 = await async_calc_nutrition("牛奶", 250)
        r5 = await async_get_recipe(["鸡蛋", "牛奶"])

        elapsed = time.time() - start
        print(f"     5个调用总耗时: {elapsed:.3f}s")
        print(f"     (约 0.3+0.3+0.3+0.3+0.5 = 1.7s)")
        return elapsed

    # --- 并发执行（无依赖全部并行）---
    async def parallel_execution():
        print("\n  📊 全部并发执行（忽略依赖）：")
        start = time.time()

        results = await asyncio.gather(
            async_check_safety("鸡蛋", 30, "冷藏"),
            async_check_safety("牛奶", 2, "冷藏"),
            async_calc_nutrition("鸡蛋", 120),
            async_calc_nutrition("牛奶", 250),
            async_get_recipe(["鸡蛋", "牛奶"]),
        )

        elapsed = time.time() - start
        print(f"     5个调用总耗时: {elapsed:.3f}s")
        print(f"     (约等于最慢的那个: 0.5s)")
        return elapsed

    # --- DAG编排执行（考虑依赖）---
    async def dag_execution():
        print("\n  📊 DAG编排执行（考虑依赖关系）：")
        start = time.time()

        # Layer 1: 并发检查安全性
        safety_results = await asyncio.gather(
            async_check_safety("鸡蛋", 30, "冷藏"),
            async_check_safety("牛奶", 2, "冷藏"),
        )
        print(f"     Layer 1 完成: {time.time()-start:.3f}s")

        # 分析结果
        safe_foods = []
        if json.loads(safety_results[0]).get("安全"):
            safe_foods.append("鸡蛋")
        if json.loads(safety_results[1]).get("安全"):
            safe_foods.append("牛奶")

        # Layer 2: 对安全的食品并发执行
        tasks = []
        if "鸡蛋" in safe_foods:
            tasks.append(async_calc_nutrition("鸡蛋", 120))
        if "牛奶" in safe_foods:
            tasks.append(async_calc_nutrition("牛奶", 250))
        if safe_foods:
            tasks.append(async_get_recipe(safe_foods))

        if tasks:
            layer2_results = await asyncio.gather(*tasks)
            print(f"     Layer 2 完成: {time.time()-start:.3f}s")

        elapsed = time.time() - start
        print(f"     总耗时: {elapsed:.3f}s")
        print(f"     (Layer1: 0.3s + Layer2: 0.5s ≈ 0.8s)")
        return elapsed

    # 运行对比
    print("对比三种执行策略的耗时：\n")

    serial_time = asyncio.run(serial_execution())
    parallel_time = asyncio.run(parallel_execution())
    dag_time = asyncio.run(dag_execution())

    print(f"\n  📈 耗时对比：")
    print(f"     串行:    {serial_time:.3f}s ({'█' * int(serial_time * 20)})")
    print(f"     全并发:  {parallel_time:.3f}s ({'█' * int(parallel_time * 20)})")
    print(f"     DAG编排: {dag_time:.3f}s ({'█' * int(dag_time * 20)})")

    print(f"""
\n💡 面试回答"Python协程在AI应用中的用途"：

   "AI应用中大量操作是IO密集型的——调LLM API、查向量数据库、
   调外部工具，这些都是在等网络响应。
   
   asyncio协程非常适合这个场景：
   1. 单线程内通过await切换，没有线程安全问题
   2. gather()可以并发执行多个无依赖的调用
   3. 配合DAG编排，既保证依赖顺序又最大化并发度
   
   对比线程：协程更轻量，没有GIL竞争问题，
   也不会出现100个线程+1结果不确定的问题。"
""")


# ========================================
# 练习4：tool_choice策略
# ========================================
def exercise_4_tool_choice():
    """
    【目标】理解tool_choice参数的不同取值

    tool_choice 控制LLM是否必须调用工具：
    - "auto": LLM自己决定是否调工具（默认）
    - "none": 禁止调工具，强制直接回答
    - "required": 强制必须调至少一个工具
    - {"type":"function","function":{"name":"xxx"}}: 强制调指定工具
    """
    print("\n" + "=" * 60)
    print("练习4：tool_choice 策略对比")
    print("=" * 60)

    question = "苹果和牛奶能一起吃吗？"  # 这个问题工具可能不太适用

    strategies = [
        ("auto", "auto", "LLM自主决定"),
        ("none", "none", "禁止调用工具"),
        ("required", "required", "强制调用工具"),
    ]

    for name, choice, desc in strategies:
        print(f"\n--- tool_choice='{name}' ({desc}) ---")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是食品安全助手"},
                {"role": "user", "content": question},
            ],
            tools=TOOLS_SCHEMA,
            tool_choice=choice,
            temperature=0,
            max_tokens=300,
        )

        msg = response.choices[0].message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"   🔧 调用: {tc.function.name}({tc.function.arguments})")
        else:
            print(f"   💬 直接回答: {msg.content[:100]}")

    print(f"""
\n💡 什么时候用哪种策略：
   auto:     大多数场景，让LLM自主判断
   none:     纯聊天场景，或者工具结果已经拿到了
   required: 强制执行某个操作（如：必须先做安全检查）
   指定工具:  明确知道需要哪个工具时（跳过LLM的选择过程）
""")


# ========================================
# 练习5：多轮工具调用Agent
# ========================================
def exercise_5_multi_turn_agent():
    """
    【目标】实现完整的多轮工具调用Agent

    把今天学的所有内容串起来：
    Function Calling协议 + 依赖分析 + 并发执行 + 多轮对话
    """
    print("\n" + "=" * 60)
    print("练习5：完整的多轮工具调用Agent")
    print("=" * 60)

    def smart_agent(question):
        messages = [
            {"role": "system", "content": (
                "你是一个智能食品助手。请用工具帮助用户解决问题。"
                "如果需要多步操作，请一步一步来。"
            )},
            {"role": "user", "content": question},
        ]

        print(f"\n👤 {question}")
        print(f"{'─'*40}")

        for round_num in range(5):
            response = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS_SCHEMA,
                tool_choice="auto", temperature=0,
            )

            msg = response.choices[0].message

            # 没有工具调用 → 最终回答
            if not msg.tool_calls:
                print(f"\n🤖 回答: {msg.content}")
                return msg.content

            # 有工具调用 → 执行并继续
            messages.append(msg)

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                print(f"   Round {round_num+1} 🔧 {fn_name}({fn_args})")

                result = TOOL_MAP[fn_name](**fn_args)
                print(f"            → {result}")

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        return "达到最大轮次"

    # 测试
    tests = [
        "帮我看看冰箱里放了20天的鸡蛋还能吃吗？如果能吃帮我推荐个食谱。",
        "牛奶开了4天了，还能喝吗？",
        "今天早上吃了两个鸡蛋约120克，喝了一杯牛奶250克，总共摄入了多少蛋白质？",
    ]

    for q in tests:
        smart_agent(q)
        print()


# ========================================
# 🎯 运行入口
# ========================================
if __name__ == "__main__":
    print("🚀 AI工程师转型 - Week2 Day3 Function Calling深入\n")

    exercises = {
        "1": "Function Calling完整协议",
        "2": "依赖关系分析与DAG编排 🔥",
        "3": "asyncio并发工具调用",
        "4": "tool_choice策略对比",
        "5": "完整多轮工具调用Agent",
        "all": "运行全部练习",
    }

    print("选择要运行的练习：")
    for key, name in exercises.items():
        print(f"  {key}. {name}")

    choice = input("\n请输入编号 (1/2/3/4/5/all): ").strip()

    if choice == "1": exercise_1_full_protocol()
    elif choice == "2": exercise_2_dependent_calls()
    elif choice == "3": exercise_3_async_tools()
    elif choice == "4": exercise_4_tool_choice()
    elif choice == "5": exercise_5_multi_turn_agent()
    elif choice == "all":
        for fn in [exercise_1_full_protocol, exercise_2_dependent_calls,
                    exercise_3_async_tools, exercise_4_tool_choice, exercise_5_multi_turn_agent]:
            fn()
            print("\n" + "🔹" * 30 + "\n")

    print("\n" + "=" * 60)
    print("✅ Week2 Day3 完成！")
    print("=" * 60)
    print("""
📝 今日思考题：

1. Function Calling的完整消息流是怎样的？（5步）
2. 有依赖关系的工具调用如何处理？画出DAG并解释。
3. asyncio.gather()在AI应用中的典型使用场景？
4. tool_choice的几种取值分别在什么场景下使用？
5. 对比串行/全并发/DAG编排三种策略的优劣。

明天：LangChain Agent框架 — 用框架重写Agent 🔧
""")
