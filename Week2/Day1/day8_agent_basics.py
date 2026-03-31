"""
=============================================================
  AI工程师转型 - 第2周第1天：动手实践
  主题：Agent基础概念 — 理解Agent到底是什么
=============================================================

使用前准备：
  pip install openai

今天的目标：
  搞清楚Agent的本质，能回答中最常见的三个问题：
  1. Agent是什么？
  2. ChatGPT是不是Agent？
  3. Agent和普通LLM调用有什么区别？

  今天先用纯LLM + 简单工具模拟Agent行为，
  明天再深入ReAct模式并手写一个完整Agent。

Agent的核心定义（标准答案）：
  Agent是一个能自主感知环境、做出决策、执行行动、
  并根据反馈持续调整的系统。

  核心组成：
  ┌────────────────────────────────┐
  │           Agent                │
  │  ┌──────┐  ┌───────┐          │
  │  │ LLM  │  │Memory │          │
  │  │(大脑) │  │(记忆)  │          │
  │  └──┬───┘  └───┬───┘          │
  │     │          │               │
  │  ┌──┴──────────┴──┐           │
  │  │   Planning     │           │
  │  │  (规划/推理)    │           │
  │  └──────┬─────────┘           │
  │         │                      │
  │  ┌──────┴─────────┐           │
  │  │    Tools        │           │
  │  │ (工具/技能)     │           │
  │  └────────────────┘           │
  └────────────────────────────────┘
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ========================================
# 🔧 配置
# ========================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "你的API Key")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"


# ========================================
# 练习1：普通LLM调用 vs Agent行为
# ========================================
def exercise_1_llm_vs_agent():
    """
    【目标】直观感受"普通LLM"和"Agent"的区别

    【考点】
    普通LLM调用：
      用户提问 → LLM回答 → 结束（一问一答）
    
    Agent：
      用户提出目标 → LLM分析需要哪些步骤 → 执行步骤1 → 观察结果
      → 决定下一步 → 执行步骤2 → ... → 直到目标完成

    关键区别：
      1. Agent有"目标驱动"的循环，不是一次性回答
      2. Agent能调用外部工具（搜索、计算、数据库等）
      3. Agent会根据中间结果调整计划
    """
    print("=" * 60)
    print("练习1：普通LLM调用 vs Agent行为")
    print("=" * 60)

    question = "北京今天气温多少度？适合穿什么衣服？"

    # --- 普通LLM调用 ---
    print("\n🅰️ 普通LLM调用（一问一答，无法获取实时信息）：")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=0,
        max_tokens=300,
    )
    print(f"   {response.choices[0].message.content[:200]}")
    print(f"\n   ⚠️ 问题：LLM不知道今天的实时天气，只能给笼统建议或编造数据")

    # --- Agent思维方式（用LLM模拟规划过程）---
    print(f"\n\n🅱️ Agent的思维方式（分析→规划→执行→观察→回答）：")

    planning_prompt = f"""你是一个智能助手Agent。用户问了这个问题："{question}"

请分析回答这个问题需要哪些步骤，以及每一步需要什么工具。
用JSON格式输出你的计划：
{{
  "goal": "用户的目标",
  "steps": [
    {{"step": 1, "action": "具体操作", "tool": "需要的工具", "reason": "为什么需要这步"}}
  ]
}}

只输出JSON，不要其他内容。"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": planning_prompt}],
        temperature=0,
        max_tokens=500,
    )

    plan = response.choices[0].message.content
    print(f"   Agent的执行计划：")
    # 尝试格式化输出
    try:
        plan_text = plan.replace("```json", "").replace("```", "").strip()
        plan_json = json.loads(plan_text)
        print(f"   目标: {plan_json.get('goal', 'N/A')}")
        for step in plan_json.get("steps", []):
            print(f"   Step {step['step']}: {step['action']}")
            print(f"          工具: {step['tool']}")
            print(f"          原因: {step['reason']}")
    except:
        print(f"   {plan[:300]}")

    print(f"\n💡 关键理解：")
    print(f"   普通LLM：收到问题→直接回答→结束")
    print(f"   Agent：收到问题→规划步骤→调用工具→观察结果→继续或回答")
    print(f"   Agent的本质：LLM作为'大脑'，驱动一个 感知→思考→行动 的循环")


# ========================================
# 练习2：ChatGPT是不是Agent？
# ========================================
def exercise_2_chatgpt_is_agent():
    """
    【目标】准备这道必考题的完美回答

    【标准答案】
    基础ChatGPT不是Agent：它只做对话，没有工具、没有自主行动能力。
    但ChatGPT + Plugins / Code Interpreter / Web Browsing 具有Agent特征：
    - 能调用工具（搜索、运行代码、读取文件）
    - 能根据结果决定下一步
    - 有一定的自主规划能力

    更准确的说法：ChatGPT是一个"有Agent能力的对话系统"，
    而不是一个"纯粹的Agent"。纯粹的Agent应该有：
    - 明确的目标驱动（不只是回答问题）
    - 持久的记忆系统
    - 主动行动能力（不需要用户每次触发）
    """
    print("\n" + "=" * 60)
    print("练习2：ChatGPT是不是Agent？（模拟）")
    print("=" * 60)

    # 用代码模拟"纯LLM对话"和"带工具的Agent对话"的区别

    # --- 场景：用户问一个需要计算的问题 ---
    question = "如果我每天存100元，存365天，加上年利率3%的复利，最终有多少钱？"

    # 纯LLM方式：直接算（可能算错）
    print(f"\n❓ 问题: {question}")
    print(f"\n--- 纯LLM（ChatGPT基础版）---")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=0,
        max_tokens=500,
    )
    print(f"   {response.choices[0].message.content[:300]}")

    # Agent方式：拆解任务 + 调用计算器工具
    print(f"\n--- Agent方式（ChatGPT + Code Interpreter）---")

    # 模拟Agent的工具
    def calculator(expression):
        """模拟计算器工具"""
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {e}"

    # 让LLM决定如何使用工具
    agent_prompt = f"""你是一个智能Agent，有一个计算器工具可以使用。

用户问题：{question}

请按以下步骤工作：
1. 先分析这个问题需要什么计算
2. 给出需要执行的Python计算表达式（用于传给计算器）
3. 用JSON格式输出：{{"thought": "你的思考过程", "expression": "Python表达式"}}

只输出JSON。"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": agent_prompt}],
        temperature=0,
        max_tokens=300,
    )

    try:
        result_text = response.choices[0].message.content
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        agent_decision = json.loads(result_text)

        print(f"   💭 Agent思考: {agent_decision.get('thought', 'N/A')}")
        expr = agent_decision.get("expression", "")
        print(f"   🔧 调用计算器: {expr}")

        calc_result = calculator(expr)
        print(f"   📊 工具返回: {calc_result}")

        # 用计算结果生成最终回答
        final_prompt = f"""用户问：{question}
经过计算，结果是：{calc_result}
请用通俗的语言向用户解释这个结果。"""

        final = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0,
            max_tokens=200,
        )
        print(f"   🤖 最终回答: {final.choices[0].message.content}")
    except Exception as e:
        print(f"   Agent执行出错: {e}")
        print(f"   原始输出: {response.choices[0].message.content[:200]}")

    print(f"""
💡 回答"ChatGPT是不是Agent"的框架：
   
   第一层：基础ChatGPT不是Agent
   → 只做对话，没有工具调用能力，不能自主行动
   
   第二层：ChatGPT + 插件具有Agent特征
   → 能搜索、运行代码、读取文件
   → 能根据中间结果决定下一步
   
   第三层：但还不是"完全的Agent"
   → 缺少持久记忆（对话结束就忘了）
   → 缺少主动性（需要用户每次触发）
   → 缺少长期目标驱动
   
   总结：ChatGPT正在从"对话系统"向"Agent"演进""")


# ========================================
# 练习3：定义工具（Function/Tool）
# ========================================
def exercise_3_define_tools():
    """
    【目标】理解Agent的工具是怎么定义和注册的

    【考点】
    Q：Agent是如何使用工具的？
    A：
      1. 开发者用JSON Schema定义工具（名称、描述、参数）
      2. 把工具定义发给LLM
      3. LLM根据用户问题，决定要不要调用工具、调用哪个
      4. LLM返回的是JSON格式的调用指令（函数名+参数）
      5. 你的代码负责实际执行函数
      6. 把执行结果返回给LLM，让它继续推理或生成最终回答

    重点：LLM不直接执行函数！它只是"建议"调用哪个函数。
    """
    print("\n" + "=" * 60)
    print("练习3：定义Agent的工具")
    print("=" * 60)

    # --- 定义几个工具 ---

    # 工具的Python实现
    def get_food_shelf_life(food_name):
        """查询食品保质期"""
        db = {
            "鸡蛋": {"冷藏": "30-45天", "常温": "10-15天"},
            "牛奶": {"冷藏": "开封后2-3天", "常温": "UHT可6个月"},
            "米饭": {"冷藏": "1-2天", "常温": "不超过2小时"},
            "苹果": {"冷藏": "1-2个月", "常温": "1-2周"},
        }
        info = db.get(food_name)
        if info:
            return json.dumps(info, ensure_ascii=False)
        return f"抱歉，暂无「{food_name}」的保质期信息"

    def calculate_nutrition(food_name, weight_grams):
        """计算食品营养成分"""
        nutrition_per_100g = {
            "鸡蛋": {"热量": 144, "蛋白质": 13.3, "脂肪": 8.8},
            "牛奶": {"热量": 54, "蛋白质": 3.0, "脂肪": 3.2},
            "米饭": {"热量": 116, "蛋白质": 2.6, "脂肪": 0.3},
            "苹果": {"热量": 52, "蛋白质": 0.3, "脂肪": 0.2},
        }
        base = nutrition_per_100g.get(food_name)
        if not base:
            return f"暂无「{food_name}」的营养数据"
        ratio = weight_grams / 100
        result = {k: round(v * ratio, 1) for k, v in base.items()}
        result["食品"] = food_name
        result["重量"] = f"{weight_grams}g"
        return json.dumps(result, ensure_ascii=False)

    def check_food_safety(food_name, storage_days, storage_method):
        """判断食品是否还安全"""
        limits = {
            "鸡蛋": {"冷藏": 45, "常温": 15},
            "牛奶": {"冷藏": 3, "常温": 0},
            "米饭": {"冷藏": 2, "常温": 0},
        }
        limit = limits.get(food_name, {}).get(storage_method)
        if limit is None:
            return f"暂无「{food_name}」在{storage_method}下的数据"
        safe = storage_days <= limit
        return json.dumps({
            "食品": food_name,
            "存放方式": storage_method,
            "已存放": f"{storage_days}天",
            "保质期限": f"{limit}天",
            "是否安全": "安全 ✅" if safe else "不安全 ❌",
            "建议": "可以食用" if safe else "建议丢弃，避免食物中毒",
        }, ensure_ascii=False)

    # --- 用JSON Schema定义工具（发给LLM的）---
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_food_shelf_life",
                "description": "查询指定食品的保质期信息，包括冷藏和常温下的保存时间",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "food_name": {
                            "type": "string",
                            "description": "食品名称，如：鸡蛋、牛奶、米饭",
                        }
                    },
                    "required": ["food_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_nutrition",
                "description": "计算指定重量食品的营养成分（热量、蛋白质、脂肪）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "food_name": {"type": "string", "description": "食品名称"},
                        "weight_grams": {"type": "number", "description": "重量（克）"},
                    },
                    "required": ["food_name", "weight_grams"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_food_safety",
                "description": "判断食品在特定存放条件下是否还安全可食用",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "food_name": {"type": "string", "description": "食品名称"},
                        "storage_days": {"type": "integer", "description": "已存放天数"},
                        "storage_method": {
                            "type": "string",
                            "enum": ["冷藏", "常温"],
                            "description": "存放方式",
                        },
                    },
                    "required": ["food_name", "storage_days", "storage_method"],
                },
            },
        },
    ]

    # 工具名 → 实际函数的映射
    tool_functions = {
        "get_food_shelf_life": get_food_shelf_life,
        "calculate_nutrition": calculate_nutrition,
        "check_food_safety": check_food_safety,
    }

    print("✅ 已定义3个工具:")
    for t in tools:
        f = t["function"]
        print(f"   🔧 {f['name']}: {f['description']}")
    print()

    # --- 让LLM选择工具 ---
    test_questions = [
        "鸡蛋能放多久？",
        "我吃了200克苹果，摄入了多少热量？",
        "牛奶开封后放了5天还能喝吗？是冷藏的。",
        "今天天气怎么样？",  # 故意问一个没有工具能处理的问题
    ]

    for q in test_questions:
        print(f"{'─'*50}")
        print(f"👤 {q}")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是食品安全助手，请根据用户问题选择合适的工具。"},
                {"role": "user", "content": q},
            ],
            tools=tools,
            tool_choice="auto",  # 让模型自己决定是否调用工具
            temperature=0,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                print(f"   🧠 LLM决定调用: {fn_name}")
                print(f"   📋 参数: {fn_args}")

                # 实际执行函数
                result = tool_functions[fn_name](**fn_args)
                print(f"   📊 执行结果: {result}")
        else:
            print(f"   💬 LLM直接回答（未调用工具）: {msg.content[:100]}")
        print()

    print("💡 关键理解（必讲）：")
    print("   1. 工具用JSON Schema定义 → LLM知道有哪些工具可用")
    print("   2. LLM返回的是调用指令（函数名+参数），不是执行结果！")
    print("   3. 你的代码负责实际执行函数，再把结果返回给LLM")
    print("   4. 如果没有合适的工具，LLM会直接用自己的知识回答")
    print("   5. tool_choice='auto' 让模型自主决定是否需要工具")

    return tools, tool_functions


# ========================================
# 练习4：完整的单轮工具调用循环
# ========================================
def exercise_4_tool_calling_loop(tools, tool_functions):
    """
    【目标】实现完整的工具调用循环（不只是调一次）

    【考点】
    Q：Agent调用工具后，返回内容是什么类型？
    A：LLM返回的是JSON格式的结构化数据（函数名+参数），
       不是自然语言！你的代码执行后，把结果作为
       tool role的消息返回给LLM，LLM再生成最终回答。

    完整流程：
    User → LLM（决定调工具）→ 你的代码执行工具 → 结果返回LLM → LLM生成回答
    """
    print("\n" + "=" * 60)
    print("练习4：完整的工具调用循环")
    print("=" * 60)

    def agent_with_tools(question):
        """
        完整的工具调用循环。
        这个函数的逻辑就是最基本的Agent实现。
        """
        messages = [
            {"role": "system", "content": "你是一个专业的食品安全助手。使用工具来帮助回答用户问题，给出实用的建议。"},
            {"role": "user", "content": question},
        ]

        print(f"\n👤 用户: {question}")
        print(f"{'─'*40}")

        # 最多循环3次（防止无限循环）
        for step in range(3):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
            )

            msg = response.choices[0].message

            # 情况1：LLM决定调用工具
            if msg.tool_calls:
                # 先把LLM的回复（含tool_calls）加入消息历史
                messages.append(msg)

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)
                    print(f"   Step {step+1} 🔧 调用工具: {fn_name}({fn_args})")

                    # 执行工具
                    result = tool_functions[fn_name](**fn_args)
                    print(f"           📊 结果: {result}")

                    # 把工具结果加入消息历史（注意role是"tool"）
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

            # 情况2：LLM直接给出最终回答（不再需要工具）
            else:
                print(f"\n   🤖 最终回答: {msg.content}")
                return msg.content

        print("   ⚠️ 达到最大循环次数")
        return messages[-1].get("content", "无法完成回答")

    # 测试
    test_questions = [
        "我冰箱里有鸡蛋放了40天了，还能吃吗？",
        "中午吃了两个鸡蛋（约120克），摄入了多少蛋白质？",
        "牛奶开了3天了放在冰箱里，还安全吗？如果安全的话，它有多少热量？（250ml约250克）",
    ]

    for q in test_questions:
        agent_with_tools(q)
        print()

    print("💡 注意第三个问题：")
    print("   它可能需要先调check_food_safety判断是否安全")
    print("   然后再调calculate_nutrition计算热量")
    print("   这就是Agent的「多步工具调用」—— 根据中间结果决定下一步")
    print("   这也引出了明天要学的ReAct模式")


# ========================================
# 练习5：Agent的核心组成总结
# ========================================
def exercise_5_agent_components():
    """
    【目标】整理Agent的四大组成部分，形成知识体系

    今天你实际接触了Agent的每个部分：
    - LLM（大脑）：练习1-2，理解LLM如何做决策
    - Tools（工具）：练习3，定义和注册工具
    - Planning（规划）：练习1的规划prompt，练习4的多步调用
    - Memory（记忆）：练习4的messages列表就是短期记忆
    """
    print("\n" + "=" * 60)
    print("练习5：Agent知识体系总结")
    print("=" * 60)

    summary = """
┌─────────────────────────────────────────────────────────────┐
│                    Agent 的四大组成                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. LLM（大脑）                                             │
│     - 理解用户意图                                          │
│     - 决定下一步行动                                         │
│     - 生成最终回答                                          │
│     ✅ 今天练习1-2已体验                                     │
│                                                             │
│  2. Tools（工具）                                           │
│     - 用JSON Schema定义                                     │
│     - LLM选择工具，代码执行                                  │
│     - Function Calling是最常见的实现方式                      │
│     ✅ 今天练习3已体验                                       │
│                                                             │
│  3. Planning（规划）                                        │
│     - 拆解复杂任务为子步骤                                   │
│     - 根据中间结果调整计划                                   │
│     - ReAct是最经典的规划模式（明天重点学）                    │
│     ✅ 今天练习1、4初步体验                                   │
│                                                             │
│  4. Memory（记忆）                                          │
│     - 短期记忆：对话历史（messages列表）                      │
│     - 长期记忆：向量数据库存储（上周学的RAG）                  │
│     ✅ 今天练习4的messages就是短期记忆                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Agent vs 普通LLM调用（必答）：                          │
│                                                             │
│  普通LLM:  输入 → 输出（一次性）                             │
│  Agent:    目标 → [思考→行动→观察] × N → 完成               │
│                                                             │
│  普通LLM:  只能用训练数据中的知识                            │
│  Agent:    能调用外部工具获取实时信息                         │
│                                                             │
│  普通LLM:  不会主动寻求缺失信息                              │
│  Agent:    发现信息不足时会主动调用工具补充                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""
    print(summary)

    print("📝 模拟 — 用30秒回答「Agent是什么」：")
    print()
    print('   "Agent是一个以LLM为核心的自主决策系统。')
    print('    它和普通LLM调用的区别在于：普通LLM是一问一答，')
    print('    而Agent是目标驱动的——它会自主规划步骤，调用外部')
    print('    工具执行，观察结果，然后决定下一步，直到目标完成。')
    print('    Agent有四个核心组成：LLM作为决策大脑，Tools提供')
    print('    外部能力，Planning负责任务拆解，Memory维持上下文。"')


# ========================================
# 🎯 运行入口
# ========================================
if __name__ == "__main__":
    print("🚀 AI工程师转型 - Week2 Day1 Agent基础\n")

    exercises = {
        "1": "普通LLM vs Agent行为",
        "2": "ChatGPT是不是Agent",
        "3": "定义Agent工具",
        "4": "完整工具调用循环 🔥",
        "5": "Agent知识体系总结",
        "all": "运行全部练习",
    }

    print("选择要运行的练习：")
    for key, name in exercises.items():
        print(f"  {key}. {name}")

    choice = input("\n请输入编号 (1/2/3/4/5/all): ").strip()

    tools_def = None
    tool_fns = None

    def ensure_tools():
        global tools_def, tool_fns
        if tools_def is None:
            tools_def, tool_fns = exercise_3_define_tools()
        return tools_def, tool_fns

    if choice == "1":
        exercise_1_llm_vs_agent()
    elif choice == "2":
        exercise_2_chatgpt_is_agent()
    elif choice == "3":
        exercise_3_define_tools()
    elif choice == "4":
        ensure_tools()
        exercise_4_tool_calling_loop(tools_def, tool_fns)
    elif choice == "5":
        exercise_5_agent_components()
    elif choice == "all":
        exercise_1_llm_vs_agent()
        print("\n" + "🔹" * 30 + "\n")
        exercise_2_chatgpt_is_agent()
        print("\n" + "🔹" * 30 + "\n")
        tools_def, tool_fns = exercise_3_define_tools()
        print("\n" + "🔹" * 30 + "\n")
        exercise_4_tool_calling_loop(tools_def, tool_fns)
        print("\n" + "🔹" * 30 + "\n")
        exercise_5_agent_components()

    print("\n\n" + "=" * 60)
    print("✅ Week2 Day1 完成！")
    print("=" * 60)
    print("""
📝 今日思考题：

1. 用一句话定义Agent。它和普通LLM调用的核心区别是什么？
2. ChatGPT是不是Agent？准备一个有层次的回答。
3. Function Calling中，LLM返回的是什么？谁负责实际执行函数？
4. 练习4中的messages列表扮演了什么角色？（提示：Memory）
5. 如果用户的问题需要调用多个工具，Agent是如何处理的？

明天的内容：ReAct模式 — Agent最经典的推理框架 🧠
  → 手写一个不用框架的ReAct Agent（第二周最重要的一天）
""")
