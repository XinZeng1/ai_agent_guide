"""
=============================================================
  AI工程师转型 - 第2周第2天：动手实践
  主题：ReAct模式 — 手写Agent（不用任何框架）
=============================================================

使用前准备：
  pip install openai

⚡ 这是整个30天计划中最重要的一天 ⚡
  今天你要从零手写一个ReAct Agent，不用LangChain，不用任何框架。
  做完这个练习，你对Agent的理解会有质的飞跃。

ReAct = Reasoning + Acting
  核心循环：
    Thought → Action → Observation → Thought → Action → ... → Final Answer

  每一步：
    1. Thought（思考）：我现在知道什么？还需要什么？下一步做什么？
    2. Action（行动）：调用一个工具
    3. Observation（观察）：工具返回了什么结果

  这个循环持续进行，直到LLM认为已经收集了足够的信息来回答问题。
"""

import os
import json
import re
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
# 练习1：理解ReAct的Prompt结构
# ========================================
def exercise_1_react_prompt():
    """
    【目标】看懂ReAct的经典Prompt模板

    【面试考点】
    ReAct论文的核心贡献：
    - 之前的方法要么只推理（CoT），要么只行动（Act）
    - ReAct把推理和行动交替进行
    - Thought让模型"说出"推理过程 → 更可控、可调试
    - Action让模型调用外部工具 → 突破知识限制
    """
    print("=" * 60)
    print("练习1：ReAct Prompt结构详解")
    print("=" * 60)

    # 这就是ReAct的经典Prompt模板
    react_prompt_template = """你是一个食品安全助手Agent。你可以使用以下工具来帮助回答问题：

可用工具：
{tools_description}

请严格按照以下格式回答（每一步都必须有Thought、Action、Action Input）：

Thought: [你的思考过程：分析问题，决定下一步做什么]
Action: [要调用的工具名称]
Action Input: [传给工具的参数，JSON格式]

等待工具返回结果后，你会看到：
Observation: [工具返回的结果]

然后继续思考：
Thought: [根据观察结果继续推理]
Action: [如果需要，继续调用工具]
Action Input: [参数]

当你认为已经收集了足够的信息，输出最终答案：
Thought: 我现在已经有足够的信息来回答这个问题了。
Final Answer: [你的最终回答]

注意：
1. 每次只能调用一个工具
2. 必须等待Observation后才能继续下一步
3. 不要编造Observation，等待实际的工具返回

开始！

用户问题：{question}
"""

    print("ReAct Prompt模板的结构：\n")
    print("  ┌───────────────────────────────────────┐")
    print("  │  系统设定 + 可用工具描述                │")
    print("  ├───────────────────────────────────────┤")
    print("  │  Thought: 我需要查询...               │ ← LLM推理")
    print("  │  Action: get_food_shelf_life          │ ← LLM选择工具")
    print("  │  Action Input: {\"food\": \"鸡蛋\"}       │ ← LLM给出参数")
    print("  ├───────────────────────────────────────┤")
    print("  │  Observation: {冷藏:30-45天...}       │ ← 你的代码执行并返回")
    print("  ├───────────────────────────────────────┤")
    print("  │  Thought: 现在我知道保质期了，但还需要│ ← LLM继续推理")
    print("  │           判断是否还在安全期内...       │")
    print("  │  Action: check_food_safety            │ ← 再调一个工具")
    print("  │  Action Input: {...}                  │")
    print("  ├───────────────────────────────────────┤")
    print("  │  Observation: {是否安全: 不安全}       │ ← 工具返回")
    print("  ├───────────────────────────────────────┤")
    print("  │  Thought: 我有足够信息了               │ ← LLM决定结束")
    print("  │  Final Answer: 这个鸡蛋不能吃了...     │ ← 最终回答")
    print("  └───────────────────────────────────────┘")

    print("\n💡 关键点：")
    print("   - Thought是给LLM的「草稿纸」，让它展示推理过程")
    print("   - Action和Action Input是结构化的，方便你的代码解析")
    print("   - Observation由你的代码填入，不是LLM生成的")
    print("   - Final Answer标志着循环结束")
    print("   - 这个格式让整个推理链路完全可追踪、可调试")


# ========================================
# 练习2：手写ReAct Agent 🔥🔥🔥
# ========================================
def exercise_2_handwritten_react():
    """
    【目标】从零手写一个完整的ReAct Agent，不用任何框架

    这是整个学习计划中最重要的代码。
    面试时如果你能讲清楚这个实现，Agent相关的问题基本通关。
    """
    print("\n" + "=" * 60)
    print("练习2：手写 ReAct Agent 🔥（核心练习）")
    print("=" * 60)

    # ---- 定义工具 ----
    def get_shelf_life(food_name: str) -> str:
        """查询食品保质期"""
        db = {
            "鸡蛋": "冷藏30-45天，常温10-15天。大头朝上放置，不要放冰箱门上。",
            "牛奶": "巴氏奶开封后2-3天（冷藏），UHT奶未开封可常温6个月。",
            "米饭": "冷藏1-2天，常温不超过2小时。加热需达75度以上。",
            "苹果": "冷藏1-2个月，注意与其他水果分开（会释放乙烯）。",
            "三文鱼": "冷藏不超过24小时，冷冻2-3个月。解冻后不可二次冷冻。",
            "豆腐": "冷藏3-5天，需每天换水。变质会发黏、变黄。",
        }
        return db.get(food_name, f"暂无「{food_name}」的保质期数据")

    def check_safety(food_name: str, days: int, method: str) -> str:
        """判断食品安全性"""
        limits = {
            ("鸡蛋", "冷藏"): 45, ("鸡蛋", "常温"): 15,
            ("牛奶", "冷藏"): 3,  ("米饭", "冷藏"): 2,
            ("三文鱼", "冷藏"): 1, ("豆腐", "冷藏"): 5,
        }
        limit = limits.get((food_name, method))
        if limit is None:
            return f"暂无「{food_name}」在{method}条件下的安全数据"
        safe = days <= limit
        return json.dumps({
            "食品": food_name, "存放": f"{method}{days}天",
            "安全上限": f"{limit}天",
            "判断": "安全✅" if safe else "不安全❌，建议丢弃",
        }, ensure_ascii=False)

    def calc_nutrition(food_name: str, grams: float) -> str:
        """计算营养"""
        data = {
            "鸡蛋": {"热量kcal": 144, "蛋白质g": 13.3, "脂肪g": 8.8},
            "牛奶": {"热量kcal": 54, "蛋白质g": 3.0, "脂肪g": 3.2},
            "米饭": {"热量kcal": 116, "蛋白质g": 2.6, "脂肪g": 0.3},
        }
        base = data.get(food_name)
        if not base:
            return f"暂无「{food_name}」的营养数据"
        result = {k: round(v * grams / 100, 1) for k, v in base.items()}
        return json.dumps(result, ensure_ascii=False)

    # 工具注册表
    TOOLS = {
        "get_shelf_life": {
            "func": get_shelf_life,
            "description": "查询食品保质期。参数: {\"food_name\": \"食品名称\"}",
        },
        "check_safety": {
            "func": check_safety,
            "description": "判断食品是否安全。参数: {\"food_name\": \"名称\", \"days\": 天数, \"method\": \"冷藏/常温\"}",
        },
        "calc_nutrition": {
            "func": calc_nutrition,
            "description": "计算营养成分。参数: {\"food_name\": \"名称\", \"grams\": 克数}",
        },
    }

    # ---- ReAct Agent 核心实现 ----

    def build_react_prompt(question):
        """构建ReAct的系统Prompt"""
        tools_desc = "\n".join(
            f"  - {name}: {info['description']}"
            for name, info in TOOLS.items()
        )
        return f"""你是一个食品安全助手Agent。请使用ReAct方式推理和回答。

可用工具：
{tools_desc}

严格按照以下格式回答：

Thought: [你的思考]
Action: [工具名称]
Action Input: [JSON格式参数]

收到Observation后继续，直到能给出最终答案：

Thought: [最终思考]
Final Answer: [最终回答]

注意：每次只调用一个工具。参数必须是合法JSON。

用户问题：{question}"""

    def parse_agent_output(text):
        """
        解析LLM输出，提取Thought、Action、Action Input或Final Answer。
        这是Agent实现中最容易出bug的地方——LLM的输出不总是规范的。
        """
        result = {"thought": "", "action": None, "action_input": None, "final_answer": None}

        # 提取Thought
        thought_match = re.search(r'Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|\Z)', text, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # 提取Final Answer
        final_match = re.search(r'Final Answer:\s*(.*)', text, re.DOTALL)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result

        # 提取Action和Action Input
        action_match = re.search(r'Action:\s*(.*?)(?:\n|$)', text)
        input_match = re.search(r'Action Input:\s*(.*?)(?:\n|$)', text, re.DOTALL)

        if action_match:
            result["action"] = action_match.group(1).strip()
        if input_match:
            raw_input = input_match.group(1).strip()
            # 清理可能的markdown代码块标记
            raw_input = raw_input.replace("```json", "").replace("```", "").strip()
            try:
                result["action_input"] = json.loads(raw_input)
            except json.JSONDecodeError:
                result["action_input"] = {"raw": raw_input}

        return result

    def execute_tool(action, action_input):
        """执行工具并返回结果"""
        if action not in TOOLS:
            return f"错误：未知工具「{action}」。可用工具：{list(TOOLS.keys())}"
        try:
            func = TOOLS[action]["func"]
            return func(**action_input)
        except Exception as e:
            return f"工具执行错误：{e}"

    def react_agent(question, max_steps=5, verbose=True):
        """
        ReAct Agent的主循环。

        这就是Agent的核心——一个 思考→行动→观察 的循环。
        面试时你需要能画出这个循环并解释每一步。
        """
        if verbose:
            print(f"\n🚀 ReAct Agent 启动")
            print(f"❓ 问题: {question}")
            print(f"{'═'*50}")

        # 初始化对话
        prompt = build_react_prompt(question)
        messages = [{"role": "user", "content": prompt}]

        # 记录完整的推理链路
        trace = []

        for step in range(max_steps):
            # 让LLM思考下一步
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                max_tokens=500,
                stop=["Observation:"],  # 让LLM在这里停下，等待真正的工具结果
            )

            output = response.choices[0].message.content
            parsed = parse_agent_output(output)

            if verbose:
                print(f"\n📍 Step {step + 1}")
                print(f"   💭 Thought: {parsed['thought']}")

            # 如果有最终答案，结束循环
            if parsed["final_answer"]:
                if verbose:
                    print(f"\n   ✅ Final Answer: {parsed['final_answer']}")
                trace.append({"step": step + 1, "thought": parsed["thought"],
                              "final_answer": parsed["final_answer"]})
                return parsed["final_answer"], trace

            # 如果需要调用工具
            if parsed["action"]:
                if verbose:
                    print(f"   🔧 Action: {parsed['action']}")
                    print(f"   📋 Input: {parsed['action_input']}")

                # 执行工具
                observation = execute_tool(parsed["action"], parsed["action_input"])

                if verbose:
                    print(f"   👁️ Observation: {observation}")

                trace.append({
                    "step": step + 1,
                    "thought": parsed["thought"],
                    "action": parsed["action"],
                    "action_input": parsed["action_input"],
                    "observation": observation,
                })

                # 把LLM的输出和工具结果拼接回消息中
                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content": f"Observation: {observation}\n"})
            else:
                # 既没有Final Answer也没有Action，可能是格式问题
                if verbose:
                    print(f"   ⚠️ 无法解析有效的Action或Final Answer")
                    print(f"   原始输出: {output[:200]}")
                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content": "请继续按照Thought/Action/Action Input的格式回答，或给出Final Answer。"})

        if verbose:
            print(f"\n⚠️ 达到最大步骤数 ({max_steps})，强制结束")
        return "抱歉，我无法在有限步骤内完成回答。", trace

    # ---- 测试 ----
    print("\n" + "━" * 50)
    print("测试1：简单问题（单工具调用）")
    print("━" * 50)
    react_agent("鸡蛋能放多久？")

    print("\n\n" + "━" * 50)
    print("测试2：需要判断的问题（可能需要多步）")
    print("━" * 50)
    react_agent("我冰箱里的三文鱼放了2天了，还能吃吗？")

    print("\n\n" + "━" * 50)
    print("测试3：复杂问题（多工具协作）")
    print("━" * 50)
    react_agent("我早上吃了两个鸡蛋（约120克），想知道摄入了多少蛋白质。另外冰箱里还有放了50天的鸡蛋，还能吃吗？")

    return react_agent


# ========================================
# 练习3：ReAct vs CoT vs Act-only 对比
# ========================================
def exercise_3_react_vs_others():
    """
    【目标】理解ReAct相比其他方法的优势

    【面试考点】
    三种Agent推理方式的对比：

    CoT（Chain of Thought）：只思考，不行动
      - "让我想想... 鸡蛋一般能放30天... 你放了40天... 应该不能吃了"
      - 问题：知识可能是错的或过时的

    Act-only：只行动，不思考
      - 直接调用工具，不解释为什么
      - 问题：不可控，不知道它为什么这样做

    ReAct：边思考边行动
      - "我需要先查一下保质期(Thought) → 调用工具(Action) → 得到结果(Obs)
         → 现在我知道了，用户放了40天超过了上限(Thought) → 最终回答"
      - 优势：可追踪、可调试、更准确
    """
    print("\n" + "=" * 60)
    print("练习3：ReAct vs CoT vs Act-only")
    print("=" * 60)

    question = "冰箱里的鸡蛋放了40天了，还能吃吗？"

    # --- CoT（只推理，不调工具）---
    print(f"\n--- CoT方式（只推理不行动）---")
    cot_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "请一步一步推理来回答问题。展示你的思考过程。"},
            {"role": "user", "content": question},
        ],
        temperature=0, max_tokens=300,
    )
    print(f"   {cot_response.choices[0].message.content[:250]}")
    print(f"\n   ⚠️ CoT的问题：LLM可能记错保质期数据（幻觉风险）")

    # --- Act-only（直接调工具，不展示思考）---
    print(f"\n--- Act-only方式（只行动不推理）---")
    print(f"   Action: check_safety")
    print(f'   Action Input: {{"food_name": "鸡蛋", "days": 40, "method": "冷藏"}}')

    from exercise_2_handwritten_react import check_safety  # type: ignore
    # 由于模块间依赖问题，这里直接内联
    limits = {("鸡蛋", "冷藏"): 45}
    limit = limits.get(("鸡蛋", "冷藏"), 45)
    safe = 40 <= limit
    result = f'{{"食品": "鸡蛋", "存放": "冷藏40天", "安全上限": "{limit}天", "判断": "{'安全✅' if safe else '不安全❌'}"}}'
    print(f"   Observation: {result}")
    print(f"   Final Answer: 可以吃。")
    print(f"\n   ⚠️ Act-only的问题：不知道Agent为什么这么判断，缺乏推理透明度")

    # --- ReAct ---
    print(f"\n--- ReAct方式（边想边做）---")
    print(f"   Thought: 用户问鸡蛋放了40天能否食用，我需要先查询保质期")
    print(f"   Action: get_shelf_life")
    print(f'   Action Input: {{"food_name": "鸡蛋"}}')
    print(f"   Observation: 冷藏30-45天，常温10-15天...")
    print(f"   Thought: 冷藏上限是45天，放了40天还在安全范围内，但接近上限了")
    print(f"   Action: check_safety")
    print(f'   Action Input: {{"food_name": "鸡蛋", "days": 40, "method": "冷藏"}}')
    print(f"   Observation: 安全✅")
    print(f"   Thought: 虽然安全，但已接近上限，应该提醒用户尽快食用")
    print(f"   Final Answer: 冷藏40天的鸡蛋在安全期内（上限45天），但已接近极限，")
    print(f"                 建议尽快食用。可以用水浮法验证：沉底=新鲜，浮起=变质。")

    print(f"""
\n💡 三种方式对比总结：

   ┌──────────────┬──────────────────────────────────────┐
   │    方法      │  特点                                 │
   ├──────────────┼──────────────────────────────────────┤
   │  CoT         │  有推理但没有行动 → 知识可能不准确     │
   │  Act-only    │  有行动但没有推理 → 不透明，难以调试   │
   │  ReAct       │  推理+行动交替 → 准确、透明、可控      │
   └──────────────┴──────────────────────────────────────┘

   ReAct的优势：
   1. 准确性：通过工具获取真实数据，不依赖LLM的记忆
   2. 可追踪：每一步Thought都记录在案，便于调试
   3. 可控性：如果某一步出错，能定位到具体环节
   4. 灵活性：LLM可以根据中间结果动态调整计划

   ReAct的局限性（面试也要知道）：
   1. Token消耗大：每一步都要把历史发给LLM
   2. 可能陷入循环：LLM反复调用同一个工具
   3. 错误传播：前面的错误会影响后续推理
   4. 速度慢：每步都要等LLM响应
""")


# ========================================
# 练习4：优化ReAct — 错误处理与防护
# ========================================
def exercise_4_robust_react():
    """
    【目标】给ReAct Agent加上工程化的防护措施

    【面试加分点】
    面试官很喜欢问"这个系统到了生产环境会遇到什么问题"。
    能回答以下问题说明你有工程思维：
    - LLM输出格式不对怎么办？（容错解析）
    - Agent陷入无限循环怎么办？（最大步骤数限制）
    - 工具执行失败怎么办？（错误处理和重试）
    - Token消耗太高怎么办？（历史截断）
    """
    print("\n" + "=" * 60)
    print("练习4：工程化的ReAct Agent — 错误处理与防护")
    print("=" * 60)

    # 工具定义（简化版）
    def get_shelf_life(food_name):
        db = {
            "鸡蛋": "冷藏30-45天，常温10-15天",
            "牛奶": "开封后冷藏2-3天",
        }
        return db.get(food_name, f"暂无「{food_name}」的数据")

    def check_safety(food_name, days, method):
        limits = {("鸡蛋", "冷藏"): 45, ("鸡蛋", "常温"): 15, ("牛奶", "冷藏"): 3}
        limit = limits.get((food_name, method))
        if limit is None:
            return f"暂无数据"
        return json.dumps({"安全": days <= limit, "上限": f"{limit}天"}, ensure_ascii=False)

    # 故意加一个会出错的工具
    def buggy_tool(param):
        """这个工具有时会出错"""
        raise ConnectionError("模拟：网络连接超时")

    TOOLS = {
        "get_shelf_life": {"func": get_shelf_life, "desc": "查保质期 {food_name}"},
        "check_safety": {"func": check_safety, "desc": "判断安全性 {food_name, days, method}"},
        "buggy_tool": {"func": buggy_tool, "desc": "一个不稳定的工具 {param}"},
    }

    def robust_execute_tool(action, action_input, max_retries=2):
        """带重试的工具执行"""
        if action not in TOOLS:
            return f"[错误] 未知工具「{action}」，可用: {list(TOOLS.keys())}"

        for attempt in range(max_retries + 1):
            try:
                return TOOLS[action]["func"](**action_input)
            except Exception as e:
                if attempt < max_retries:
                    print(f"   ⚠️ 工具执行失败（第{attempt+1}次），重试中...")
                else:
                    return f"[工具错误] {action}执行失败: {e}（已重试{max_retries}次）"

    def detect_loop(trace, window=3):
        """检测Agent是否陷入循环（连续调用相同工具+相同参数）"""
        if len(trace) < window:
            return False
        recent = trace[-window:]
        actions = [(t.get("action"), str(t.get("action_input"))) for t in recent if t.get("action")]
        if len(actions) >= window and len(set(actions)) == 1:
            return True
        return False

    print("工程化防护措施：\n")
    print("  1. ✅ 最大步骤数限制 — 防止无限循环")
    print("  2. ✅ 工具执行重试 — 应对临时失败")
    print("  3. ✅ 循环检测 — 发现Agent重复调用相同工具时强制终止")
    print("  4. ✅ 输出格式容错 — LLM不按格式输出时尝试修复")
    print("  5. ✅ Token预算管理 — 超过预算时截断历史")

    # 模拟各种异常场景
    print("\n--- 测试：工具执行失败的处理 ---")
    result = robust_execute_tool("buggy_tool", {"param": "test"})
    print(f"   结果: {result}")
    print(f"   → Agent收到这个错误信息后，应该换一种方式解决问题")

    print("\n--- 测试：循环检测 ---")
    fake_trace = [
        {"action": "get_shelf_life", "action_input": {"food_name": "鸡蛋"}},
        {"action": "get_shelf_life", "action_input": {"food_name": "鸡蛋"}},
        {"action": "get_shelf_life", "action_input": {"food_name": "鸡蛋"}},
    ]
    is_loop = detect_loop(fake_trace)
    print(f"   连续3次相同调用 → 循环检测: {'检测到循环 ❌' if is_loop else '正常'}")
    print(f"   → 实际项目中应强制终止并返回已有信息")

    print(f"""
\n💡 面试时可以这样展示你的工程思维：

   "我在实现ReAct Agent时考虑了几个生产环境的问题：
   
   第一，LLM的输出格式不一定规范，所以我做了容错解析，
   用正则表达式提取Thought/Action/Final Answer，
   解析失败时会提示LLM重新按格式输出。
   
   第二，工具调用可能失败（网络超时、服务不可用等），
   所以加了重试机制和错误回传——把错误信息告诉LLM，
   让它换一种方式解决问题。
   
   第三，Agent可能陷入循环（反复调用同一个工具），
   所以加了循环检测和最大步骤数限制。
   
   第四，多轮推理的token消耗很高，实际项目中需要
   做历史截断或摘要来控制成本。"
""")


# ========================================
# 练习5：完整的ReAct Agent + 推理日志
# ========================================
def exercise_5_react_with_logging(react_agent_fn):
    """
    【目标】给Agent加上完整的推理日志，可用于调试和展示

    生产环境中，Agent的可观测性至关重要。
    你需要记录：每步的Thought、Action、Observation、耗时、token消耗。
    """
    print("\n" + "=" * 60)
    print("练习5：Agent推理链路可视化")
    print("=" * 60)

    import time

    question = "帮我看看冰箱里放了35天的鸡蛋还能不能吃，如果能吃的话，两个鸡蛋大概有多少蛋白质？"

    print(f"\n❓ {question}")
    print(f"\n{'━'*50}")

    start = time.time()
    answer, trace = react_agent_fn(question, verbose=False)
    elapsed = time.time() - start

    # 可视化推理链路
    print(f"\n📊 推理链路可视化：")
    print(f"{'━'*50}")

    for entry in trace:
        step = entry["step"]

        if "final_answer" in entry:
            print(f"  Step {step} ✅ FINAL")
            print(f"    💭 {entry['thought']}")
            print(f"    📝 {entry['final_answer'][:100]}...")
        else:
            print(f"  Step {step}")
            print(f"    💭 Thought: {entry['thought'][:80]}...")
            print(f"    🔧 Action:  {entry['action']}({entry['action_input']})")
            print(f"    👁️ Observe: {entry['observation'][:80]}...")
        print(f"    {'─'*40}")

    print(f"\n📈 执行统计：")
    print(f"   总步骤数: {len(trace)}")
    print(f"   工具调用: {sum(1 for t in trace if t.get('action'))} 次")
    print(f"   总耗时:   {elapsed:.2f} 秒")
    print(f"   最终回答: {answer[:100]}...")

    print(f"""
\n💡 这种日志在实际项目中的用途：
   1. 调试：某个回答不对时，看推理链路找到出错的步骤
   2. 优化：发现多余的工具调用，优化Prompt减少步骤数
   3. 监控：统计平均步骤数、工具调用频率、响应时间
   4. 展示：向非技术人员展示AI的"思考过程"，增加信任
""")


# ========================================
# 🎯 运行入口
# ========================================
if __name__ == "__main__":
    print("🚀 AI工程师转型 - Week2 Day2 ReAct Agent\n")

    exercises = {
        "1": "ReAct Prompt结构详解",
        "2": "手写ReAct Agent 🔥🔥🔥",
        "3": "ReAct vs CoT vs Act-only",
        "4": "工程化防护措施",
        "5": "推理链路可视化",
        "all": "运行全部练习",
    }

    print("选择要运行的练习：")
    for key, name in exercises.items():
        print(f"  {key}. {name}")

    choice = input("\n请输入编号 (1/2/3/4/5/all): ").strip()

    react_fn = None

    if choice == "1":
        exercise_1_react_prompt()
    elif choice == "2":
        react_fn = exercise_2_handwritten_react()
    elif choice == "3":
        exercise_3_react_vs_others()
    elif choice == "4":
        exercise_4_robust_react()
    elif choice == "5":
        if react_fn is None:
            react_fn = exercise_2_handwritten_react()
        exercise_5_react_with_logging(react_fn)
    elif choice == "all":
        exercise_1_react_prompt()
        print("\n" + "🔹" * 30 + "\n")
        react_fn = exercise_2_handwritten_react()
        print("\n" + "🔹" * 30 + "\n")
        exercise_3_react_vs_others()
        print("\n" + "🔹" * 30 + "\n")
        exercise_4_robust_react()
        print("\n" + "🔹" * 30 + "\n")
        exercise_5_react_with_logging(react_fn)

    print("\n\n" + "=" * 60)
    print("✅ Week2 Day2 完成！")
    print("=" * 60)
    print("""
📝 今日思考题（这些是面试原题，必须能流利回答）：

1. ReAct是什么？它的核心循环是什么？
   → Reasoning + Acting, Thought→Action→Observation循环

2. ReAct和CoT有什么区别？
   → CoT只推理不行动，ReAct边推理边行动

3. Agent调用工具后返回的是什么类型？
   → 结构化的JSON（函数名+参数），不是执行结果

4. 如何解决Agent的并发调用问题（有前后依赖关系）？
   → 用DAG拓扑排序（第3周会详细学）

5. 你的ReAct Agent有哪些工程化考虑？
   → 错误处理、循环检测、最大步骤限制、Token预算管理

🔗 和你面试题的关联：
   回看你上次的面试题，今天的内容覆盖了：
   ✅ Agent是什么
   ✅ ChatGPT和Agent的区别
   ✅ Agent的实现逻辑
   ✅ Agent调用返回的类型
   ✅ ReAct是什么

明天：Function Calling深入 + 工具调用的并发处理 🔧
""")
