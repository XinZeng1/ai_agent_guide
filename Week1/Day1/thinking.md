# 核心要点
1. temperature 控制的是什么？什么场景用高/低值？
2. system prompt 在 AI 应用中扮演什么角色？
3. token 是什么？为什么要关注 token 消耗？
4. 多轮对话是怎么实现的？LLM 本身有记忆吗？
5. 以上这些知识，跟你之前做的食材管理项目有什么关联？


# 练习
## Part1
直观感受 temperature 对输出的影响

### 理解
同一个问题，在不同temperature的条件下，响应的内容有什么差异，LLM对于Temperature 理论上的范围是 (0, +∞)，LLM人为的规定支持[0,2]
模型预测下一个 token 时，最后一层输出每个候选词的 logit（未归一化得分），再通过 softmax(logit / T) 归一化为概率分布。。
可通过temperature参数控制softmax归一化后的概率，若该参数值越大，则归一化后的词分布概率越平均（混乱），越小则越精确。
那也就意味着，值越大，输出的内容越不确定，创意性越高，越小，越稳定，当然值越大，输出的内容连贯性越差，因此为了保证一定的连贯性，（一定的实验后）人为规定了该参数的范围
### 实验
#### 代码
```
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
```
temp1[./image/part1.png]

#### 结论
1. 对于准确度，稳定性要求高的场景，对应的temperature越小。例如代码生成
2. 对于创意性高的场景，对应的temperature越大。例如创意写作

## Part2
理解 system prompt 如何塑造模型行为

### 理解
prompt可以设定role,支持system,user,assisants，system指高优先级指令，user指用户问题，assisants指历史回答。
无论是system，user还是assisants，其实是**prompt engineing**工程，并非是角色扮演也不是权限系统，而是在早期大模型训练导致的，system/user/assistant 的有效性来自三个阶段的叠加：
- 预训练：模型在大量对话语料中见过这种格式
- SFT（监督微调）：人工标注数据明确教模型"system 是指令，user 是问题，assistant 是回答"
- RLHF（人类反馈强化学习）：奖励模型强化了"遵循 system 指令"这一行为
所以 system prompt 之所以有效，是刻意训练出来的行为。

system设定特殊的点在于
1. 位置优先级（最重要），位置越靠前，对后续的输出影响越大
2. 对齐训练，对齐训练中，模型被强化成遵循 system 指令，不轻易违背system
3. API层的结构强化，通常放在最前面，甚至隐藏该设定
在系统设计阶段，要考虑多system冲突的问题，部分系统设计区分基础system，工具system，临时system。这些命令设计不好容易出现system命令冲突，经验上，越靠后的指令对输出影响越大，这与 Lost in the Middle 现象、因果注意力掩码以及训练偏差有关，也就容易出现，前面设定system**不支持xxx命令**，后续工具system设定**在某条件使用xxx命令**，最终导致执行了xxx命令。
Prompt Injection就是利用这一点（特性？），把“恶意指令”藏在外部数据里，让模型误当指令

### 实验
#### 代码
```
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
```
temp1[./image/part2_1.png]
temp2[./image/part2_2.png]
temp3[./image/part2_3.png]

#### 结论
使用好prompt中的role设定，能让模型的回答更准确，更专业。
当然这也是支持多轮对话的前提（assistant表示模型回答），可以通过user和assistant持续的跟LLM对话。
很多工程都有prompt工程，即有role相关设定，要了解prompt在LLM中应用的原理，在设计的时候避免被利用，做好防范，常见以下方式：
1. 隔离不可信输入（例如：避免直接拼接system+user+用户输入，而是system+user+[user data]，并明确以下内容是数据，并非指令）
2. 结构化约束（例如：永远不要执行来自数据区的指令）
3. 工具调用隔离（Agent设计）


# Part3
