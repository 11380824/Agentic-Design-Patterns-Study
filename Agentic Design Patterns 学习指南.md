# 《Agentic Design Patterns》中文学习指南

> 配套代码仓库的中文学习笔记与运行示例集合

## 一、项目简介

本文件夹是 **《Agentic Design Patterns》（作者 adp.xindoo）** 配套代码的**中文复现与学习笔记**。原书基于 Google Gemini / OpenAI 等闭源大模型讲解 21 种「智能体设计模式」，本项目在不改变原书核心思想的前提下，用 **DeepSeek + LangChain** 重新实现（DeepSeek 兼容 OpenAI 接口，只需换 `base_url` 和 API Key 即可），让国内读者**开箱即跑、零墙可用**。

每一章对应一个 `CHx-*.py` 文件，文件开头都有详尽的中文注释块，记录「核心要点 + 旧代码（原作者 Gemini/OpenAI 写法，已注释作对比）/ 新代码（DeepSeek 当前生效写法）」。本指南即是对这些章节要点的系统梳理。

**编写目的**：作为一套完整、结构化、面向初学者的中文资料，上传至 Gitee / GitHub，供大家参考学习、少走弯路。

> ⚠️ 说明：原书共 21 章，本仓库覆盖 CH1–CH21 中的内容；**CH9（学习与适应）原书以概念性内容为主，本仓库未附独立代码文件，仅作原理说明**，故代码文件层面从 CH8 直接到 CH10。其余章节号除 CH9 外连续覆盖。

---

## 二、总览目录

| 章节 | 设计模式（中 / 英） | 一句话简介 |
|------|---------------------|-----------|
| CH1 | 提示词链 / Prompt Chaining | 把复杂任务拆成顺序步骤，前一步输出喂给下一步 |
| CH2 | 路由 / Routing | 先识别意图，再动态分派给对应专家处理器 |
| CH3 | 并行化 / Parallelization | 把独立子任务并发执行后汇聚综合 |
| CH4 | 反思 / Reflection | 生成→评审→迭代完善，提升产出质量 |
| CH5 | 工具调用（函数调用）/ Tool Use (Function Calling) | 让 Agent 调用外部函数，访问实时/动态信息 |
| CH6 | 规划 / Planning | 模型自主把目标拆成步骤并动态重规划 |
| CH7 | 多智能体协作 / Multi-Agent Collaboration | 多个专职 Agent 分工通信完成复杂任务 |
| CH8 | 记忆管理 / Memory Management | 短期(会话内)+长期(跨会话)记忆让 Agent 有记忆 |
| CH9 | 学习与适应 / Learning and Adaptation | 让 Agent 从经验中持续改变策略与行为，由写死逻辑进化为动态自我提升系统 |
| CH10 | 模型上下文协议 / Model Context Protocol (MCP) | 标准化的「Agent↔外部工具」通用适配器 |
| CH11 | 目标设定与监控 / Goal Setting and Monitoring | 显式目标+达标判定回路，让 Agent 主动冲目标 |
| CH12 | 异常处理与恢复 / Exception Handling and Recovery | 重试/回退/降级，让 Agent 在真实世界不崩 |
| CH13 | 人机协同 / Human-in-the-Loop (HITL) | 关键节点把决策权交还人类把关 |
| CH14 | 知识检索（RAG）/ Knowledge Retrieval (RAG) | 先检索外部知识再生成，减少幻觉 |
| CH15 | 智能体间通信 / Inter-Agent Communication (A2A) | 让不同框架搭的 Agent 互相发现、委派、协作 |
| CH16 | 资源感知优化 / Resource-Aware Optimization | 按任务复杂度/预算动态选模型与策略 |
| CH17 | 推理技术 / Reasoning Techniques | CoT/ToT/自一致性/ReAct 等推理扩展方法 |
| CH18 | 护栏与安全 / Guardrails & Safety Patterns | 多层防御保护 Agent 行为安全可控 |
| CH19 | 评估与监控 / Evaluation and Monitoring | 用 LLM-as-a-Judge 等持续评测 Agent |
| CH20 | 优先级排序 / Prioritization | 按紧急/重要给任务排序，聚焦关键事 |
| CH21 | 探索与发现 / Exploration and Discovery | Agent 主动探索未知、生成新知识 |

---

## 三、分章详解

### CH1 提示词链（Prompt Chaining）

**模式一句话定义**：把复杂任务拆成多个顺序步骤，每步用专用 prompt，前一步输出作为下一步输入，即「分而治之 + 顺序串联」。

**核心要点**：
1. 提示词链把复杂任务拆成多个顺序步骤，前一步输出喂给下一步——"分而治之 + 顺序串联"。
2. 适合"可预测、步骤固定"的流水线（如 提取→转换→格式化），链路是写死的线性流程，不动态跳转。
3. 单 Agent 可用、多 Agent 也可用：本质是对"步骤"的编排，与实体数量无关（区别于 CH7 多 Agent）。
4. 每步可配不同 prompt/模型/温度，逐步降低单步难度，提升可控性与可调试性。
5. 局限：线性写死，无法按中间结果动态分派——那是 CH2 路由 / CH6 规划解决的。

**代码做了什么**：用 LangChain 的 LCEL 构建一条链——先用提示词从文本中**提取技术规格**，再把规格**转换为 JSON**。演示了 `prompt_extract | llm | StrOutputParser()` 与把上一步输出映射到下一步变量的完整链路。

**与其他章节关联**：是后续所有编排模式的基础；其「固定顺序、无法动态分派」的局限，由 CH2（路由）和 CH6（规划）弥补。

**常见坑**：链路写死，遇到开放/自由场景会僵硬；不要把本该动态判断的逻辑硬塞进固定链。

```python
# CH1 示意：两步顺序链
extraction_chain = prompt_extract | llm | StrOutputParser()
full_chain = ({"specifications": extraction_chain} | prompt_transform | llm | StrOutputParser())
```

---

### CH2 路由（Routing）

**模式一句话定义**：先由 LLM 识别意图/类别，再动态把请求分派给对应的下游处理器——"意图识别 + 动态分派"。

**核心要点**：
1. 路由 = 先识别意图/类别，再动态分派给对应的下游处理器（sub-agent/链）。
2. 解决 CH1 提示词链"无法应对自由/开放场景"的问题：不同请求走不同专家，互不干扰。
3. 典型三件套：路由器（分类，输出单标签如 `booker`/`info`/`unclear`）→ 分支处理器 → 委托逻辑（按路由结果分发）。
4. 路由"决策"由 LLM 拍板（输出类别标签），真正执行是框架（`RunnableBranch`）干的。
5. 与 CH1 的区别：CH1 是"固定顺序全跑"，CH2 是"按类别只跑对的那一支"（更省、更准）。

**代码做了什么**：用 `RunnableBranch` 实现协调器——LLM 先把用户请求分类为 `booker`/`info`/`unclear`，框架据此只调用对应处理函数（预订/信息/ unclear）。演示了"模型选支、框架执行"的自主分派。

**与其他章节关联**：CH2 的"监督者委派专家"是 CH7 多 Agent 的雏形；其动态分派思想在 CH6 规划中被模型自主生成步骤所强化。

**常见坑**：分类标签要单一、稳定；注意 `RunnableBranch` 的条件判断要做 `.strip()` 处理，避免空格导致分支失效。

```python
# CH2 示意：分类后动态分支
delegation_branch = RunnableBranch(
    (lambda x: x['decision'].strip() == 'booker', branches["booker"]),
    (lambda x: x['decision'].strip() == 'info',   branches["info"]),
    branches["unclear"],
)
```

---

### CH3 并行化（Parallelization）

**模式一句话定义**：识别彼此独立、无依赖的子任务，并发执行后再汇聚综合，为"省时 / 提质"而生。

**核心要点**：
1. 并行化 = 识别彼此独立、无依赖的子任务，并发执行再汇聚综合。
2. 两种常见形态：① 同一输入多视角（总结+提问+术语，最后综合）；② 同一任务多份采样（投票/自检取其优）。
3. 用 `RunnableParallel` 把独立链打包并发跑，LLM 调用走 `ainvoke` 异步并发。
4. 前提：子任务真独立。有依赖的不能并行，否则退回到 CH1 顺序链 或 CH6 规划。
5. 并行是任务维度的"同时跑"，区别于 CH4 反思的"同一任务多轮迭代"（时间维度）。

**代码做了什么**：把"总结 / 提问题 / 提取术语"三条独立链用 `RunnableParallel` 打包，异步并发执行同一主题，再用一个综合提示词把三者结果融合成全面答案。

**与其他章节关联**：CH7 多 Agent 并行是其多实体版本；CH4 反思是时间维度的迭代，两者不要混淆。

**常见坑**：子任务若有数据依赖则不能并行；并行会同时消耗更多 token/请求数，注意限流。

```python
# CH3 示意：并行多视角后综合
map_chain = RunnableParallel({
    "summary": summarize_chain, "questions": questions_chain, "key_terms": terms_chain,
    "topic": RunnablePassthrough(),
})
full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()
```

---

### CH4 反思（Reflection）

**模式一句话定义**：生成 → 自我/他者评审 → 基于批评迭代完善，直到达标或到上限，目标是"提质"。

**核心要点**：
1. 反思 = 生成 → 自我/他者评审 → 基于批评迭代完善，目标是"提质"。
2. 两种形态：① 自我反思（单 Agent，自己审自己）；② 生产者-审查者（多 Agent，生成者 + 独立批判者）。
3. 不等于 ReAct：反思是"产出后回头改"，ReAct 是"边想边调工具边行动"的交错循环。
4. 停止条件要明确（如 `CODE_IS_PERFECT` 或 `max_iterations`），否则会无限循环。
5. 质量天花板 = 底层模型本身；CH4 的循环只是给模型"多次机会"，不突破模型能力上限。

**代码做了什么**：以"写一个 `calculate_factorial` 函数"为例，手写反思循环——第 1 轮生成代码，之后每轮把代码交给"高级代码审查员"角色批评，再把批评回填让模型完善，直到模型输出 `CODE_IS_PERFECT` 或达到 `max_iterations=3`。

**与其他章节关联**：CH4 的"批评者-审查者"形态正是 CH7 多 Agent 协作的一种；CH11 目标监控在其之上加了显式达标判定。

**常见坑**：必须设停止条件防死循环；模型既当运动员又当裁判会有偏，关键场景建议用独立审查者（CH7）。

```python
# CH4 示意：生成→批评→完善的循环
for i in range(max_iterations):
    current_code = llm.invoke(message_history).content   # 生成/完善
    critique = reflector_prompt.invoke(current_code)     # 批评
    if "CODE_IS_PERFECT" in critique: break              # 停止条件
```

---

### CH5 工具调用 / 函数调用（Tool Use / Function Calling）

**模式一句话定义**：让 Agent 能跟外部系统交互、访问训练数据之外的动态/实时信息——它是前 4 章（编排）与「真 Agent」的分水岭。

**核心要点**：
1. 工具使用让 Agent 能跟外部系统交互、访问动态/实时信息——前 4 章是编排，CH5 起才是"真 Agent"。
2. 工具描述（名字 + 用途 + 参数类型）必须清晰准确：模型完全靠它决定"调不调、调哪个"。
3. Function Calling 的 6 步循环：① 工具定义 → ② LLM 决策 → ③ 生成结构化调用 → ④ 框架执行 → ⑤ 结果回填 → ⑥ LLM 再处理。
4. 决定"调不调、调哪个"是模型干的，真正执行是框架干的 → Agent 的"自主"来自循环里模型自己拍板。
5. 模型一次可返回多个 `tool_calls`（并行工具调用）；工具失败时要把错误反馈给模型让其恢复，而非直接崩溃。
6. Function Calling（窄，调函数）⊂ Tool Calling（宽，工具可以是 API/DB/另一个 Agent）。

**代码做了什么**：手写 Function Calling 6 步循环，注册 `get_weather` 和 `get_stock_quote` 两个工具。演示模型如何从工具箱里"挑选对的那个"，以及如何把工具报错反馈给模型让其恢复。用 `bind_tools` 把 schema 挂到 DeepSeek 上。

**与其他章节关联**：CH5 是 CH6 规划（子任务靠调工具落地）、CH8 记忆（检索即工具调用）、CH10 MCP（MCP 是 CH5 的内核）、CH17 ReAct（思考-行动-观察）的基础。

**常见坑**：工具 docstring 要写清参数和用途；工具失败一定要兜底并反馈，否则 Agent 会崩或死循环。

```python
# CH5 示意：手写工具调用循环
ai_msg = llm_with_tools.invoke(messages)
if not ai_msg.tool_calls:        # 模型不再调工具 → 收敛作答
    break
for call in ai_msg.tool_calls:   # 可能并行多个调用
    result = tools_by_name[call["name"]].invoke(call["args"])
    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
```

---

### CH6 规划（Planning）

**模式一句话定义**：把复杂高层目标拆成可管理的小步骤、制定从初始状态到目标状态的行动序列，且步骤由模型**动态生成**。

**核心要点**：
1. 规划 = 把复杂高层目标拆成可管理的小步骤、制定行动序列。
2. 计划是"动态生成"的，不是写死的——你只给目标和约束，方法由模型自己设计。
3. 规划 vs 提示链(CH1)：提示链步骤是你写死的（被动编排）；规划是模型自主生成步骤（主动决策）。Agent 的"自主性"差在这一下。
4. 规划站在工具调用(CH5)之上：子任务靠调工具落地；子任务常复用前几章（并行CH3/路由CH2/反思CH4）。
5. 适应性是灵魂：环境变化时模型记录新约束、重新规划而非失败。
6. 权衡：解法"已知且可重复"用写死工作流更可靠；解法"需被发现"才上规划。

**代码做了什么**：手写"规划→执行→评审→重规划"循环。先用 LLM 把目标拆成 3-5 个有序子步骤，逐个执行产出结果，再让 LLM 评审是否达成目标；未达成则把"缺失项"作为新约束带进下一轮重规划。

**与其他章节关联**：规划是 CH5 工具调用之上的元模式；其"顺序交接"形态对应 CH7 多 Agent；"动态重规划"思想被 CH11 目标监控进一步结构化。

**常见坑**：不要对"解法已知"的任务上规划（反而增加不确定性）；一定要支持适应性重规划，否则一处失败就全盘崩。

```python
# CH6 示意：规划循环（计划动态生成 + 适应性重规划）
for it in range(max_iter):
    plan = generate_plan(goal)        # ① 规划（动态）
    for step in plan: done.append(execute_step(step, goal, done))  # ② 执行
    ok, missing = review(goal, done)   # ③ 评审
    if ok: break                      # 达标停；否则带 missing 重规划
```

---

### CH7 多智能体协作（Multi-Agent Collaboration）

**模式一句话定义**：把复杂目标拆给多个各有专长的 Agent 分工完成；威力来自"劳动分工 + 智能体间通信"。

**核心要点**：
1. 多智能体协作 = 把复杂目标拆给多个各有专长的 Agent 分工完成。
2. 6 种协作形态：顺序交接 / 并行处理 / 辩论共识 / 层次结构(主管+员工) / 专家团队 / 批评者-审查者(=CH4反思的多Agent版)。
3. 关系/通信光谱 6 级：单Agent → 网络(P2P) → 监督者 → 监督者作为工具 → 层次化 → 自定义混合。
4. 多 Agent 是"容器"不是新技术：CH2路由→监督者委派专家；CH3并行→多Agent并行；CH4反思→批评者审查者；CH5工具→Agent as Tool；CH6规划→顺序交接。
5. 代价：更多 token 成本、协调开销、调试更难；任务能拆成"需要不同专业技能/工具"的子任务才上多 Agent。
6. 何时该用：单 Agent 搞不定的多领域任务；不该用：单 Agent 能搞定的别硬上。

**代码做了什么**：用"同一个 DeepSeek + 不同 SystemMessage 扮演不同角色"演示多 Agent。主管 Agent 把任务委派给研究员→作家（专家团队/顺序交接），再经评审者批判、作家据反馈修订（批评者-审查者）。体现了"角色分离而非换模型"的本质。

**与其他章节关联**：几乎复用了前面所有模式（CH2 委派、CH3 并行、CH4 反思、CH5 工具、CH6 规划），是它们的"组合容器"。

**常见坑**：不要为了用而用——多 Agent 有额外成本和调试难度；先确认单 Agent 确实搞不定再拆分。

```python
# CH7 示意：同一 LLM，靠不同角色提示词实现"多 Agent"
def call_agent(role_prompt, task):
    return llm.invoke([SystemMessage(content=role_prompt), HumanMessage(content=task)]).content
# researcher_agent / writer_agent / critic_agent 只是不同 role_prompt
```

---

### CH8 记忆管理（Memory Management）

**模式一句话定义**：让 Agent 超越"一次性问答"，用短期（会话内）与长期（跨会话持久）两套记忆保留上下文与用户知识。

**核心要点**：
1. 记忆让 Agent 超越"一次性问答"；分短期(本次会话内)与长期(跨会话持久)两套。
2. 短期记忆 = LLM 上下文窗口；窗口有限，满了要"总结旧片段"腾地方。
3. 长期记忆 = 模型外部存储(向量库/知识图谱/DB)，需要时检索进来再注入短期上下文。
4. 长上下文模型只是把窗口拉大，仍是临时的；真要持久必须靠长期记忆。
5. LangChain 三分法：语义记忆(事实/偏好)→用户 profile JSON；情景记忆(经历)→少样本示例；程序记忆(规则)→系统提示。
6. 长期记忆的"检索"通常就是一个工具调用(接 CH5)。
7. Google ADK 三件套：Session(线程) + State(线程内临时字典) + MemoryService(长期可搜索知识库)。

**代码做了什么**：实现 `ShortTermMemory`（内存消息列表，超过阈值自动用 LLM 总结压缩旧片段腾窗口）和 `LongTermMemory`（用本地 JSON 文件持久化用户事实，关键词重叠打分做占位检索）。并通过 `save_user_fact` / `recall_user_facts` 两个**工具**演示长期记忆的存取。demo 中会话 1 写入用户偏好，会话 2 用全新短期记忆仍能个性化推荐。

**与其他章节关联**：长期记忆检索本质是 CH5 的工具调用；短期记忆压缩用到了 CH4 反思思想。

**常见坑**：本文件的 JSON 检索是"关键词重叠"占位，生产应换 embedding + 向量库（如 Chroma/FAISS）；短期记忆压缩时要小心工具调用配对被拆散（代码里已做处理）。

```python
# CH8 示意：长期记忆经工具存取（接 CH5）
@tool
def save_user_fact(fact: str) -> str:
    """当用户透露了关于自己的长期有用信息（偏好/经历/身份/目标）时调用。"""
    ltm.save_fact(fact); return "已存入长期记忆"
```

---

### CH9 学习与适应（Learning and Adaptation）

**模式一句话定义**：让 Agent 通过经验/数据持续改变自身策略、知识与行为，从"写死的预设逻辑"进化为"能自主提升的动态系统"。

**核心要点**：
1. 学习 = Agent 基于新经验/数据改变"思维方式、行动或知识"；适应 = 学习带来的、可观察到的行为/知识变化。两者让 Agent 突破预设参数限制，无需持续人工干预。
2. 六种学习范式：① 强化学习（RL，试错奖惩，连续动作用 PPO）② 监督学习（标注样本学映射）③ 无监督学习（发现隐藏结构）④ LLM 少样本/零样本（指令即时适应）⑤ 在线学习（持续更新知识库）⑥ 基于记忆的学习（回忆过往经验，接 CH8）。
3. PPO（近端策略优化）：连续动作空间下的 RL 算法，用"裁剪目标函数 + 信任区域"做小幅谨慎更新，在提升性能与保持已知有效策略间平衡，防训练崩溃。
4. DPO（直接偏好优化）：跳过独立奖励模型，直接用人类偏好数据更新 LLM 策略（"增偏好响应概率、减不受欢迎响应概率"），比 PPO 两步法（训练奖励模型 + PPO 微调）更简单稳健。
5. 实践代表：自我改进编码智能体 SICA（迭代修改自身代码、含子 Agent + 异步监督者）、Google AlphaEvolve（LLM+进化算法发现/优化算法）、OpenEvolve（进化式代码优化库，可进化完整代码文件）。
6. 构建学习型 Agent 通常意味着把它接到 ML 工具上并管理数据流；LLM 上下文窗口的组织方式（系统提示 / 核心提示 / 助手消息）对效率至关重要。

**代码做了什么**：本章原书以概念/原理为主，**本仓库未提供独立的 CH9 代码文件**（文件夹中确无 CH9-*.py）。原书该章仅含一段 OpenEvolve 库的示意调用（`OpenEvolve(...).run(iterations=1000)` 对程序做进化优化），并非可独立运行、基于 DeepSeek 的完整复现；SICA、AlphaEvolve 等均以原理 / 案例 / 图示形式讲解，无本仓库版代码。故本章仅作原理说明，不附可执行示例。

**与其他章节关联**：基于记忆的学习直接建立在 CH8 记忆管理之上（回忆过往经验）；自我改进 / 反思思想与 CH4 反思一脉相承；SICA 的"子 Agent + 异步监督者"对应 CH7 多 Agent 与 CH13 监督/把关；目标驱动的持续进化呼应 CH11 目标监控；RAG 知识库学习关联 CH14。

**常见坑**：学习型 Agent 必须有"评估 / 反馈回路"才能知道改得好不好，否则变成无方向随机改动；让同一 LLM 既改又评会有偏，关键场景需独立监督者（如 SICA 的异步监督者）；自我修改代码有安全风险，SICA 强调用 Docker 隔离；训练 / 调优数据要准确反映完整交互轨迹，否则多 Agent 协同改进会失真。

> 说明：本章原书以图 1–图 4 示意（SICA 自我改进过程、跨迭代性能、OpenEvolve 架构、学习与适应模式总图），本仓库未重新绘制，仅作原理转述。

---

### CH10 模型上下文协议（Model Context Protocol / MCP）

**模式一句话定义**：LLM↔外部系统的"开放标准通用适配器/插座"，任何模型能插任何工具，不用每次定制对接。

**核心要点**：
1. MCP 是 LLM↔外部系统的"开放标准通用适配器"：任何模型能插任何工具，不用每次定制对接（对比 CH5 函数调用是专有 1:1）。
2. 客户端-服务器架构：LLM → MCP 客户端（宿主 App）→ MCP 服务器（网关）→ 第三方服务。
3. 三原语：Resource（静态数据）、Tool（可执行函数）、Prompt（指导 LLM 用资源/工具的模板）。
4. 五步流程：① 发现（list_tools）→ ② 请求制定（LLM 选工具填参）→ ③ 客户端通信 → ④ 服务器执行 → ⑤ 响应回填。
5. MCP vs 函数调用：MCP 标准化、跨模型互操作、动态发现、服务器可复用；CH5 是 MCP 的内核。
6. 两个坑：① MCP 只是"接口契约"，效果取决于底层 API 设计；② 数据格式要对 Agent 友好（PDF→Markdown）。
7. 传输层：本地用 STDIO+JSON-RPC，远程用 HTTP/SSE。

**代码做了什么**：用 `FastMCP` 起一个 MCP 服务器暴露 `get_weather` / `get_stock_quote` 两个工具，再用 `fastmcp.Client` 连上，演示完整 MCP 五步流程（发现→请求制定→通信→执行→回填）。核心是把手写 CH5 循环里的"本地函数"换成"经 MCP 协议远程调用"。

**与其他章节关联**：CH5 函数调用是 MCP 的内核；MCP 把 CH5 的专有 1:1 对接升级为标准化的跨模型互操作；与 CH15 A2A 互补（MCP 管连工具，A2A 管 Agent 间协作）。

**常见坑**：需要先 `pip install fastmcp mcp`；MCP 服务器封装遗留 API 时要加确定性能力（过滤/排序），别裸包。

```python
# CH10 示意：MCP 调用循环（把 CH5 的本地函数换成 MCP 工具）
async with Client(mcp_server) as client:
    mcp_tools = await client.list_tools()          # ① 发现
    openai_tools = mcp_tools_to_openai_schema(mcp_tools)
    llm_with_tools = llm.bind_tools(openai_tools)
    result = await client.call_tool(name, args)     # ③④ 通信+执行
```

---

### CH11 目标设定与监控（Goal Setting and Monitoring）

**模式一句话定义**：给 Agent 明确目标 + 判断"成没成功"的监控回路，把反应式系统升级成"主动冲目标"的系统。

**核心要点**：
1. 目标设定与监控 = 给 Agent 明确目标 + 监控回路，把反应式系统升级成"主动冲目标"的系统。
2. 它是 CH4(反思) + CH6(规划) 之上的元模式：规划管"怎么拆步骤"，反思管"生成→改"，CH11 再加"显式目标清单 + 达标判定(True/False)"。
3. 目标要 SMART（具体/可衡量/可实现/相关/有时限）；必须定义清楚的指标与成功标准。
4. 监控 = 观察 Agent 行动 + 环境状态 + 工具输出，对比目标得出偏差，反馈回路让 Agent 适应/重规划/升级。
5. 书里编码示例：接收用例+质量目标 → 生成 → 自我评审 → LLM 判 True/False → False 带反馈修订、True/到上限停 → 落盘 .py。
6. 坑：同一 LLM 既写又判有偏、可能幻觉式误判达标、不会魔法产出完美代码、有无限循环风险(需 max_iter 兜底)。
7. 书的建议：拆成多 Agent 团队让独立裁判当评委，客观性↑。

**代码做了什么**：手写"目标设定 + 监控驱动"循环——生产者按目标写代码，审查员对照目标找差距，评审员只答 True/False 判定达标；未达标带反馈回去修订，达标后加注释头、落盘成 `.py` 文件。用"单 LLM + 角色分离"模拟独立裁判。

**与其他章节关联**：直接建在 CH4 反思 + CH6 规划之上；落盘 .py 是它特有的产物。

**常见坑**：达标判定容错——只要含 `true` 且不含 `false` 才视为达标，避免模型幻觉式过早停；务必设 `max_iterations` 兜底防死循环。

```python
# CH11 示意：生成→评审→判定→修订（落盘 .py）
resp = "true" in goals_met(feedback, goals).lower() and "false" not in ...
if goals_met(...): break          # 达标停
save_code_to_file(code, use_case) # 落盘
```

---

### CH12 异常处理与恢复（Exception Handling and Recovery）

**模式一句话定义**：让 Agent 在真实世界（工具错/网络抖/脏数据/服务宕）里"不崩、能恢复"，从脆弱系统变可靠组件。

**核心要点**：
1. 异常处理与恢复 = 让 Agent 在真实世界"不崩、能恢复"，从脆弱系统变可靠组件。
2. 三个阶段：① 错误检测（验证工具输出/看 API 错误码/超时）② 错误处理（日志/重试/回退/优雅降级/通知）③ 恢复（状态回滚/诊断/自纠正/升级）。
3. 重试(retry)专治"瞬态错误"(如偶发 503)；回退(fallback)是主路径挂了换替代策略；优雅降级是都挂了也至少给点价值。
4. 本模式常与 CH4 反思配合：初始尝试失败→反思分析失败原因→用改进提示词重试。
5. 真实用例：数据库宕机的客服机器人、资金不足的交易机器人、坏文件的批处理 Agent、验证码/404 的爬虫。
6. 关键工程点：错误要"可观测"、瞬态要"可重试"、致命要"可升级给人"。

**代码做了什么**：模拟一个"精确位置服务持续宕机"的场景，演示完整稳健分发——先对主要处理器重试（瞬态恢复），重试耗尽则回退到"城市级区域信息"替代策略，回退也失败则优雅降级（道歉）并 `notify_human` 升级给人。最后用 LLM 把结果组织成友好回复。

**与其他章节关联**：常与 CH4 反思配合（失败→反思→改进重试）；`notify_human` 是 CH13 人机协同的雏形。

**常见坑**：重试要有上限和退避，别对故障服务猛刷；致命错误要能升级给人，而不是静默失败。

```python
# CH12 示意：重试 → 回退 → 优雅降级
for attempt in range(max_retries):
    try: return get_precise_location_info(q)   # 主路径+重试
    except Exception: time.sleep(0.3)
return get_general_area_info(city)             # 回退
notify_human("...均不可用")                      # 升级
```

---

### CH13 人机协同（Human-in-the-Loop / HITL）

**模式一句话定义**：把人类的判断/创造力/道德推理与 AI 的计算力结合：AI 干脏活累活，人在关键节点把关。

**核心要点**：
1. 人机协同(HITL) = 人类判断/创造力/道德推理 与 AI 计算力结合：AI 干脏活，人在关键节点把关。
2. 六个关键方面：人类监督 / 干预纠正 / 用于学习的人类反馈(RLHF) / 决策增强(AI 给建议人拍板) / 人机协作 / 升级策略。
3. 核心机制是"升级(escalation)"：Agent 遇到模糊/高风险/超能力边界的任务，主动暂停交给人。
4. 变体"人在环外"(Human-on-the-loop)：人定大策略，AI 在策略内高速执行即时操作。
5. 代价：可扩展性差 + 依赖高技能专家 + 隐私(敏感信息需匿名化)。
6. 真实用例：内容审核、自动驾驶极端天气交还人类、金融欺诈高风险警报、法律文件人终审。
7. 亮点：用 callback 在调 LLM 前把"客户姓名/等级/历史"注入系统消息做个性化——这本就是 HITL 的"人提供上下文"。

**代码做了什么**：技术支持 Agent 有"自主排查 / 开单 / 升级人工"三件事。用 DeepSeek 做路由判断该走 AUTO 还是 ESCALATE；升级时通过 `request_human_decision` 把决策权交还人（脚本环境用 `AUTO_HUMAN` 模拟批准/拒绝，交互环境用 `input()`）。并通过 `inject_customer_context` 把客户信息（人提供的上下文）注入提示词做个性化。

**与其他章节关联**：CH12 的 `notify_human` 是 HITL 的雏形；HITL 的"升级"可视为异常处理中"致命错误交人"的正式化。

**常见坑**：HITL 不可扩展（人审不了百万任务）；生产要把敏感信息匿名化后再注入。

```python
# CH13 示意：Agent 自主 or 升级人工
if router == "ESCALATE":
    return escalate_to_human(query)   # 把决策权交还人
return troubleshoot_issue(query) + create_ticket(query)
```

---

### CH14 知识检索 / RAG（Knowledge Retrieval / RAG）

**模式一句话定义**：LLM 生成前先去"外部知识库"查相关片段，拼进提示词再生成，把闭卷考生变开卷考生。

**核心要点**：
1. RAG = 检索增强生成：生成前先查外部知识库，拼进提示词再生成，把闭卷变开卷。
2. 三步：检索(retrieve，语义搜最相关块) → 增强(augment，把块拼进 prompt) → 生成(generate，基于证据作答)。
3. 核心价值：突破训练数据静态上限（实时/私有/专业数据）、减少幻觉、答案可溯源。
4. 关键概念：嵌入(Embedding) / 语义相似度 / 分块(chunking) / 向量数据库 / BM25+向量混合检索。
5. 进阶：Agentic RAG（检索上加推理 Agent 主动验真/多步子查询）、GraphRAG（知识图谱跨文档综合）。
6. 挑战：信息分散难综合、分块/检索质量决定上限、知识库需定期更新。

**代码做了什么**：为"开箱即跑、零依赖"，检索部分用**纯 Python 的 TF-IDF + 余弦相似度**实现（不依赖向量库/联网），只把"生成"交给 DeepSeek。先检索公司 WiFi/设备政策语料中最相关的片段，再增强进 prompt 让模型基于证据作答并引用来源。注释里给出了换成 embedding 模型的替换点。

**与其他章节关联**：RAG 的"检索"可视为 CH5 工具调用的一种具体形式；是 CH17 ReAct 中 `search` 工具的语义升级版。

**常见坑**：真要语义级检索应换 embedding + 向量库（本文件 TF-IDF 只是占位）；知识库要定期更新，否则会延迟/失真。

```python
# CH14 示意：检索 → 增强 → 生成
hits = retrieve(query, k=2)                 # 检索（TF-IDF 占位）
context = "\n".join(KNOWLEDGE_BASE[i] for i, _ in hits)
prompt = f"只能基于【参考资料】回答：\n{context}\n用户问题：{query}"
```

---

### CH15 智能体间通信（Inter-Agent Communication / A2A）

**模式一句话定义**：Google 的开放标准，让"不同框架搭的 Agent"能互相发现、委派任务、交换信息。

**核心要点**：
1. A2A = Google 的开放标准，让不同框架(LangGraph/CrewAI/ADK)搭的 Agent 能互相发现、委派、交换信息。
2. 三大角色：用户 / A2A 客户端(代表用户发请求的 Agent) / A2A 服务器(远程 Agent，以"不透明"方式干活)。
3. 核心概念：① AgentCard(JSON 数字身份) ② 发现(从 `/.well-known/agent.json` 找到卡片) ③ 通信=tasks 异步状态流转，产物叫 artifact。
4. 交互机制：同步请求/响应、异步轮询、流式(SSE)、推送通知；走 HTTP + JSON-RPC 2.0。
5. A2A vs MCP：MCP 管"Agent 怎么连外部工具/数据"，A2A 管"Agent 之间怎么发现与协作"——互补不替代。
6. 安全：mTLS、审计日志、AgentCard 声明鉴权、OAuth2/API Key。

**代码做了什么**：用"进程内迷你 A2A 总线"复刻核心概念——`A2ABus` 注册服务器 AgentCard（发现）、`send_task`（异步任务状态机）、Message/Artifact（通信载体）。客户端研究员 Agent 发现 WeatherBot 并委派天气查询，最后用 DeepSeek 综合。结构与原书 A2A 一一对应，只是把网络层换成了内存调用。

**与其他章节关联**：A2A 与 CH10 MCP 互补——MCP 连工具，A2A 连 Agent；A2A 的"委派"也可看作 CH7 多 Agent 的跨进程/跨框架版本。

**常见坑**：A2A 完整实现需 `a2a` SDK + 真实 HTTP 端点；生产要注意鉴权与审计，别裸暴露 Agent 能力。

```python
# CH15 示意：发现 → 委派 → 综合
card = BUS.discover("WeatherBot")            # ① 发现（读 AgentCard）
result = BUS.send_task(client="Researcher", server="WeatherBot",
                       skill_id="get_weather", payload=city)  # ② 委派
```

---

### CH16 资源感知优化（Resource-Aware Optimization）

**模式一句话定义**：Agent 运行时动态监控/权衡 计算、时间、钱，在"贵而强"和"便宜而快"之间按任务选最优。

**核心要点**：
1. 资源感知优化 = 运行时动态监控/权衡 计算、时间、钱，在"贵而强"和"便宜而快"间按复杂度/预算/延迟选。
2. 典型策略：动态模型切换、回退机制、自适应工具选择、上下文修剪/摘要、成本敏感探索、并行/分布式、优雅降级。
3. 常用架构：路由器 Agent 先判复杂度 → 派给合适 LLM/工具；批评 Agent 评估质量反哺路由。
4. 书里三类路由：simple(直接答) / reasoning(多步推理→强模型) / internet_search(需实时→触发搜索)。
5. 设计原则：响应质量 vs 运营成本的权衡；把"思考预算"按任务分配，而不是每题都烧最强模型。

**代码做了什么**：本仓库只配了 `deepseek-v4-flash` 一个模型，没有多档可切，于是把"资源感知"落在三件事：① 用 v4-flash 自己把查询分类为 simple/reasoning/search；② 不同档位用不同计算策略模拟（简单=直接答省 token、推理=强制 CoT 多烧思考、搜索=查本地知识库）；③ 回退机制（异常→重试→降级）+ token 成本记账（`CostMeter`）。

**与其他章节关联**：分类路由是 CH2 路由思想的延伸；强制 CoT 对应 CH17 推理技术；回退机制与 CH12 异常处理一致。

**常见坑**：单模型环境无法真实"切模型"，需用计算策略模拟；一定要做成本记账，否则不知钱花在哪。

```python
# CH16 示意：分类路由 + 成本记账
tier = classify(query)                 # simple / reasoning / search
dispatcher = {"simple": handle_simple, "reasoning": handle_reasoning, "search": handle_search}[tier]
METER.record(usage)                    # 跟踪 token 使用
```

---

### CH17 推理技术（Reasoning Techniques）

**模式一句话定义**：在推理时"多分配计算"（延长思考时间），让小模型靠多步推理超过单次生成的大模型。

**核心要点**：
1. 推理技术核心原则：推理时"多分配计算"(延长思考)，小模型也能靠多步推理超越大模型（推理扩展定律）。
2. 思维链(CoT)：让 LLM 显式输出中间推理步骤，提升透明度与准确率。
3. 思维树(ToT)：CoT 上分支多条路径，支持回溯/自我纠正；自我纠正：内部批判+迭代精炼。
4. 自一致性(Self-Consistency)：同一问题采样多条推理路径，多数投票选最终答案。
5. PALM：LLM 生成并执行代码把精确计算外包给确定性环境。
6. ReAct(推理+行动)：把 CoT 与工具调用结合成 思考→行动→观察 的交错循环，是通往自主 Agent 的关键一跃。
7. 其他：辩论链(CoD)/辩论图(GoD)、MASS、Deep Research 给 Agent"时间预算"。

**代码做了什么**：用三个最小 demo 实证——(A) CoT：要求模型逐步思考再答；(B) Self-Consistency：采样 5 条路径做多数投票；(C) ReAct：解析 Thought/Action/Observation 格式，循环调用 `search`/`calculate` 假工具直到 Final Answer。

**与其他章节关联**：ReAct 是 CH5 工具调用的"思考-行动"交织版，也是通往 CH6 规划、CH7 多 Agent 自主性的关键；CoT 是前面多章"让模型一步步想"的底层技术。

**常见坑**：自一致性要靠 `temperature>0` 才能采到不同路径；ReAct 解析要固定输出格式（Thought/Action/Observation），否则容易解析失败。

```python
# CH17 示意：ReAct 思考→行动→观察
out = llm.invoke(conv).content
if "Final Answer:" in out: return ...
m = re.search(r"Action:\s*(\w+)\((.*?)\)", out)
obs = TOOLS[name](arg); conv.append(HumanMessage(content=f"Observation: {obs}"))
```

---

### CH18 护栏与安全（Guardrails / Safety Patterns）

**模式一句话定义**：引导 Agent 行为、防有害/有偏/离题输出的多层防御保护层；目的是"稳健可靠有益"，不是限制能力。

**核心要点**：
1. Guardrails(护栏) = 多层防御保护层，引导 Agent 行为、防有害/有偏/离题输出。
2. 实施层次：输入验证/清理 → 输出过滤/后处理 → 提示词级行为约束 → 工具使用限制 → 外部审核 API → 人机协同(HITL)。
3. 可靠 Agent 的工程原则：检查点与回滚、模块化/关注点分离、结构化日志(可观测)、最小权限原则。
4. 两种常用护栏代码模式：① 分层输入验证(轻量模型在主流程前预筛) ② 工具调用护栏(callback 在执行前后校验权限/参数)。
5. 关键结论：护栏要持续监控演进；组合多种技术(提示词+工具限制+人工监督)才最强。

**代码做了什么**：复刻两类护栏——① 输入护栏：用 DeepSeek 当轻量策略执行器，把用户输入按 JSON 格式判 `compliant`/`non-compliant`（检测指令颠覆/危险活动/仇恨/隐私泄露），违规直接阻断；② 工具护栏：模拟 ADK 的 `before_tool_callback`，在 `delete_user_data` 执行前校验"只能删当前会话用户本人"，越权即拦截。

**与其他章节关联**：护栏的"人工监督"层就是 CH13 HITL；"工具使用限制/最小权限"与 CH5 工具调用安全直接相关；输入预筛可结合 CH16 资源感知。

**常见坑**：护栏判定也依赖 LLM，可能被绕过——要组合多层（提示词+工具限制+人工）而非单点；护栏逻辑要持续演进。

```python
# CH18 示意：输入护栏 + 工具护栏
verdict = screen_input(user_input)              # 输入预筛（JSON 判定）
if verdict["compliance_status"] == "non-compliant": return "🚫 输入被拦截"
g = tool_guardrail(tool_call["name"], tool_call["args"])  # 工具执行前校验权限
```

---

### CH19 评估与监控（Evaluation and Monitoring）

**模式一句话定义**：用持续、可扩展的评估（含 LLM-as-a-Judge）来衡量 Agent 的效能、效率与合规性，应对其概率性/非确定性。

**核心要点**：
1. 评估与监控必要性：Agent 是概率性系统，传统"通过/失败"测试不够，要持续测效能/效率/合规性，防概念漂移。
2. 关键指标：响应准确性(精确匹配→语义相似→嵌入余弦→LLM-as-a-Judge→RAG忠实性)、延迟、Token 用量、Agent 轨迹(trajectory)。
3. 评估方法对比：人工(细但不可扩) / LLM-as-a-Judge(一致可扩但可能忽略中间步) / 自动化指标(客观但受限)。
4. 轨迹评估：检查步骤序列(工具选择/策略)，比较方式有精确匹配/按序/任意顺序/精确度/召回率。
5. 多 Agent 评估：合作有效性、计划坚持度、任务-智能体匹配。
6. 进阶"承包商(Contractor)模式"：把松散提示词变正式可协商可验证合约。

**代码做了什么**：对照演示两种评估——① 朴素精确匹配（字面相等才给 1.0，处理不了释义）；② 用 DeepSeek 当 LLM-as-a-Judge，输出 1-5 分 + 理由（语义等价即可给高分），并回传 token 用量做资源监控。demo 中"Paris is the capital"两种字面不同但语义相同的答案，朴素匹配误判 0 分，LLM 评判给高分。

**与其他章节关联**：LLM-as-a-Judge 的思想在 CH4 反思、CH11 达标判定中已用到；token 监控与 CH16 资源感知呼应。

**常见坑**：朴素精确匹配会误杀释义相同的好答案；LLM 评委也可能有偏，关键指标建议人工抽检校准。

```python
# CH19 示意：LLM-as-a-Judge
judge = deepseek_judge(question, agent_answer)  # 输出 {"overall_score":1-5,"rationale":...}
# 对比朴素匹配：expected=="The capital of France is Paris." vs 实际 "Paris is the capital..."
```

---

### CH20 优先级排序（Prioritization）

**模式一句话定义**：Agent 按重要性/紧急性/依赖关系/成本收益给任务排序，把精力放在最关键的事上。

**核心要点**：
1. 优先级排序 = 按重要性/紧急性/依赖/资源/成本收益给任务/目标排序，聚焦关键事。
2. 四要素：① 标准定义(紧急性/重要性/依赖/资源/成本收益/用户偏好) ② 任务评估(规则→LLM 评分) ③ 调度选择(队列/规划) ④ 动态重新优先级。
3. 排序可发生在三层：总体目标优先级 / 计划内步骤优先级 / 下一个即时动作选择。
4. 真实用例：客服(宕机>改密码)、云调度、自动驾驶(刹车>省油)、交易、项目管理看板、网安警报、个人助理。
5. 本质：让 Agent 从"被动执行脚本"变成"主动战略决策者"，类似人类团队的管理者。

**代码做了什么**：`Task` + `PriorityQueue`（按 P0/P1/P2 权重排序）。用 DeepSeek 给新任务定优先级，并演示"动态重新优先级"——突发"生产环境数据库宕机"强制 P0 插队后，自动顶到队列最前。

**与其他章节关联**：优先级调度是 CH2 路由（按类别分派）和 CH16 资源感知（按复杂度分配）的"重要性维度"补充；可嵌入 CH6 规划的步骤排序。

**常见坑**：优先级标准要可执行（别只写"重要"）；动态重优先级要有触发机制，否则突发高优事件会被埋没。

```python
# CH20 示意：LLM 定优先级 + 动态重排
pri = classify_priority(desc)     # 返回 P0/P1/P2
pq.add(Task(desc, pri))           # 按权重降序排
pq.add(Task("生产环境数据库宕机", "P0"))  # 突发高优插队顶前
```

---

### CH21 探索与发现（Exploration and Discovery）

**模式一句话定义**：Agent 主动进入陌生领域、尝试新方法、生成新知识/理解（区别于被动执行或已知空间内优化）。

**核心要点**：
1. 探索与发现 = Agent 主动进入陌生领域、尝试新方法、生成新知识（区别于被动执行或已知空间优化）。
2. 关键在"探索-利用困境"：在探索未知(可能发现更好方案)和利用已知(稳妥见效)间权衡。
3. Google 协同科学家(多 Agent)：生成(假设)→反思(同行评审)→排名(Elo 锦标赛)→进化(精炼)→邻近(聚类)→元审查。
4. Agent Laboratory：文献综述→实验→报告→知识共享(AgentRxiv)，用专门角色自动化科研。
5. 应用：科研假设生成、游戏策略发现(AlphaGo)、市场趋势、安全漏洞探测、创意内容、个性化教学。
6. 定位：增强而非替代人类；"科学家在环"协作；继承底层 LLM 局限，需安全护栏防滥用。

**代码做了什么**：迷你可跑 demo——给定问题 → 生成多个候选假设（探索，temperature=0.8 提高多样性）→ 批评者打新颖性/质量分 → 按分排名 → 对头部假设"进化"精炼。对应 Google 协同科学家的生成→排名→进化小循环。

**与其他章节关联**：探索循环复用了 CH4 反思（批评）、CH7 多 Agent（多视角评审）、CH20 排名；探索产出的假设可用 CH18 护栏防滥用。

**常见坑**：探索阶段要调高 temperature 才有多样性；探索-利用要平衡，别一直乱试；产出仍受底层 LLM 幻觉/知识截止限制。

```python
# CH21 示意：生成 → 批评排名 → 进化
hyps = generate_hypotheses(problem, n=3)   # 探索（高随机性）
scored = [(h, criticize(h)) for h in hyps] # 打分排名
evolved = llm.invoke(f"基于假设改进：{top}") # 进化精炼
```

---

## 四、运行环境说明

### 1. Python 虚拟环境
项目使用 `uv` 管理的虚拟环境（`.venv`）。建议先激活：
```bash
uv venv        # 或在项目根目录用已有 .venv
source .venv/bin/activate
```

### 2. 依赖安装
大部分章节共用：
```bash
pip install langchain-openai python-dotenv
```
部分章节需要额外依赖：
- **CH10（MCP）**：`pip install fastmcp mcp`
- **CH14（RAG）**：本文件用纯 Python TF-IDF，零额外依赖；若换真 embedding 需装向量库
- 旧代码（已注释）涉及的 `crewai` / `google-adk` / `langgraph` / `weaviate` 等**无需安装**，仅作对比参考

### 3. `.env` 环境变量
所有新代码统一从**项目根目录的 `.env`** 加载（代码中 `Path(__file__).resolve().parent.parent / ".env"`），需配置：
```env
LEARN_AI_KEY=你的DeepSeek_API_Key
```
并在代码中以 `base_url="https://api.deepseek.com"` 指向 DeepSeek（OpenAI 兼容接口）。原书基于 Gemini/OpenAI 的写法已整段注释保留在"旧代码"块中，供对比学习。

### 4. 文件落盘产物
- **CH8**：运行后会生成 `ch8_longterm_memory.json`，持久化用户长期记忆（跨会话保留）。
- **CH11**：达标后会落盘生成 `<用例slug>_<随机>.py` 文件（代码产物）。
- 其余章节为一次性演示，不写文件（CH10 的 MCP 用 in-memory 传输，CH15 的 A2A 用进程内总线）。

### 5. 模型说明
全书统一用 `deepseek-v4-flash`（temperature 因章节而异：工具调用用 0 求确定，探索/自一致性用 0.7~0.8 求多样）。**新代码 = DeepSeek 当前生效；旧代码（Google ADK / OpenAI / CrewAI 等）= 注释保留作对比**。

---

## 五、学习建议

### 推荐阅读顺序
章节本身已基本按**递进关系**排列，建议顺着编号读：

1. **基础编排（CH1–CH4）**：先理解"怎么把任务串起来"——顺序链(CH1) → 动态路由(CH2) → 并行(CH3) → 反思提质(CH4)。
2. **迈向真 Agent（CH5–CH6）**：工具调用(CH5)是分水岭，让 Agent 能动手；规划(CH6)让 Agent 自主拆目标。
3. **扩展能力（CH7–CH8）**：多 Agent 协作(CH7)、记忆管理(CH8)让 Agent 能分工、有记忆；CH9 学习与适应以原理为主，可穿插阅读。
4. **连接外部（CH10–CH14）**：MCP 标准协议(CH10)、目标监控(CH11)、异常恢复(CH12)、人机协同(CH13)、RAG(CH14)。
5. **系统化与高级（CH15–CH21）**：Agent 间通信(CH15)、资源优化(CH16)、推理技术(CH17)、安全护栏(CH18)、评估监控(CH19)、优先级(CH20)、探索发现(CH21)。

### 给小白的上手建议
- **先跑通一个**：从 CH1 配好 `.env` 跑通，再逐个章节运行感受差异，比纯读文档快得多。
- **重点看文件开头的注释块**：每个 `CHx-*.py` 前 ~50 行都是该章核心要点 + 新旧代码对照，是最好入口。
- **理解"新 vs 旧"**：新代码统一 DeepSeek 可跑；旧代码（Gemini/OpenAI）注释保留，帮你理解原书语境，但别直接跑。
- **抓主线模式**：CH1–CH7 是"编排七模式"，CH8/CH10/CH14 是"连接外部"，CH11–CH13/CH18 是"可靠性与安全"，CH15–CH21 是"系统化进阶"——分块记忆更高效。
- **动手改参数**：比如把 CH4 的 `max_iterations`、CH21 的 `temperature` 改一改，看行为变化，比看文字印象深。

---

> 📌 仓库实际收录 CH1–CH21 各章内容；其中 CH9 为概念性章节（无独立代码文件，仅原理说明），其余 CH1–CH8、CH10–CH21 均附可运行代码。本文档按实际内容编写，未虚构任何章节。
