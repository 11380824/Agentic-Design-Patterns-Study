# Agentic Design Patterns 学习指南（DeepSeek + LangChain 复现）

这是《Agentic Design Patterns》（作者 adp.xindoo，原书站点 adp.xindoo.xyz）配套代码的中文学习笔记与可运行示例；原书多基于 Gemini/OpenAI，本仓库统一用 DeepSeek + LangChain 复现，并保留原书 Google ADK 等旧代码作对比注释。

📘 完整分章详解见：[Agentic Design Patterns 学习指南.md](Agentic Design Patterns 学习指南.md)

## 目录总览

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
| CH9 | 学习与适应 / Learning and Adaptation | 让 Agent 从经验中持续改变策略与行为，由写死逻辑进化为动态自我提升系统（**概念性章节，无独立代码**） |
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

## 快速开始

- **环境**：Python 虚拟环境（项目使用 `uv` 管理的 `.venv`）
  ```bash
  uv venv
  source .venv/bin/activate
  ```
- **安装依赖**（大部分章节共用）：
  ```bash
  pip install langchain-openai python-dotenv
  ```
  CH10 还需：`pip install fastmcp mcp`
- **配置**：在项目根目录的 `.env` 中填入你的 DeepSeek 密钥，代码统一以 `base_url="https://api.deepseek.com"` 指向 DeepSeek（OpenAI 兼容接口）：
  ```env
  LEARN_AI_KEY=你的DeepSeek_API_Key
  ```
- **运行**示例（新代码统一走 DeepSeek）：
  ```bash
  python "CH1-Prompt Chaining.py"
  ```
- **注意**：`.env` 已被 `.gitignore` 忽略，不会上传；CH8 运行会生成本地 JSON 长期记忆文件，CH11 会落盘生成的 `.py`，均为运行时产物，非源码。

## 章节说明

21 章按「由基础到系统化」递进排列：

- **基础编排（CH1–CH4）**：顺序链 → 动态路由 → 并行 → 反思提质。
- **迈向真 Agent（CH5–CH6）**：工具调用是分水岭（让 Agent 能动手），规划让 Agent 自主拆目标。
- **扩展能力（CH7–CH8）**：多 Agent 协作、记忆管理让 Agent 能分工、有记忆。
- **连接外部（CH10–CH14）**：MCP 标准协议、目标监控、异常恢复、人机协同、RAG。
- **系统化与高级（CH15–CH21）**：Agent 间通信、资源优化、推理技术、安全护栏、评估监控、优先级、探索发现。

> 说明：**CH9（学习与适应）为概念性章节，本仓库未附独立代码文件，仅作原理说明**，故代码文件层面从 CH8 直接到 CH10；其余章节号除 CH9 外连续覆盖。

## 学习建议

- 先跑通一个：从 CH1 配好 `.env` 跑通，再逐个章节运行感受差异，比纯读文档快得多。
- 重点看每个 `CHx-*.py` 文件开头的注释块（核心要点 + 新旧代码对照），是最好入口。
- 理解「新 vs 旧」：新代码统一 DeepSeek 可跑；旧代码（Gemini/OpenAI/ADK）注释保留作对比，别直接跑。
- 动手改参数：如 CH4 的 `max_iterations`、CH21 的 `temperature`，看行为变化印象更深。

## 声明

本仓库内容整理自 adp.xindoo 的《Agentic Design Patterns》（原书站点 adp.xindoo.xyz），仅供学习交流使用。原书基于 Gemini/OpenAI，本仓库代码为统一用 **DeepSeek + LangChain** 复现的可运行示例，并保留原书相关旧代码作对比注释，版权归原作者所有。
