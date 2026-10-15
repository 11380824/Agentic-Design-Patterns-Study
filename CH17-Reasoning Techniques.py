## ============ 第17章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 17: Reasoning Techniques）============
# 1. 推理技术核心原则：在推理时"多分配计算"(延长思考时间)，小模型也能靠多步推理超过单次生成的大模型（推理扩展定律）。
# 2. 思维链(CoT)：让 LLM 显式输出中间推理步骤，提升透明度与准确率。
# 3. 思维树(ToT)：CoT 上分支多条路径，支持回溯/自我纠正；自我纠正(Self-Correction)：内部批判+迭代精炼。
# 4. 自一致性(Self-Consistency)：同一问题采样多条推理路径，多数投票选最终答案——用小算力换稳健。
# 5. PALM：LLM 生成并执行代码(如 Python)把精确计算外包给确定性环境。
# 6. ReAct(推理+行动)：把 CoT 与工具调用结合成 思考→行动→观察 的交错循环，是通往自主 Agent 的关键一跃（本章最基础模式）。
# 7. 其他：辩论链(CoD)/辩论图(GoD)多模型争论减偏、MASS 自动搜索多 Agent 拓扑、Deep Research 给 Agent"时间预算"多轮搜-推-综。

## ============ 旧代码（Google ADK PALM / LangGraph DeepSearch，已注释，仅供对比）============
# 原书① ADK 把 SearchAgent(google_search) + CodeAgent(BuiltInCodeExecutor) 组合成 RootAgent，体现 PALM(代码外包精确计算)。
# 原书② LangGraph 实现 DeepSearch：StateGraph(generate_query→web_research→reflection→finalize_answer)，
#    conditional_edges 在 reflection 后决定继续搜还是收尾——是 ReAct 式反思循环的工业版。
# builder = StateGraph(OverallState); builder.add_node("reflection", reflection)
# builder.add_conditional_edges("reflection", evaluate_research, ["web_research","finalize_answer"])

## ============ 新代码（DeepSeek，当前生效）============
# 这里用三个最小可跑 Demo 实证第17章：(A) CoT  (B) Self-Consistency 多数投票  (C) ReAct 思考/行动/观察循环。
import os
import re
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # 加载 myProject/.env 中的 LEARN_AI_KEY

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

if not os.getenv("LEARN_AI_KEY"):
    raise ValueError("在 .env 文件中未找到 LEARN_AI_KEY。请添加它。")
try:
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=0.7,   # 自一致性要靠温度>0 采不同路径
        api_key=os.getenv("LEARN_AI_KEY"),
        base_url="https://api.deepseek.com",
    )
    print(f"语言模型已初始化: {getattr(llm, 'model', 'deepseek-v4-flash')}")
except Exception as e:
    print(f"初始化语言模型时出错: {e}")
    llm = None


## --- (A) 思维链 CoT ---
def chain_of_thought(question: str) -> str:
    """演示 CoT：要求模型先逐步思考再给答案。"""
    print("  🧠 [CoT] 让模型逐步思考...")
    return llm.invoke([SystemMessage(content="请一步步思考(Chain-of-Thought)，最后用'最终答案：'给出结论。"),
                       HumanMessage(content=question)]).content


## --- (B) 自一致性 Self-Consistency ---
def _extract_final(text: str) -> str:
    m = re.search(r"最终答案[：:]\s*(.+)", text)
    return (m.group(1).strip() if m else text.strip())[-40:]

def self_consistency(question: str, samples: int = 5) -> str:
    """演示自一致性：采样多条推理路径，对最终答案做多数投票。"""
    print(f"  🔁 [Self-Consistency] 采样 {samples} 条路径并多数投票...")
    votes = Counter()
    for _ in range(samples):
        ans = chain_of_thought(question)
        votes[_extract_final(ans)] += 1
    return votes.most_common(1)[0][0]


## --- (C) ReAct：思考 → 行动 → 观察 循环 ---
# 两个"假工具"，让 Agent 能调用外部确定性能力（对应 ReAct 的 Action/Observation）
KB = {
    "rag": "RAG 是检索增强生成，先检索外部知识再生成以减少幻觉。",
    "mcp": "MCP 是模型上下文协议，标准化连接外部工具与数据。",
}
def search(keyword: str) -> str:
    return KB.get(keyword.lower(), f"未找到关于 {keyword} 的资料")
def calculate(expr: str) -> str:
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))  # 受限 eval，仅做四则运算
    except Exception as e:
        return f"计算错误: {e}"

TOOLS = {"search": search, "calculate": calculate}

def react_loop(question: str, max_steps: int = 4) -> str:
    """演示 ReAct：解析 Thought/Action/Observation，循环最多 max_steps 轮。"""
    print("  🔄 [ReAct] 思考→行动→观察 循环...")
    sys = SystemMessage(content=(
        "你是一个 ReAct Agent。严格按以下格式输出，每轮最多一个 Action：\n"
        "Thought: <你的思考>\n"
        "Action: <工具名>(<参数>)\n"
        "可用工具：search(关键词) 查知识库；calculate(算式) 做四则运算。\n"
        "当你有足够信息时，输出：Thought: 已完成\\nFinal Answer: <答案>"))
    conv = [sys, HumanMessage(content=question)]
    for step in range(max_steps):
        out = llm.invoke(conv).content
        print(f"    【第{step+1}轮】\n{out}")
        if "Final Answer:" in out:
            return out.split("Final Answer:")[-1].strip()
        m = re.search(r"Action:\s*(\w+)\((.*?)\)", out, re.S)
        if m:
            name, arg = m.group(1).strip(), m.group(2).strip().strip('"\'')
            obs = TOOLS.get(name, lambda a: "未知工具")(arg)
            print(f"    → Observation: {obs}")
            conv.append(HumanMessage(content=f"Observation: {obs}"))
        else:
            conv.append(HumanMessage(content="请继续，按格式给出 Action 或 Final Answer。"))
    return "（达到最大步数，未能收敛）"


if __name__ == "__main__":
    q = "RAG 和 MCP 有什么区别？另外帮我算一下 (8 + 6) * 3 等于多少。"
    print(">>> 问题：", q)
    print("\n===== A. 思维链 CoT =====")
    print(chain_of_thought(q))
    print("\n===== B. 自一致性 Self-Consistency =====")
    print("多数投票结果：", self_consistency("法国的首都是哪里？", samples=5))
    print("\n===== C. ReAct 循环 =====")
    print("ReAct 最终：", react_loop(q))
