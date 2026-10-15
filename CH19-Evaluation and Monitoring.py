## ============ 第19章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 19: Evaluation and Monitoring）============
# 1. 评估与监控必要性：Agent 是概率性/非确定性系统，传统"通过/失败"测试不够；要持续测效能、效率、合规性，防概念漂移/退化。
# 2. 关键指标：响应准确性(精确匹配→语义相似→嵌入余弦→LLM-as-a-Judge→RAG忠实性/相关性)、延迟、Token 用量、Agent 轨迹(trajectory)。
# 3. 评估方法对比：人工(细但不可扩) / LLM-as-a-Judge(一致可扩但可能忽略中间步) / 自动化指标(客观但受限)。
# 4. 轨迹评估：检查步骤序列(工具选择/策略)，比较方式有精确匹配、按序、任意顺序、精确度、召回率。
# 5. 多 Agent 评估：合作有效性、计划坚持度、任务-智能体匹配、加 Agent 是否提升整体。
# 6. 进阶"承包商(Contractor)模式"：把松散提示词变正式可协商可验证合约(AI 变可问责系统)。Google ADK 提供 web/pytest/cli 三种评估入口。

## ============ 旧代码（Google Generative AI / ADK，已注释，仅供对比）============
# 原书① evaluate_response_accuracy 用不区分大小写字符串精确匹配返回 1.0/0.0（无法处理释义）。
# 原书② LLMJudgeForLegalSurvey 用 google.generativeai(Gemini) 按法律评分标准返回 JSON 评判。
# 原书③ ADK 评估：adk web / AgentEvaluator.evaluate(pytest) / adk eval 跑测试集(JSON: 回合含查询/预期工具调用/响应)。
# def evaluate_response_accuracy(expected, actual):
#     return 1.0 if expected.strip().lower() == actual.strip().lower() else 0.0

## ============ 新代码（DeepSeek，当前生效）============
# 这里复刻第19章核心：① 朴素精确匹配(对照其局限) ② DeepSeek 当 LLM-as-a-Judge(语义评判) ③ Token 用量监控。
import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # 加载 myProject/.env 中的 LEARN_AI_KEY

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

if not os.getenv("LEARN_AI_KEY"):
    raise ValueError("在 .env 文件中未找到 LEARN_AI_KEY。请添加它。")
try:
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=0,
        api_key=os.getenv("LEARN_AI_KEY"),
        base_url="https://api.deepseek.com",
    )
    print(f"语言模型已初始化: {getattr(llm, 'model', 'deepseek-v4-flash')}")
except Exception as e:
    print(f"初始化语言模型时出错: {e}")
    llm = None


## --- ① 朴素精确匹配（书里指出它处理不了释义）---
def evaluate_response_accuracy(expected: str, actual: str) -> float:
    return 1.0 if expected.strip().lower() == actual.strip().lower() else 0.0


## --- ② LLM-as-a-Judge：用 DeepSeek 做语义级评判（对应评估监控设计模式）---
def deepseek_judge(question: str, agent_answer: str) -> dict:
    """让 DeepSeek 当评委，输出 1-5 分 + 理由；同时回传 token 用量(资源监控)。"""
    if not llm:
        return {"overall_score": None, "rationale": "(LLM 未初始化)", "token_usage": 0}
    prompt = f"""你是严格的评估专家。请评估智能体回答对问题的质量（语义等价即可给高分，不要求字面一致）。
问题: {question}
回答: {agent_answer}
仅输出 JSON：{{"overall_score": 1到5的整数, "rationale": "简短理由"}}"""
    r = llm.invoke([SystemMessage(content="你是公正的评审，只输出 JSON。"),
                    HumanMessage(content=prompt)])
    data = {}
    try:
        data = json.loads(r.content)
    except Exception:
        m = re.search(r"\{.*\}", r.content, re.S)
        if m:
            try: data = json.loads(m.group(0))
            except Exception: pass
    data["token_usage"] = getattr(getattr(r, "usage", None), "total_tokens", 0)
    return data


## --- 主流程：同一条答案，朴素匹配 vs LLM 评判 对照（第19章精髓）---
def evaluate_demo(question: str, expected: str, agent_answer: str):
    naive = evaluate_response_accuracy(expected, agent_answer)
    judge = deepseek_judge(question, agent_answer)
    print(f"\n>>> 问题：{question}")
    print(f">>> Agent 回答：{agent_answer}")
    print(f"   📏 朴素精确匹配得分：{naive}  （预期答案='{expected}'）")
    print(f"   🧑‍⚖️ LLM-as-a-Judge：{judge.get('overall_score')} 分 | {judge.get('rationale')}")
    print(f"   💡 监控：本次评判消耗 token = {judge.get('token_usage')}")


if __name__ == "__main__":
    q = "What is the capital of France?"
    # 两个语义相同但字面不同的回答：朴素匹配会误判后一个为 0，LLM 评判会给高分
    evaluate_demo(q, "The capital of France is Paris.", "The capital of France is Paris.")
    evaluate_demo(q, "The capital of France is Paris.", "Paris is the capital of France.")
