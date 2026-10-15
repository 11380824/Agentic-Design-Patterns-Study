## ============ 第21章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 21: Exploration and Discovery）============
# 1. 探索与发现 = Agent 主动进入陌生领域、尝试新方法、生成新知识/理解（区别于被动执行或已知空间内优化）。
# 2. 关键在"探索-利用困境"：要在探索未知(可能发现更好方案)和利用已知(稳妥见效)间权衡。
# 3. Google 协同科学家(多 Agent)：生成(假设)→反思(同行评审)→排名(Elo 锦标赛)→进化(精炼)→邻近(聚类)→元审查，模拟科研循环。
# 4. Agent Laboratory：文献综述→实验→报告撰写→知识共享(AgentRxiv)，用专门角色(教授/博士后/审稿人/ML工程师)自动化科研。
# 5. 应用：科研假设生成、游戏策略发现(AlphaGo)、市场趋势发现、安全漏洞探测、创意内容生成、个性化教学路径。
# 6. 定位：增强而非替代人类；"科学家在环"协作；继承底层 LLM 局限(幻觉/知识截止)，需安全护栏防滥用。

## ============ 旧代码（Agent Laboratory / MIT，已注释，仅供对比）============
# 原书 ReviewersAgent 用 3 个不同视角的 reviewer(gpt-4o-mini) 对 plan+report 打分，get_score 让模型按固定 JSON 模板
# 输出 Summary/Strengths/Weaknesses/Originality/Quality/.../Decision(Accept/Reject)。教授/博士后/审稿人 Agent 分工协作。
# class ReviewersAgent:
#     def inference(self, plan, report):
#         review_1 = get_score(..., reviewer_type="严格但公平的审稿人，期望好实验")
#         review_2 = get_score(..., reviewer_type="挑剔但公平，找有影响力想法")
#         review_3 = get_score(..., reviewer_type="思想开放，找新颖想法")
#         return f"审稿人#1:\n{review_1}\n审稿人#2:\n{review_2}\n审稿人#3:\n{review_3}"

## ============ 新代码（DeepSeek，当前生效）============
# 这里把第21章"探索-发现"落为迷你可跑 demo：给定问题 → 生成多个候选假设(探索) → 批评者打分(新颖性/质量)
# → 按分排名 → 对头部假设"进化"精炼。对应 Google 协同科学家的生成→排名→进化 小循环。
import os
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
        temperature=0.8,   # 探索阶段要高随机性，产生多样假设
        api_key=os.getenv("LEARN_AI_KEY"),
        base_url="https://api.deepseek.com",
    )
    print(f"语言模型已初始化: {getattr(llm, 'model', 'deepseek-v4-flash')}")
except Exception as e:
    print(f"初始化语言模型时出错: {e}")
    llm = None


## --- ① 生成：探索未知空间，产出多个候选假设 ---
def generate_hypotheses(problem: str, n: int = 3) -> list:
    print(f"  🌱 [生成] 针对问题探索 {n} 个候选假设...")
    r = llm.invoke([SystemMessage(content=f"你是研究助手。针对下面问题，给出 {n} 条不同的、有想象力的解决假设，每条一行，以 '- ' 开头。"),
                    HumanMessage(content=problem)]).content
    return [line[2:].strip() for line in r.splitlines() if line.strip().startswith("-")]


## --- ② 批评：对单条假设打 新颖性/质量(1-5) ---
def criticize(hypothesis: str) -> dict:
    r = llm.invoke([SystemMessage(content=(
        "你是严苛审稿人。对假设打分，仅输出 JSON：{\"novelty\":1-5, \"quality\":1-5, \"comment\":\"简短\"}")),
        HumanMessage(content=hypothesis)]).content
    try:
        import json
        d = json.loads(r)
        d["score"] = (d.get("novelty", 1) + d.get("quality", 1)) / 2
        return d
    except Exception:
        m = re.search(r"\{.*\}", r, re.S)
        if m:
            try:
                d = json.loads(m.group(0)); d["score"] = (d.get("novelty",1)+d.get("quality",1))/2; return d
            except Exception: pass
    return {"novelty": 1, "quality": 1, "comment": r[:40], "score": 1.0}


## --- ③ 排名 + ④ 进化：对头部假设精炼 ---
def rank_and_evolve(problem: str, hypotheses: list) -> list:
    scored = []
    for h in hypotheses:
        c = criticize(h)
        scored.append((h, c))
        print(f"     • 假设：{h}\n       评分：新颖{c.get('novelty')}/质量{c.get('quality')} → {c.get('comment')}")
    scored.sort(key=lambda x: x[1].get("score", 0), reverse=True)
    print("\n  🏆 [排名] 最优假设：", scored[0][0])
    # 进化：让 DeepSeek 在最优假设基础上精炼
    top = scored[0][0]
    evolved = llm.invoke([SystemMessage(content="你是进化智能体。基于下面已有假设，给出一个更具体、可落地的改进版本，一行。"),
                          HumanMessage(content=top)]).content.strip()
    print(f"  🧬 [进化] 精炼后：{evolved}")
    return scored


if __name__ == "__main__":
    problem = "如何显著延长 ESP32 导览设备的电池续航，同时不牺牲 GPS/BLE 触发体验？"
    print(">>> 探索问题：", problem)
    hyps = generate_hypotheses(problem, n=3)
    rank_and_evolve(problem, hyps)
